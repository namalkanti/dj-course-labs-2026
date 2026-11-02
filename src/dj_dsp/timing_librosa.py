"""Production timing analysis using canonical MIR libraries (librosa).

Implements the TimingBackend protocol using librosa.onset, librosa.feature.tempo,
and librosa.beat.beat_track.
"""

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dj_dsp.timing_data import (
    Meter,
    MeterResult,
    OnsetEnvelope,
    TempoResult,
    TemporalBeatAlignment,
    TimingBackend,
)

DEFAULT_HOP_LENGTH = 512
DEFAULT_MIN_BPM = 60.0
DEFAULT_MAX_BPM = 200.0

# Meter Discrimination Constants
BAR_BEATS_THREE_FOUR = 3
BAR_BEATS_FOUR_FOUR = 4
MAX_EVALUATION_BEATS = 5
LAG_SEARCH_TOLERANCE_FRACTION = 0.125  # +/- 1/8th beat window around nominal lag
AUTOCORR_SMOOTHING_SIGMA_FRAMES = 1.2  # Blurs ~1 frame to bridge fractional lags
NUMERICAL_EPSILON = 1e-9


def compute_onset_envelope(
    audio: np.ndarray,
    fs: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> OnsetEnvelope:
    """Compute onset strength envelope using librosa.

    Note: Maps internal audio -> librosa 'y' and fs -> librosa 'sr'.
    """
    onset_vals = librosa.onset.onset_strength(y=audio, sr=fs, hop_length=hop_length)
    times = librosa.times_like(onset_vals, sr=fs, hop_length=hop_length)

    return OnsetEnvelope(
        values=onset_vals.astype(np.float64),
        times=times.astype(np.float64),
        fs=fs,
        hop_length=hop_length,
    )


def estimate_tempo(
    onset_env: OnsetEnvelope,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> TempoResult:
    """Estimate global BPM and nominal beat interval using librosa."""
    bpm_arr = librosa.feature.tempo(
        onset_envelope=onset_env.values,
        sr=onset_env.fs,
        hop_length=onset_env.hop_length,
        max_tempo=max_bpm,
    )
    bpm = float(np.atleast_1d(bpm_arr)[0])
    beat_interval_frames = _bpm_to_beat_interval_frames(bpm, onset_env.frame_rate)

    return TempoResult(bpm=bpm, beat_interval_frames=beat_interval_frames)


def align_beats(
    onset_env: OnsetEnvelope,
    bpm: float | None = None,
) -> TemporalBeatAlignment:
    """Align dynamic beat locations across time using librosa.beat.beat_track."""
    target_bpm = bpm if bpm is not None else estimate_tempo(onset_env).bpm

    detected_bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env.values,
        sr=onset_env.fs,
        hop_length=onset_env.hop_length,
        bpm=target_bpm,
        units="frames",
    )
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(
        beat_frames, sr=onset_env.fs, hop_length=onset_env.hop_length
    )
    effective_bpm = _derive_effective_bpm(
        beat_times, fallback_bpm=float(np.atleast_1d(detected_bpm)[0])
    )

    return TemporalBeatAlignment(
        beat_frames=beat_frames,
        beat_times=beat_times,
        bpm=effective_bpm,
    )


def detect_meter(
    onset_env: OnsetEnvelope,
    beat_interval_frames: int | None = None,
) -> MeterResult:
    """Classify 3/4 vs 4/4 meter by sampling autocorrelation harmonics."""
    if beat_interval_frames is None:
        beat_interval_frames = estimate_tempo(onset_env).beat_interval_frames

    return _compute_meter_from_autocorr(onset_env.values, beat_interval_frames)


LIBROSA_BACKEND = TimingBackend(
    name="librosa",
    compute_onset_envelope=compute_onset_envelope,
    estimate_tempo=estimate_tempo,
    align_beats=align_beats,
    detect_meter=detect_meter,
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _bpm_to_beat_interval_frames(bpm: float, frame_rate: float) -> int:
    """Convert tempo in BPM to nominal inter-beat lag in envelope frames."""
    seconds_per_beat = 60.0 / bpm
    return int(round(seconds_per_beat * frame_rate))


def _derive_effective_bpm(beat_times: np.ndarray, fallback_bpm: float) -> float:
    """Derive effective global BPM from median inter-beat interval."""
    if len(beat_times) < 2:
        return fallback_bpm
    diffs = np.diff(beat_times)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return fallback_bpm
    median_ibi = float(np.median(positive_diffs))
    return 60.0 / median_ibi if median_ibi > 0 else fallback_bpm


def _bound_max_autocorr_lag(
    total_frames: int,
    beat_interval_frames: int,
    max_beats: int = MAX_EVALUATION_BEATS,
) -> int:
    """Bound evaluation lag to max_beats, preventing out-of-bounds on short audio."""
    target_lag_frames = int(round(max_beats * beat_interval_frames))
    return min(total_frames, target_lag_frames)


def _find_peak_in_lag_window(
    autocorr: np.ndarray,
    center_lag: int,
    radius: int,
) -> float:
    """Find maximum autocorrelation value within a window centered at center_lag."""
    start_idx = max(0, center_lag - radius)
    end_idx = min(len(autocorr), center_lag + radius + 1)
    if start_idx >= len(autocorr) or start_idx >= end_idx:
        return 0.0
    return float(np.max(autocorr[start_idx:end_idx]))


def _calculate_normalized_ratio_confidence(ratio: float) -> float:
    """Compute symmetric confidence in [0, 1] based on distance of ratio from 1.0."""
    relative_difference = abs(ratio - 1.0) / (ratio + 1.0)
    return float(np.clip(relative_difference, 0.0, 1.0))


def _normalize_and_smooth_autocorr(
    autocorr: np.ndarray,
    total_frames: int,
    smoothing_sigma: float = AUTOCORR_SMOOTHING_SIGMA_FRAMES,
) -> np.ndarray:
    """Normalize raw autocorrelation by overlap count and smooth fractional lags.

    Why this is required:
    1. Unbiased Normalization: Raw autocorrelation sums (total_frames - lag) terms.
       Without dividing by overlap count, longer lags naturally decay toward zero,
       penalizing 4-beat lag relative to 3-beat lag.
    2. Fractional Lag Smoothing: Tempos often yield non-integer frame intervals
       (e.g., 128 BPM has 40.375 frames/beat, placing 4 beats at 161.5 frames).
       A mild Gaussian blur recombines energy split across adjacent integer bins.
    """
    lags = np.arange(len(autocorr))
    overlap_counts = total_frames - lags
    unbiased_autocorr = autocorr / overlap_counts
    return gaussian_filter1d(unbiased_autocorr, sigma=smoothing_sigma)


def _compute_meter_from_autocorr(
    onset_values: np.ndarray,
    beat_interval_frames: int,
) -> MeterResult:
    """Sample onset autocorrelation at 3-beat vs 4-beat bar lags to classify meter."""
    max_lag = _bound_max_autocorr_lag(len(onset_values), beat_interval_frames)
    autocorr = librosa.autocorrelate(onset_values, max_size=max_lag)
    smoothed_autocorr = _normalize_and_smooth_autocorr(autocorr, len(onset_values))

    search_radius = max(
        1, int(round(beat_interval_frames * LAG_SEARCH_TOLERANCE_FRACTION))
    )
    prominence_3_beat = _find_peak_in_lag_window(
        smoothed_autocorr,
        center_lag=BAR_BEATS_THREE_FOUR * beat_interval_frames,
        radius=search_radius,
    )
    prominence_4_beat = _find_peak_in_lag_window(
        smoothed_autocorr,
        center_lag=BAR_BEATS_FOUR_FOUR * beat_interval_frames,
        radius=search_radius,
    )

    ratio = (prominence_4_beat + NUMERICAL_EPSILON) / (
        prominence_3_beat + NUMERICAL_EPSILON
    )
    meter = Meter.FOUR_FOUR if ratio >= 1.0 else Meter.THREE_FOUR
    confidence = _calculate_normalized_ratio_confidence(ratio)

    return MeterResult(meter=meter, ratio=ratio, confidence=confidence)
