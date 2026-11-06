"""Pure digital signal processing and numerical utilities for DJ DSP labs."""

from typing import Tuple
import numpy as np
from dj_dsp.timing_data import Meter, MeterResult
from scipy.ndimage import gaussian_filter1d

BAR_BEATS_THREE_FOUR = 3
BAR_BEATS_FOUR_FOUR = 4
DEFAULT_LAG_TOLERANCE_FRACTION = 0.15
DEFAULT_SMOOTHING_SIGMA_FRAMES = 1.2
NUMERICAL_EPSILON = 1e-9

# --- Conversions & Time-Frequency Indexing ---


def bpm_to_lag_frames(bpm: float, fs: int, hop_length: int) -> int:
    """Convert tempo in beats-per-minute to discrete autocorrelation lag in frames."""
    if bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm}")
    frame_rate = fs / hop_length
    return int(round((60.0 * frame_rate) / bpm))


def lag_frames_to_bpm(lag_frames: int, frame_rate: float) -> float:
    """Convert discrete lag frames back to beats-per-minute."""
    if lag_frames <= 0:
        raise ValueError(f"Lag frames must be positive, got {lag_frames}")
    return (60.0 * frame_rate) / lag_frames


def bpm_range_to_lag_bounds(
    min_bpm: float, max_bpm: float, frame_rate: float
) -> Tuple[int, int]:
    """Convert BPM range [min_bpm, max_bpm] to inclusive lag bounds [min_lag, max_lag]."""
    if min_bpm <= 0 or max_bpm <= 0:
        raise ValueError("BPM bounds must be positive")
    if min_bpm > max_bpm:
        raise ValueError(f"min_bpm ({min_bpm}) cannot exceed max_bpm ({max_bpm})")

    min_lag = int(round((60.0 * frame_rate) / max_bpm))
    max_lag = int(round((60.0 * frame_rate) / min_bpm))
    return min_lag, max_lag


def frames_to_times(num_frames: int, hop_length: int, fs: int) -> np.ndarray:
    """Compute physical timestamps in seconds for STFT/envelope frame indices."""
    return np.arange(num_frames) * (hop_length / fs)


# --- Signal Normalization & Dynamics ---


def normalize_peak(audio: np.ndarray, headroom_db: float = 0.0) -> np.ndarray:
    """Normalize signal peak amplitude to specified headroom below 0 dBFS."""
    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio
    target_peak = 10.0 ** (-abs(headroom_db) / 20.0)
    return (audio / peak) * target_peak


def log_compression(mag: np.ndarray, factor: float = 100.0) -> np.ndarray:
    """Apply log-dynamic range compression log(1 + factor * |X|) to magnitude frames."""
    return np.log1p(factor * mag)


# --- Autocorrelation & Smoothing ---


def smooth_and_normalize_autocorr(
    autocorr: np.ndarray,
    signal_len: int,
    sigma_frames: float = 1.2,
) -> np.ndarray:
    """Apply unbiased lag-normalization and 1D Gaussian smoothing to autocorrelation.

    Unbiased normalization compensates for the linear decay of finite overlapping frames (N - tau).
    Gaussian smoothing prevents fractional beat grid straddling from attenuating discrete peaks.
    """
    lags = np.arange(len(autocorr))
    overlap_counts = np.maximum(1, signal_len - lags)
    unbiased = autocorr / overlap_counts
    return gaussian_filter1d(unbiased, sigma=sigma_frames)


def find_peak_in_lag_window(
    autocorr: np.ndarray,
    target_lag: int,
    tolerance_frames: int,
) -> float:
    """Find maximum autocorrelation peak value within a local tolerance window around target lag."""
    window_start = max(0, target_lag - tolerance_frames)
    window_end = min(len(autocorr), target_lag + tolerance_frames + 1)
    if window_start >= window_end:
        return 0.0
    return float(np.max(autocorr[window_start:window_end]))


def calculate_normalized_ratio_confidence(ratio: float) -> float:
    """Calculate bounded confidence score [0.0, 1.0] from meter ratio |r - 1| / (r + 1)."""
    if ratio <= 0.0:
        return 0.0
    return abs(ratio - 1.0) / (ratio + 1.0)


def classify_meter_from_autocorr(
    autocorr: np.ndarray,
    signal_len: int,
    beat_interval_frames: int,
    tolerance_fraction: float = DEFAULT_LAG_TOLERANCE_FRACTION,
    smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA_FRAMES,
) -> MeterResult:
    """Sample autocorrelation at 3-beat and 4-beat bar lags to classify 3/4 vs 4/4 meter.

    Applies unbiased normalization and 1D Gaussian smoothing to avoid lag-decay
    and grid-straddling attenuation artifacts.
    """
    smoothed = smooth_and_normalize_autocorr(
        autocorr, signal_len=signal_len, sigma_frames=smoothing_sigma
    )
    search_radius = max(1, int(round(beat_interval_frames * tolerance_fraction)))

    prominence_3_beat = find_peak_in_lag_window(
        smoothed,
        target_lag=BAR_BEATS_THREE_FOUR * beat_interval_frames,
        tolerance_frames=search_radius,
    )
    prominence_4_beat = find_peak_in_lag_window(
        smoothed,
        target_lag=BAR_BEATS_FOUR_FOUR * beat_interval_frames,
        tolerance_frames=search_radius,
    )

    ratio = (prominence_4_beat + NUMERICAL_EPSILON) / (
        prominence_3_beat + NUMERICAL_EPSILON
    )
    meter = Meter.FOUR_FOUR if ratio >= 1.0 else Meter.THREE_FOUR
    confidence = calculate_normalized_ratio_confidence(ratio)

    return MeterResult(meter=meter, ratio=ratio, confidence=confidence)
