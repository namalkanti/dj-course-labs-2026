"""Pedagogical timing analysis implemented from scratch using NumPy and SciPy.

Implements the TimingBackend protocol using first-principles DSP math:
- Spectral flux: half-wave rectified STFT differencing with Hann window.
- Autocorrelation: Wiener-Khinchin theorem via FFT and IFFT.
- Beat tracking: 1D Ellis dynamic programming trellis and backtracking.
- Meter detection: autocorrelation harmonic ratio R(4*tau_0) / R(3*tau_0).
"""

import numpy as np

from dj_dsp.timing_data import (
    Meter,
    MeterResult,
    OnsetEnvelope,
    TempoResult,
    TemporalBeatAlignment,
    TimingBackend,
)
from dj_dsp.utils import (
    bpm_range_to_lag_bounds,
    classify_meter_from_autocorr,
    frames_to_times,
    lag_frames_to_bpm,
    log_compression,
)

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_MIN_BPM = 60.0
DEFAULT_MAX_BPM = 200.0
DEFAULT_PRIOR_BPM = 120.0
DEFAULT_PRIOR_STD_OCTAVES = 1.0
LOG_COMPRESSION_FACTOR = 100.0

# Meter Discrimination Constants
BAR_BEATS_THREE_FOUR = 3
BAR_BEATS_FOUR_FOUR = 4
LAG_SEARCH_TOLERANCE_FRACTION = 0.125  # +/- 1/8th beat window around nominal lag
AUTOCORR_SMOOTHING_SIGMA_FRAMES = 1.2  # Blurs ~1 frame to bridge fractional lags
NUMERICAL_EPSILON = 1e-9


def compute_onset_envelope(
    audio: np.ndarray,
    fs: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> OnsetEnvelope:
    """Compute onset strength envelope from scratch via STFT spectral flux."""
    padded = _pad_signal_centered(audio, n_fft=DEFAULT_N_FFT)
    frames = _segment_stft_windows(
        padded, frame_length=DEFAULT_N_FFT, hop_length=hop_length
    )
    mag_spec = _compute_magnitude_spectrogram(frames, n_fft=DEFAULT_N_FFT)
    flux = _compute_halfwave_rectified_flux(mag_spec)
    times = frames_to_times(len(flux), hop_length=hop_length, fs=fs)

    return OnsetEnvelope(
        values=flux.astype(np.float64),
        times=times.astype(np.float64),
        fs=fs,
        hop_length=hop_length,
    )


def estimate_tempo(
    onset_env: OnsetEnvelope,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> TempoResult:
    """Estimate global BPM and nominal beat interval via Wiener-Khinchin autocorrelation.

    Applies a standard log-normal prior centered at 120 BPM over the search bounds
    to mimic human tapping preference and resolve octave confusion (half-time/double-time).
    """
    autocorr = _autocorrelate_wiener_khinchin(onset_env.values)
    min_lag, max_lag = bpm_range_to_lag_bounds(min_bpm, max_bpm, onset_env.frame_rate)
    peak_lag = _find_dominant_lag_with_prior(
        autocorr, min_lag, max_lag, onset_env.frame_rate
    )
    bpm = lag_frames_to_bpm(peak_lag, onset_env.frame_rate)

    return TempoResult(bpm=bpm, beat_interval_frames=peak_lag)


def align_beats(
    onset_env: OnsetEnvelope,
    bpm: float | None = None,
) -> TemporalBeatAlignment:
    """Track dynamic beat locations via 1D Ellis DP trellis recursion and backtracking."""
    raise NotImplementedError("Stub: naive align_beats")


def detect_meter(
    onset_env: OnsetEnvelope,
    beat_interval_frames: int | None = None,
) -> MeterResult:
    """Classify 3/4 vs 4/4 meter by sampling autocorrelation at 3*tau_0 and 4*tau_0."""
    tau_0 = (
        beat_interval_frames
        if beat_interval_frames is not None
        else estimate_tempo(onset_env).beat_interval_frames
    )
    autocorr = _autocorrelate_wiener_khinchin(onset_env.values)
    return classify_meter_from_autocorr(
        autocorr=autocorr,
        signal_len=len(onset_env.values),
        beat_interval_frames=tau_0,
    )


NAIVE_BACKEND = TimingBackend(
    name="naive",
    compute_onset_envelope=compute_onset_envelope,
    estimate_tempo=estimate_tempo,
    align_beats=align_beats,
    detect_meter=detect_meter,
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _pad_signal_centered(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """Center-pad signal with reflection to match standard STFT window centering."""
    signal_f64 = np.asarray(signal, dtype=np.float64)
    pad_width = n_fft // 2
    return np.pad(signal_f64, pad_width, mode="reflect")


def _segment_stft_windows(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """Segment 1D signal into overlapping STFT analysis windows using stride tricks."""
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


def _compute_magnitude_spectrogram(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Apply Hann window and compute one-sided real FFT magnitude."""
    window = np.hanning(n_fft)
    spec = np.fft.rfft(frames * window, axis=-1)
    return np.abs(spec)


def _compute_halfwave_rectified_flux(mag_spec: np.ndarray) -> np.ndarray:
    """Compute half-wave rectified difference of log-compressed magnitude frames."""
    log_spec = log_compression(mag_spec, factor=LOG_COMPRESSION_FACTOR)
    diff = np.diff(log_spec, axis=0)
    rectified = np.maximum(0.0, diff)
    flux = np.sum(rectified, axis=-1)
    return np.pad(flux, (1, 0), mode="constant", constant_values=0.0)


def _autocorrelate_wiener_khinchin(signal: np.ndarray) -> np.ndarray:
    """Compute 1D autocorrelation for non-negative lags via Wiener-Khinchin FFT/IFFT."""
    signal_f64 = np.asarray(signal, dtype=np.float64)
    zero_mean_signal = signal_f64 - np.mean(signal_f64)

    # Pad to next power of 2 >= 2*N - 1 to prevent circular convolution
    n = len(zero_mean_signal)
    n_fft = 1 << (2 * n - 1).bit_length()

    freq_spectrum = np.fft.rfft(zero_mean_signal, n=n_fft)
    power_spectrum = np.abs(freq_spectrum) ** 2
    raw_autocorr = np.fft.irfft(power_spectrum, n=n_fft)

    return raw_autocorr[:n]


def _find_dominant_lag_with_prior(
    autocorr: np.ndarray,
    min_lag: int,
    max_lag: int,
    frame_rate: float,
    prior_bpm: float = DEFAULT_PRIOR_BPM,
    prior_std_octaves: float = DEFAULT_PRIOR_STD_OCTAVES,
) -> int:
    """Find dominant lag weighted by log-normal human tapping tempo prior."""
    bounded_min_lag = max(1, min_lag)
    bounded_max_lag = min(len(autocorr) - 1, max_lag)
    if bounded_min_lag >= bounded_max_lag:
        return bounded_min_lag

    candidate_lags = np.arange(bounded_min_lag, bounded_max_lag + 1)
    candidate_bpms = (60.0 * frame_rate) / candidate_lags
    octave_deviations = np.log2(candidate_bpms / prior_bpm) / prior_std_octaves
    prior_weights = np.exp(-0.5 * (octave_deviations**2))

    weighted_autocorr = autocorr[candidate_lags] * prior_weights
    return int(candidate_lags[np.argmax(weighted_autocorr)])

