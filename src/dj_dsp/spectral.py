"""Pure functional DSP transformations producing spectral lenses/views.

All functions are pure: they accept NumPy audio arrays and sampling parameters,
and return typed, frozen view dataclasses defined in dj_dsp.spectral_data:
- compute_welch_spectrum -> SpectrumView
- compute_stft_spectrogram -> SpectrogramView
- compute_three_band_waveform -> ThreeBandWaveformView
- analyze_spectral_cutoff -> CutoffDetectionAnalysis
"""

from typing import Optional, Tuple
import numpy as np
import scipy.signal as signal

from dj_dsp.spectral_data import (
    CutoffDetectionAnalysis,
    SpectrogramView,
    SpectrumView,
    ThreeBandWaveformView,
)

SILENCE_THRESHOLD_DBFS = -85.0
INSUFFICIENT_HF_ENERGY_DBFS = -75.0
PASSBAND_REF_F_MIN_HZ = 11000.0
PASSBAND_REF_F_MAX_HZ = 14000.0
TEST_BAND_START_HZ = 14000.0
ROLLOFF_DROP_THRESHOLD_DB = 22.0
STOPBAND_TRANSITION_OFFSET_HZ = 1200.0
TRANSCODE_CUTOFF_CEILING_HZ = 19500.0
TRANSCODE_MIN_ATTENUATION_DB = 28.0


def compute_welch_spectrum(
    audio: np.ndarray,
    fs: int = 44100,
    window_length: int = 4096,
    overlap_length: Optional[int] = None,
) -> SpectrumView:
    """Compute Welch's Power Spectral Density (PSD) estimate across discrete frequencies."""
    if overlap_length is None:
        overlap_length = window_length // 2

    actual_nperseg = min(window_length, len(audio))
    actual_noverlap = min(overlap_length, actual_nperseg - 1)

    frequencies, psd = signal.welch(
        audio,
        fs=fs,
        window="hann",
        nperseg=actual_nperseg,
        noverlap=actual_noverlap,
        scaling="spectrum",
    )

    power_db = 10.0 * np.log10(psd + 1e-12)

    return SpectrumView(
        frequencies=frequencies.astype(np.float64),
        power_db=power_db.astype(np.float64),
    )


def compute_stft_spectrogram(
    audio: np.ndarray,
    fs: int = 44100,
    window_length: int = 2048,
    overlap_length: Optional[int] = None,
    dynamic_range_db: float = 80.0,
) -> SpectrogramView:
    """Compute 2D Short-Time Fourier Transform (STFT) power spectrogram."""
    if overlap_length is None:
        overlap_length = (window_length * 3) // 4

    actual_nperseg = min(window_length, len(audio))
    actual_noverlap = min(overlap_length, actual_nperseg - 1)

    frequencies, times, zxx = signal.stft(
        audio,
        fs=fs,
        window="hann",
        nperseg=actual_nperseg,
        noverlap=actual_noverlap,
    )

    raw_power_db = 20.0 * np.log10(np.abs(zxx) + 1e-9)
    vmax = float(np.max(raw_power_db))
    vmin = vmax - dynamic_range_db
    power_db = np.clip(raw_power_db, vmin, vmax)

    return SpectrogramView(
        times=times.astype(np.float64),
        frequencies=frequencies.astype(np.float64),
        power_db=power_db.astype(np.float64),
        dynamic_range_db=float(dynamic_range_db),
    )


def compute_three_band_waveform(
    audio: np.ndarray,
    fs: int = 44100,
    hop_length: int = 256,
    low_cutoff: float = 250.0,
    high_cutoff: float = 2500.0,
) -> ThreeBandWaveformView:
    """Compute Pioneer Rekordbox 3-band amplitude envelopes (CDJ-3000 style)."""
    sig_low, sig_mid, sig_high = filter_three_bands(audio, fs, low_cutoff, high_cutoff)
    rms_low, rms_mid, rms_high = compute_framed_rms_bands(
        [sig_low, sig_mid, sig_high], hop_length=hop_length
    )

    num_frames = len(rms_low)
    times = (np.arange(num_frames) * hop_length + (hop_length / 2.0)) / fs

    return ThreeBandWaveformView(
        times=times.astype(np.float64),
        low=rms_low.astype(np.float64),
        mid=rms_mid.astype(np.float64),
        high=rms_high.astype(np.float64),
    )


def analyze_spectral_cutoff(
    audio: np.ndarray,
    fs: int = 44100,
    window_length: int = 2048,
) -> CutoffDetectionAnalysis:
    """Inspect high-frequency roll-off using a bandwidth/attenuation heuristic."""
    nyquist = fs / 2.0
    view = compute_welch_spectrum(audio, fs=fs, window_length=window_length)
    freqs, power_db = view.frequencies, view.power_db

    peak_power = float(np.max(power_db))
    if peak_power < SILENCE_THRESHOLD_DBFS:
        return make_silent_cutoff_analysis(peak_power)

    ref_mask = (freqs >= PASSBAND_REF_F_MIN_HZ) & (freqs <= PASSBAND_REF_F_MAX_HZ)
    ref_db = float(np.mean(power_db[ref_mask])) if np.any(ref_mask) else peak_power
    if ref_db < INSUFFICIENT_HF_ENERGY_DBFS:
        return make_insufficient_energy_analysis(ref_db, nyquist=nyquist)

    test_mask = freqs >= TEST_BAND_START_HZ
    test_freqs, test_power = freqs[test_mask], power_db[test_mask]

    drop_threshold_db = ref_db - ROLLOFF_DROP_THRESHOLD_DB
    detected_cutoff = estimate_cutoff_frequency(test_freqs, test_power, drop_threshold_db, nyquist)
    noise_floor = measure_stopband_noise_floor(test_freqs, test_power, detected_cutoff)
    attenuation = float(ref_db - noise_floor)

    is_suspicious, verdict, confidence = classify_transcode_verdict(detected_cutoff, attenuation)

    return CutoffDetectionAnalysis(
        detected_cutoff_hz=float(detected_cutoff),
        peak_shelf_db=float(ref_db),
        noise_floor_db=float(noise_floor),
        stopband_attenuation_db=float(attenuation),
        verdict=verdict,
        is_suspicious_transcode=is_suspicious,
        confidence=float(confidence),
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def filter_three_bands(
    audio: np.ndarray, fs: int, low_cutoff: float, high_cutoff: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split audio into Low, Mid, High bands using 2nd-order Butterworth IIR filters."""
    nyquist = fs / 2.0
    safe_low = min(low_cutoff, nyquist * 0.95)
    safe_high = min(high_cutoff, nyquist * 0.98)

    sos_low = signal.butter(2, safe_low / nyquist, btype="lowpass", output="sos")
    sos_mid = signal.butter(
        2, [safe_low / nyquist, safe_high / nyquist], btype="bandpass", output="sos"
    )
    sos_high = signal.butter(2, safe_high / nyquist, btype="highpass", output="sos")

    return (
        signal.sosfilt(sos_low, audio),
        signal.sosfilt(sos_mid, audio),
        signal.sosfilt(sos_high, audio),
    )


def compute_framed_rms_bands(
    signals: list[np.ndarray], hop_length: int
) -> list[np.ndarray]:
    """Pad signals to full hops and compute frame-wise RMS power."""
    raw_len = len(signals[0])
    num_frames = int(np.ceil(raw_len / hop_length))
    pad_len = num_frames * hop_length - raw_len

    rms_results = []
    for sig in signals:
        padded = np.pad(sig, (0, pad_len)) if pad_len > 0 else sig
        frames = padded.reshape(num_frames, hop_length)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        rms_results.append(rms)

    return rms_results


def estimate_cutoff_frequency(
    freqs: np.ndarray, power: np.ndarray, drop_threshold_db: float, nyquist: float
) -> float:
    """Find the first frequency bin dropping below threshold with linear interpolation."""
    drop_indices = np.where(power <= drop_threshold_db)[0]
    if len(drop_indices) == 0:
        return float(nyquist)

    idx = int(drop_indices[0])
    if idx == 0:
        return float(freqs[0])

    f0, f1 = freqs[idx - 1], freqs[idx]
    p0, p1 = power[idx - 1], power[idx]
    frac = (drop_threshold_db - p0) / (p1 - p0) if abs(p1 - p0) > 1e-4 else 0.5
    return float(f0 + frac * (f1 - f0))


def measure_stopband_noise_floor(
    freqs: np.ndarray, power: np.ndarray, detected_cutoff_hz: float
) -> float:
    """Average power levels in stopband bins located beyond cutoff + offset."""
    stop_mask = freqs >= (detected_cutoff_hz + STOPBAND_TRANSITION_OFFSET_HZ)
    if np.any(stop_mask):
        return float(np.mean(power[stop_mask]))
    return float(power[-1]) if len(power) > 0 else 0.0


def classify_transcode_verdict(
    detected_cutoff_hz: float, attenuation_db: float
) -> Tuple[bool, str, float]:
    """Classify audio as full bandwidth vs suspicious lossy transcode."""
    is_suspicious = (detected_cutoff_hz < TRANSCODE_CUTOFF_CEILING_HZ) and (
        attenuation_db >= TRANSCODE_MIN_ATTENUATION_DB
    )
    if is_suspicious:
        return True, "Low-bitrate transcode detected (cut off below 19.5 kHz)", 0.90
    return False, "High quality / full bandwidth audio", 0.95


def make_silent_cutoff_analysis(peak_power: float) -> CutoffDetectionAnalysis:
    """Construct an analysis result for near-silent audio inputs."""
    return CutoffDetectionAnalysis(
        detected_cutoff_hz=0.0,
        peak_shelf_db=peak_power,
        noise_floor_db=peak_power,
        stopband_attenuation_db=0.0,
        verdict="Silent / Insufficient Audio Energy",
        is_suspicious_transcode=False,
        confidence=0.0,
    )


def make_insufficient_energy_analysis(ref_db: float, nyquist: float) -> CutoffDetectionAnalysis:
    """Construct an analysis result for audio lacking high-frequency energy."""
    return CutoffDetectionAnalysis(
        detected_cutoff_hz=float(nyquist),
        peak_shelf_db=ref_db,
        noise_floor_db=ref_db,
        stopband_attenuation_db=0.0,
        verdict="Insufficient high-frequency energy for forensic evaluation",
        is_suspicious_transcode=False,
        confidence=0.5,
    )
