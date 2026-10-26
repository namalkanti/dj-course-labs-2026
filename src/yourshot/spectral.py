"""Pure functional DSP transformations producing spectral lenses/views.

All functions are pure: they accept NumPy audio arrays and sampling parameters,
and return typed, frozen view dataclasses defined in yourshot.spectral_data:
- compute_welch_spectrum -> SpectrumView
- compute_stft_spectrogram -> SpectrogramView
- compute_three_band_waveform -> ThreeBandWaveformView
"""

from typing import Optional
import numpy as np
import scipy.signal as signal

from yourshot.spectral_data import (
    CutoffDetectionAnalysis,
    SpectrogramView,
    SpectrumView,
    ThreeBandWaveformView,
)


def compute_welch_spectrum(
    audio: np.ndarray,
    fs: int = 44100,
    window_length: int = 4096,
    overlap_length: Optional[int] = None,
) -> SpectrumView:
    """Compute Welch's Power Spectral Density (PSD) estimate across discrete frequencies.

    Partitions the audio into Hann-windowed segments with overlap, computes their
    periodograms, and averages them to minimize spectral variance.

    Args:
        audio: 1D NumPy array of audio samples (float32 or float64).
        fs: Sampling rate in Hz (defaults to 44100).
        window_length: FFT and window length L in samples (defaults to 4096).
        overlap_length: Number of overlapping samples between successive windows.
                        Defaults to window_length // 2 (50% overlap).

    Returns:
        SpectrumView with frequencies and power_db in dBFS.
    """
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
    """Compute 2D Short-Time Fourier Transform (STFT) power spectrogram.

    Args:
        audio: 1D NumPy array of audio samples.
        fs: Sampling rate in Hz (defaults to 44100).
        window_length: STFT window length L in samples (defaults to 2048).
        overlap_length: Number of overlapping samples (defaults to L * 3 // 4 = 1536).
        dynamic_range_db: Decibel display floor depth below signal peak (defaults to 80 dB).

    Returns:
        SpectrogramView with times, frequencies, and power_db clamped to [vmax - dynamic_range_db, vmax].
    """
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
    """Compute Pioneer Rekordbox 3-band amplitude envelopes (CDJ-3000 style).

    Splits audio using 2nd-order Butterworth crossover filterbanks into:
    - Low Band: f < low_cutoff (Bass, < 250 Hz)
    - Mid Band: low_cutoff <= f < high_cutoff (Midrange, 250 Hz - 2.5 kHz)
    - High Band: f >= high_cutoff (Treble, >= 2.5 kHz)

    Then performs block-wise RMS pooling over windows of hop_length samples.
    (Note: Standard Butterworth is optimal for envelope extraction; Linkwitz-Riley
    is only required when summing filtered bands back into audible audio, as in Week 3).

    Args:
        audio: 1D NumPy array of audio samples.
        fs: Sampling rate in Hz.
        hop_length: Window step in samples between successive display frames.
        low_cutoff: Crossover frequency between Bass and Mids in Hz (defaults to 250 Hz).
        high_cutoff: Crossover frequency between Mids and Treble in Hz (defaults to 2500 Hz).

    Returns:
        ThreeBandWaveformView with times, low, mid, and high envelopes.
    """
    nyquist = fs / 2.0
    safe_low = min(low_cutoff, nyquist * 0.95)
    safe_high = min(high_cutoff, nyquist * 0.98)

    sos_low = signal.butter(2, safe_low / nyquist, btype="lowpass", output="sos")
    sos_mid = signal.butter(
        2, [safe_low / nyquist, safe_high / nyquist], btype="bandpass", output="sos"
    )
    sos_high = signal.butter(2, safe_high / nyquist, btype="highpass", output="sos")

    sig_low = signal.sosfilt(sos_low, audio)
    sig_mid = signal.sosfilt(sos_mid, audio)
    sig_high = signal.sosfilt(sos_high, audio)

    num_frames = int(np.ceil(len(audio) / hop_length))
    pad_len = num_frames * hop_length - len(audio)

    if pad_len > 0:
        sig_low = np.pad(sig_low, (0, pad_len))
        sig_mid = np.pad(sig_mid, (0, pad_len))
        sig_high = np.pad(sig_high, (0, pad_len))

    low_frames = sig_low.reshape(num_frames, hop_length)
    mid_frames = sig_mid.reshape(num_frames, hop_length)
    high_frames = sig_high.reshape(num_frames, hop_length)

    rms_low = np.sqrt(np.mean(low_frames**2, axis=1))
    rms_mid = np.sqrt(np.mean(mid_frames**2, axis=1))
    rms_high = np.sqrt(np.mean(high_frames**2, axis=1))

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
    """Inspect high-frequency roll-off using a bandwidth/attenuation heuristic.

    Examines the top register (10 kHz to Nyquist) using Welch PSD. Low-pass
    patterns can be consistent with lossy encoding, mastering, or other filtering.
    Neither a cutoff nor its absence establishes bitrate or encoding history.
    The legacy confidence field contains fixed, uncalibrated heuristic scores,
    not probabilities of authenticity.

    Args:
        audio: 1D monophonic float array of audio samples.
        fs: Sampling rate in Hz (defaults to 44100).
        window_length: FFT window length in samples (defaults to 2048).

    Returns:
        CutoffDetectionAnalysis with detected cutoff frequency, stopband metrics, and verdict.
    """
    nyquist = fs / 2.0
    view = compute_welch_spectrum(audio, fs=fs, window_length=window_length)
    freqs = view.frequencies
    power_db = view.power_db

    # Edge case: Pure silence or DC offset
    peak_power = float(np.max(power_db))
    if peak_power < -85.0:
        return CutoffDetectionAnalysis(
            detected_cutoff_hz=0.0,
            peak_shelf_db=peak_power,
            noise_floor_db=peak_power,
            stopband_attenuation_db=0.0,
            verdict="Silent / Insufficient Audio Energy",
            is_suspicious_transcode=False,
            confidence=0.0,
        )

    # Reference band in the upper-mid passband: [11.0 kHz, 14.0 kHz]
    ref_mask = (freqs >= 11000.0) & (freqs <= 14000.0)
    ref_db = float(np.mean(power_db[ref_mask])) if np.any(ref_mask) else peak_power

    # Inspection candidate region: [14.0 kHz, Nyquist]
    test_mask = freqs >= 14000.0
    test_freqs = freqs[test_mask]
    test_power = power_db[test_mask]

    # Roll-off threshold: drop of 10 dB relative to passband reference
    # (Captures steep brick-wall lowpass transition before descending into noise floor)
    drop_threshold_db = ref_db - 10.0
    drop_indices = np.where(test_power <= drop_threshold_db)[0]

    if len(drop_indices) == 0:
        detected_cutoff = float(nyquist)
        noise_floor = float(np.min(test_power))
        attenuation = ref_db - noise_floor
        is_suspicious = False
    else:
        first_drop_idx = int(drop_indices[0])
        if first_drop_idx > 0:
            f0, f1 = test_freqs[first_drop_idx - 1], test_freqs[first_drop_idx]
            p0, p1 = test_power[first_drop_idx - 1], test_power[first_drop_idx]
            frac = (drop_threshold_db - p0) / (p1 - p0) if abs(p1 - p0) > 1e-4 else 0.5
            detected_cutoff = float(f0 + frac * (f1 - f0))
        else:
            detected_cutoff = float(test_freqs[0])

        # Measure stopband attenuation in bins >= detected_cutoff + 1200 Hz
        stop_mask = test_freqs >= (detected_cutoff + 1200.0)
        if np.any(stop_mask):
            noise_floor = float(np.mean(test_power[stop_mask]))
        else:
            noise_floor = float(test_power[-1])

        attenuation = float(ref_db - noise_floor)
        # Heuristic thresholds, not a general distinction between mastering and codecs.
        is_suspicious = (detected_cutoff < 19500.0) and (attenuation >= 28.0)

    # Offer familiar codec comparisons as clues, not provenance certification.
    if detected_cutoff >= 20000.0 or not is_suspicious:
        verdict = "No obvious signs of a low-bitrate cutoff"
        confidence = 0.95
    elif detected_cutoff < 17200.0:
        verdict = "Possible 128 kbps MP3-like source"
        confidence = 0.92
    elif detected_cutoff < 19500.0:
        verdict = "Possible lower-bitrate source — roughly 192 kbps MP3-like"
        confidence = 0.88
    else:
        verdict = "Bandlimiting flagged"
        confidence = 0.75

    return CutoffDetectionAnalysis(
        detected_cutoff_hz=float(detected_cutoff),
        peak_shelf_db=float(ref_db),
        noise_floor_db=float(noise_floor),
        stopband_attenuation_db=float(attenuation),
        verdict=verdict,
        is_suspicious_transcode=is_suspicious,
        confidence=float(confidence),
    )

