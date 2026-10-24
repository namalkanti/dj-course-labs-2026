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

    Splits audio using 2nd-order Linkwitz-Riley/Butterworth crossover filters into:
    - Low Band: f < low_cutoff (Bass, < 250 Hz)
    - Mid Band: low_cutoff <= f < high_cutoff (Midrange, 250 Hz - 2.5 kHz)
    - High Band: f >= high_cutoff (Treble, >= 2.5 kHz)

    Then performs block-wise RMS pooling over windows of hop_length samples.

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
