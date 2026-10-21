"""Unit tests verifying pure functional spectral transformations.

Mathematical and DSP Theory Under Test:
---------------------------------------
1. Welch's Method (Power Spectral Density Estimation):
   - Solves the inconsistent variance problem of the raw periodogram by partitioning
     the input signal into overlapping Hann-windowed segments and averaging their periodograms.
   - Frequency resolution is determined by window length L: Delta_f = fs / L.
   - For a real-valued input, the one-sided spectrum contains (L // 2 + 1) discrete bins
     spanning DC (0 Hz) to Nyquist (fs / 2 Hz).

2. Short-Time Fourier Transform (STFT Spectrogram):
   - Computes time-localized frequency spectra via sliding analysis windows.
   - Dynamic range clipping to [vmax - dynamic_range_db, vmax] avoids visual obscuration
     by numerical noise floors, matching professional DAW and RTA analyzer displays.

3. Rekordbox 3-Band RGB Waveform:
   - Emulates Pioneer DJ CDJ/Rekordbox RGB waveform rendering:
     * Low Band (< 250 Hz): Basslines and kick drums mapped to the Red channel.
     * Mid Band (250 Hz - 2500 Hz): Vocals, snare bodies, and synths mapped to Green.
     * High Band (>= 2500 Hz): Hi-hats, cymbals, and vocal air mapped to Blue.
   - Envelopes are computed via block-wise extrema, while colors are derived from
     the relative RMS energy per band normalized into the RGB unit cube [0.0, 1.0]^3.

4. High-Frequency Shelf Extraction:
   - Isolates the upper spectral register (typically >= 10 kHz) and normalizes peak
     power to 0 dBFS to expose codec lowpass filter brick-wall shelves (16 kHz for 128k,
     18.5 kHz for 192k, 20 kHz for 320k MP3).
"""

from typing import Tuple
import numpy as np
import pytest

from yourshot.spectral import (
    RekordboxWaveformResult,
    SpectrogramResult,
    SpectrumResult,
    compute_high_frequency_shelf,
    compute_rekordbox_rgb_waveform,
    compute_stft_spectrogram,
    compute_welch_spectrum,
)


def generate_pure_sine(freq_hz: float, duration_sec: float = 1.0, fs: int = 44100) -> np.ndarray:
    """Generate a clean, deterministic float32 sinusoidal test signal.

    Args:
        freq_hz: Tone frequency in Hz.
        duration_sec: Total duration in seconds.
        fs: Sampling rate in samples per second.

    Returns:
        1D NumPy float32 array normalized to 0.8 amplitude (-1.94 dBFS)
        to prevent clipping in down-stream filtering.
    """
    t = np.arange(int(duration_sec * fs), dtype=np.float32) / fs
    return (0.8 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_welch_spectrum
# ---------------------------------------------------------------------------


def test_welch_spectrum_shapes_and_bounds():
    """Verify Welch PSD output dimensions, frequency bounds, and monotonicity.

    Mathematical Invariants:
    - Bin Count: Real rFFT produces exactly (window_length // 2 + 1) bins.
    - Endpoints: The first bin is strictly 0.0 Hz (DC) and the final bin is
      strictly fs / 2 Hz (Nyquist).
    - Monotonicity: Frequency bins must advance with strictly positive constant step.
    - PeakdB: The returned scalar peak_db must equal the array maximum.
    """
    fs = 44100
    duration_sec = 1.0
    audio = generate_pure_sine(1000.0, duration_sec=duration_sec, fs=fs)
    window_length = 2048

    result = compute_welch_spectrum(audio, fs=fs, window_length=window_length)

    assert isinstance(result, SpectrumResult)
    num_bins = window_length // 2 + 1
    assert result.frequencies.shape == (num_bins,)
    assert result.power_db.shape == (num_bins,)
    assert result.frequencies[0] == pytest.approx(0.0)
    assert result.frequencies[-1] == pytest.approx(fs / 2.0)
    assert np.all(np.diff(result.frequencies) > 0.0), "Frequencies must be monotonically increasing"
    assert result.peak_db == pytest.approx(float(np.max(result.power_db)))


def test_welch_spectrum_peak_frequency_localization():
    """Verify that a single-frequency tone concentrates its spectral power at the expected bin.

    DSP Principle:
    A pure continuous sinusoid x(t) = sin(2*pi*f_0*t) windowed by a Hann taper
    has its Fourier transform centered at f_0, with main-lobe width proportional
    to 2 * (fs / window_length). The discrete bin with maximum power must be
    within +/- 1 bin of the true continuous frequency f_0.
    """
    fs = 44100
    target_freq = 1000.0
    audio = generate_pure_sine(target_freq, duration_sec=2.0, fs=fs)
    window_length = 4096

    result = compute_welch_spectrum(audio, fs=fs, window_length=window_length)

    peak_idx = int(np.argmax(result.power_db))
    peak_freq = result.frequencies[peak_idx]
    bin_width = fs / window_length  # ~10.74 Hz for 44.1k and L=4096

    assert abs(peak_freq - target_freq) <= bin_width, (
        f"Detected peak at {peak_freq:.2f} Hz differs from target {target_freq} Hz "
        f"by more than bin width ({bin_width:.2f} Hz)"
    )


# ---------------------------------------------------------------------------
# compute_stft_spectrogram
# ---------------------------------------------------------------------------


def test_stft_spectrogram_shapes_and_clipping():
    """Verify STFT spectrogram 2D matrix geometry, axes spans, and decibel floor clamping.

    DSP Invariants:
    - The frequency axis has dimension (window_length // 2 + 1) spanning [0, fs/2].
    - The time axis spans [0, duration_sec] with resolution (window_length - overlap_length) / fs.
    - Decibel Display Dynamic Range: All entries in power_db must be clamped to
      [vmax - dynamic_range_db, vmax] so that low-level quantization noise
      does not pollute visual inspection.
    """
    fs = 44100
    duration_sec = 1.5
    audio = generate_pure_sine(440.0, duration_sec=duration_sec, fs=fs)
    window_length = 2048
    overlap_length = 1536

    result = compute_stft_spectrogram(
        audio,
        fs=fs,
        window_length=window_length,
        overlap_length=overlap_length,
        dynamic_range_db=80.0,
    )

    assert isinstance(result, SpectrogramResult)
    assert result.times.ndim == 1
    assert result.frequencies.ndim == 1
    assert result.power_db.shape == (len(result.frequencies), len(result.times))

    # Time array should span approximately [0, duration]
    assert result.times[0] >= 0.0
    assert result.times[-1] == pytest.approx(duration_sec, abs=0.1)

    # Frequency array should span [0, Nyquist]
    assert result.frequencies[0] == pytest.approx(0.0)
    assert result.frequencies[-1] == pytest.approx(fs / 2.0)

    # Dynamic range clipping: all values must be >= vmax - dynamic_range_db
    assert result.vmax == pytest.approx(float(np.max(result.power_db)))
    assert result.vmin == pytest.approx(result.vmax - 80.0)
    assert np.all(result.power_db >= result.vmin - 1e-6)
    assert np.all(result.power_db <= result.vmax + 1e-6)


# ---------------------------------------------------------------------------
# compute_rekordbox_rgb_waveform
# ---------------------------------------------------------------------------


def test_rekordbox_rgb_waveform_shapes_and_color_ranges():
    """Verify amplitude envelope boundaries and RGB color normalizations.

    Waveform Requirements:
    - Envelope symmetry: The positive envelope must be >= 0; negative envelope <= 0.
    - Color Bounds: Each RGB channel must lie strictly within [0.0, 1.0] to serve
      as valid input for Matplotlib, HTML5 canvas, or OpenGL shaders.
    - Frame Alignment: Total frames must match ceil(N / hop_length).
    """
    fs = 44100
    duration_sec = 1.0
    audio = generate_pure_sine(500.0, duration_sec=duration_sec, fs=fs)
    hop_length = 256

    result = compute_rekordbox_rgb_waveform(audio, fs=fs, hop_length=hop_length)

    assert isinstance(result, RekordboxWaveformResult)
    num_frames = len(result.times)
    expected_frames = len(audio) // hop_length
    assert abs(num_frames - expected_frames) <= 2

    assert result.envelope_upper.shape == (num_frames,)
    assert result.envelope_lower.shape == (num_frames,)
    assert result.rgb_colors.shape == (num_frames, 3)

    # Upper envelope >= 0, Lower envelope <= 0
    assert np.all(result.envelope_upper >= -1e-6)
    assert np.all(result.envelope_lower <= 1e-6)

    # RGB values must be in [0.0, 1.0]
    assert np.all(result.rgb_colors >= 0.0)
    assert np.all(result.rgb_colors <= 1.0)


@pytest.mark.parametrize(
    "freq_hz, dominant_channel, channel_name",
    [
        (80.0, 0, "Red (Bass < 250 Hz)"),
        (1000.0, 1, "Green (Mid 250 Hz - 2.5 kHz)"),
        (8000.0, 2, "Blue (High >= 2.5 kHz)"),
    ],
)
def test_rekordbox_frequency_to_color_mapping(freq_hz: float, dominant_channel: int, channel_name: str):
    """Verify that pure tones map to Pioneer Rekordbox color channels.

    Rekordbox Crossover Model:
    - Red channel: Bassband energy (< 250 Hz). A 80 Hz kick/sub must produce Red dominance.
    - Green channel: Midband energy (250 Hz - 2.5 kHz). A 1 kHz synth/vocal must produce Green dominance.
    - Blue channel: Highband energy (>= 2.5 kHz). An 8 kHz hi-hat/sizzle must produce Blue dominance.

    The first 10% of samples are discarded before checking channel dominance
    to allow the IIR crossover filter state to reach steady-state response.
    """
    fs = 44100
    audio = generate_pure_sine(freq_hz, duration_sec=1.0, fs=fs)

    result = compute_rekordbox_rgb_waveform(
        audio,
        fs=fs,
        hop_length=256,
        low_cutoff=250.0,
        high_cutoff=2500.0,
    )

    # Exclude initial filter transient (first 10% of frames)
    steady_frames = result.rgb_colors[int(len(result.rgb_colors) * 0.1) :]
    mean_colors = np.mean(steady_frames, axis=0)

    dominant_detected = int(np.argmax(mean_colors))
    assert dominant_detected == dominant_channel, (
        f"For {freq_hz} Hz expected dominant {channel_name}, but got mean (R, G, B) = {mean_colors}"
    )


# ---------------------------------------------------------------------------
# compute_high_frequency_shelf
# ---------------------------------------------------------------------------


def test_high_frequency_shelf_extraction_and_normalization():
    """Verify high-frequency shelf slicing and peak decibel normalization.

    Analytical Goal:
    To detect lossy MP3 encoder cutoff filters (e.g., 16 kHz brick-wall),
    we inspect the upper spectral range [f_min, Nyquist].
    Normalizing the peak in this region to 0 dBFS ensures that codec cutoff
    shelves align to a standard relative vertical datum regardless of
    the track's overall recording volume or gain staging.
    """
    fs = 44100
    frequencies = np.linspace(0.0, fs / 2.0, 1025)
    # Synthetic spectrum with linear power drop: -20 dB at DC to -42 dB at Nyquist
    power_db = -20.0 - (frequencies / 1000.0)

    f_min = 10000.0
    shelf_freqs, shelf_power = compute_high_frequency_shelf(
        frequencies,
        power_db,
        f_min=f_min,
        normalize_peak=True,
    )

    # All returned frequencies must reside in [f_min, Nyquist]
    assert np.all(shelf_freqs >= f_min)
    assert shelf_freqs[-1] == pytest.approx(fs / 2.0)
    assert shelf_freqs.shape == shelf_power.shape

    # Peak normalization: maximum decibel level in the sliced shelf must be exactly 0 dB
    assert np.max(shelf_power) == pytest.approx(0.0)
