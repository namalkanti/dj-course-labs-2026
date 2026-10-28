"""Unit tests verifying physical and mathematical properties of spectral transformations.

Self-documenting assertion helpers encapsulate domain invariants and are grouped
at the bottom of the module:
- Output array dimensions and boundary frequencies.
- Peak frequency localization within discrete bin limits.
- Dynamic range clamping floors.
- 3-band individual envelope shapes and non-negativity.
- Blended RGB waveform conversion, envelope symmetry, and color unit cube bounds.
- Frequency-to-color crossover channel dominance.
- Sub-band spectrum cropping and peak decibel normalization.
"""

from typing import Tuple
import numpy as np
import pytest

from dj_dsp.spectral_data import (
    CutoffDetectionAnalysis,
    RGBBlendedWaveformView,
    SpectrogramView,
    SpectrumView,
    ThreeBandWaveformView,
)
from dj_dsp.spectral import (
    analyze_spectral_cutoff,
    compute_stft_spectrogram,
    compute_three_band_waveform,
    compute_welch_spectrum,
)


def generate_pure_sine(freq_hz: float, duration_sec: float = 1.0, fs: int = 44100) -> np.ndarray:
    """Generate a clean float32 sinusoidal test signal normalized to 0.8 amplitude."""
    t = np.arange(int(duration_sec * fs), dtype=np.float32) / fs
    return (0.8 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests (Specifications)
# ---------------------------------------------------------------------------


def test_welch_spectrum_shapes_and_bounds():
    """Welch PSD must produce (L//2 + 1) bins spanning DC (0 Hz) to Nyquist (fs/2)."""
    fs = 44100
    duration_sec = 1.0
    audio = generate_pure_sine(1000.0, duration_sec=duration_sec, fs=fs)
    window_length = 2048

    view = compute_welch_spectrum(audio, fs=fs, window_length=window_length)

    assert_spectrum_view_structure(view, expected_bins=window_length // 2 + 1, fs=fs)


def test_welch_spectrum_peak_frequency_localization():
    """Welch PSD of a pure tone must concentrate maximum power within +/- 1 bin of target."""
    fs = 44100
    target_freq = 1000.0
    audio = generate_pure_sine(target_freq, duration_sec=2.0, fs=fs)
    window_length = 4096

    view = compute_welch_spectrum(audio, fs=fs, window_length=window_length)

    assert_peak_frequency_near(view, target_freq=target_freq, tolerance_hz=fs / window_length)


def test_spectrum_view_crop_and_normalize():
    """Cropping a SpectrumView to [f_min, f_max] with normalize_peak must offset peak to 0.0 dBFS."""
    fs = 44100
    frequencies = np.linspace(0.0, fs / 2.0, 1025)
    power_db = -20.0 - (frequencies / 1000.0)
    full_view = SpectrumView(frequencies=frequencies, power_db=power_db)
    f_min = 10000.0

    cropped_view = full_view.crop(f_min=f_min, normalize_peak=True)

    assert isinstance(cropped_view, SpectrumView)
    assert_frequencies_bounded_between(cropped_view.frequencies, f_min=f_min, f_max=fs / 2.0)
    assert_spectrum_peak_normalized_to_zero(cropped_view)


def test_stft_spectrogram_shapes_and_clipping():
    """STFT 2D spectrogram must span full time/frequency bounds and clamp to top 80 dB span (e.g. -80 to 0 dBFS)."""
    fs = 44100
    duration_sec = 1.5
    audio = generate_pure_sine(440.0, duration_sec=duration_sec, fs=fs)
    window_length = 2048
    overlap_length = 1536

    view = compute_stft_spectrogram(
        audio,
        fs=fs,
        window_length=window_length,
        overlap_length=overlap_length,
        dynamic_range_db=80.0,
    )

    assert_spectrogram_view_structure(view, fs=fs, duration_sec=duration_sec)
    assert_spectrogram_dynamic_range_clipped(view, dynamic_range_db=80.0)


def test_three_band_waveform_shapes_and_bounds():
    """Rekordbox 3-band view must compute non-negative envelopes for low, mid, and high bands."""
    fs = 44100
    duration_sec = 1.0
    audio = generate_pure_sine(500.0, duration_sec=duration_sec, fs=fs)
    hop_length = 256

    view = compute_three_band_waveform(audio, fs=fs, hop_length=hop_length)

    assert_three_band_waveform_valid(view, audio_len=len(audio), hop_length=hop_length)


def test_three_band_to_rgb_blended_conversion():
    """Converting 3-band envelopes to classic RGB must yield valid [0, 1] color coordinates."""
    fs = 44100
    duration_sec = 1.0
    audio = generate_pure_sine(500.0, duration_sec=duration_sec, fs=fs)
    hop_length = 256

    three_band = compute_three_band_waveform(audio, fs=fs, hop_length=hop_length)

    # Mock full-signal peak envelope
    num_frames = len(three_band.times)
    env_upper = np.full(num_frames, 0.8)
    env_lower = np.full(num_frames, -0.8)

    rgb_view = three_band.to_rgb_blended(envelope_upper=env_upper, envelope_lower=env_lower)

    assert isinstance(rgb_view, RGBBlendedWaveformView)
    assert_rgb_blended_waveform_envelope_valid(rgb_view, expected_frames=num_frames)
    assert_rgb_colors_within_unit_cube(rgb_view.rgb_colors)


@pytest.mark.parametrize(
    "freq_hz, dominant_channel, channel_name",
    [
        (80.0, 0, "Red (Bass < 250 Hz)"),
        (1000.0, 1, "Green (Mid 250 Hz - 2.5 kHz)"),
        (8000.0, 2, "Blue (High >= 2.5 kHz)"),
    ],
)
def test_three_band_frequency_isolation(freq_hz: float, dominant_channel: int, channel_name: str):
    """Pure tones must route to their respective Rekordbox 3-band crossover register."""
    fs = 44100
    audio = generate_pure_sine(freq_hz, duration_sec=1.0, fs=fs)

    three_band = compute_three_band_waveform(
        audio,
        fs=fs,
        hop_length=256,
        low_cutoff=250.0,
        high_cutoff=2500.0,
    )

    assert_dominant_three_band_envelope(
        three_band,
        expected_channel=dominant_channel,
        freq_hz=freq_hz,
        channel_name=channel_name,
    )


def test_cutoff_detection_pristine_wideband():
    """The wideband fixture yields >= 20 kHz cutoff without a suspicious flag."""
    from dj_dsp.synth import generate_white_noise
    fs = 44100
    noise = generate_white_noise(duration_sec=2.0, fs=fs)
    analysis = analyze_spectral_cutoff(noise, fs=fs)

    assert isinstance(analysis, CutoffDetectionAnalysis)
    assert analysis.detected_cutoff_hz >= 20000.0
    assert analysis.verdict == "High quality / full bandwidth audio"
    assert not analysis.is_suspicious_transcode


@pytest.mark.parametrize(
    "simulated_cutoff, expected_range",
    [
        (16000.0, (15500.0, 17500.0)),
        (18500.0, (18000.0, 19500.0)),
    ],
)
def test_cutoff_detection_simulated_transcodes(simulated_cutoff: float, expected_range: Tuple[float, float]):
    """Lowpass-filtered signals must accurately identify the cutoff frequency and flag as suspicious transcode."""
    from dj_dsp.synth import apply_lossy_cutoff, generate_white_noise
    fs = 44100
    noise = generate_white_noise(duration_sec=2.0, fs=fs)
    filtered = apply_lossy_cutoff(noise, cutoff_hz=simulated_cutoff, fs=fs, order=10)

    analysis = analyze_spectral_cutoff(filtered, fs=fs)

    assert analysis.is_suspicious_transcode
    assert analysis.verdict == "Low-bitrate transcode detected (cut off below 19.5 kHz)"
    assert expected_range[0] <= analysis.detected_cutoff_hz <= expected_range[1]
    assert analysis.stopband_attenuation_db >= 28.0


def test_cutoff_detection_pristine_flac_fixture():
    """A real pristine lossless FLAC track must not be flagged as a suspicious transcode."""
    import soundfile as sf
    from dj_dsp.audio_catalog import get_default_fixtures_dir

    fixture_path = get_default_fixtures_dir() / "persistence_by_dabmasterars_snippet.flac"
    audio, fs = sf.read(fixture_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    analysis = analyze_spectral_cutoff(audio, fs=fs)

    assert not analysis.is_suspicious_transcode, f"FLAC falsely flagged: {analysis.verdict}"
    assert analysis.detected_cutoff_hz >= 20000.0, f"Detected cutoff {analysis.detected_cutoff_hz} Hz too low for lossless"
    assert analysis.verdict == "High quality / full bandwidth audio"


def test_cutoff_detection_natural_rolloff_high_bitrate():
    """Signals with natural spectral rolloff and a ~20.2+ kHz cutoff must not be misidentified as low-bitrate."""
    from dj_dsp.synth import apply_lossy_cutoff, generate_white_noise
    from scipy import signal

    fs = 44100
    # Generate broadband noise shaped with a gentle 1/f rolloff (natural acoustic tilt)
    white = generate_white_noise(duration_sec=3.0, fs=fs)
    b, a = signal.butter(1, 4000.0 / (fs / 2.0), btype="low")
    shaped = signal.lfilter(b, a, white)
    shaped = shaped / np.max(np.abs(shaped))

    # Apply a high-quality 20.2 kHz brickwall cutoff (simulating 320 kbps MP3 encoder)
    audio_320k = apply_lossy_cutoff(shaped, cutoff_hz=20200.0, fs=fs, order=10)

    analysis = analyze_spectral_cutoff(audio_320k, fs=fs)

    # Must accurately find the cliff >= 19.5 kHz and classify as high quality
    assert not analysis.is_suspicious_transcode, f"Falsely flagged with verdict: {analysis.verdict}"
    assert analysis.detected_cutoff_hz >= 19500.0, f"Detected cutoff {analysis.detected_cutoff_hz} Hz is too low"
    assert analysis.verdict == "High quality / full bandwidth audio"


def test_cutoff_detection_silence_edge_case():
    """Silent or near-silent signals must not crash and should classify gracefully."""
    fs = 44100
    silence = np.zeros(fs * 2, dtype=np.float32)
    analysis = analyze_spectral_cutoff(silence, fs=fs)

    assert isinstance(analysis, CutoffDetectionAnalysis)
    assert analysis.confidence <= 0.2


# ---------------------------------------------------------------------------
# Domain-Specific Assertion Helpers
# ---------------------------------------------------------------------------


def assert_spectrum_view_structure(view: SpectrumView, expected_bins: int, fs: int) -> None:
    """Assert Welch PSD output dimensions, frequency bounds, and monotonicity."""
    assert isinstance(view, SpectrumView)
    assert view.frequencies.shape == (expected_bins,)
    assert view.power_db.shape == (expected_bins,)
    assert view.frequencies[0] == pytest.approx(0.0)
    assert view.frequencies[-1] == pytest.approx(fs / 2.0)
    assert np.all(np.diff(view.frequencies) > 0.0), "Frequencies must be monotonically increasing"
    assert view.peak_db == pytest.approx(float(np.max(view.power_db)))


def assert_peak_frequency_near(
    view: SpectrumView, target_freq: float, tolerance_hz: float
) -> None:
    """Assert frequency bin with maximum spectral power lies within tolerance_hz of target."""
    peak_idx = int(np.argmax(view.power_db))
    peak_freq = view.frequencies[peak_idx]
    assert abs(peak_freq - target_freq) <= tolerance_hz, (
        f"Detected peak at {peak_freq:.2f} Hz differs from target {target_freq} Hz "
        f"by more than bin tolerance ({tolerance_hz:.2f} Hz)"
    )


def assert_frequencies_bounded_between(
    frequencies: np.ndarray, f_min: float, f_max: float
) -> None:
    """Assert frequency array has been cropped so all values lie within [f_min, f_max]."""
    assert np.all(frequencies >= f_min)
    assert frequencies[-1] == pytest.approx(f_max)


def assert_spectrum_peak_normalized_to_zero(view: SpectrumView) -> None:
    """Assert maximum decibel level in the spectrum is exactly 0.0 dBFS."""
    assert view.peak_db == pytest.approx(0.0)


def assert_spectrogram_view_structure(
    view: SpectrogramView, fs: int, duration_sec: float
) -> None:
    """Assert STFT spectrogram matrix geometry and time/frequency axis extents."""
    assert isinstance(view, SpectrogramView)
    assert view.times.ndim == 1
    assert view.frequencies.ndim == 1
    assert view.power_db.shape == (len(view.frequencies), len(view.times))

    assert view.times[0] >= 0.0
    assert view.times[-1] == pytest.approx(duration_sec, abs=0.1)
    assert view.frequencies[0] == pytest.approx(0.0)
    assert view.frequencies[-1] == pytest.approx(fs / 2.0)


def assert_spectrogram_dynamic_range_clipped(
    view: SpectrogramView, dynamic_range_db: float
) -> None:
    """Assert spectrogram power_db values are in dBFS and clamped to [vmax - dynamic_range_db, vmax] (e.g. [-80 dBFS, 0 dBFS])."""
    assert view.vmax == pytest.approx(float(np.max(view.power_db)))
    assert view.vmin == pytest.approx(view.vmax - dynamic_range_db)
    assert np.all(view.power_db >= view.vmin - 1e-6)
    assert np.all(view.power_db <= view.vmax + 1e-6)


def assert_three_band_waveform_valid(
    view: ThreeBandWaveformView, audio_len: int, hop_length: int
) -> None:
    """Assert 3-band envelopes match frame counts and have non-negative amplitude bounds."""
    assert isinstance(view, ThreeBandWaveformView)
    num_frames = len(view.times)
    expected_frames = audio_len // hop_length
    assert abs(num_frames - expected_frames) <= 2

    for band_name, band_env in (("low", view.low), ("mid", view.mid), ("high", view.high)):
        assert band_env.shape == (num_frames,), f"Band {band_name} frame mismatch"
        assert np.all(band_env >= -1e-6), f"Band {band_name} contains negative amplitudes"


def assert_rgb_blended_waveform_envelope_valid(
    view: RGBBlendedWaveformView, expected_frames: int
) -> None:
    """Assert blended RGB waveform envelope framing and sign polarity."""
    assert len(view.times) == expected_frames
    assert view.envelope_upper.shape == (expected_frames,)
    assert view.envelope_lower.shape == (expected_frames,)
    assert np.all(view.envelope_upper >= -1e-6)
    assert np.all(view.envelope_lower <= 1e-6)


def assert_rgb_colors_within_unit_cube(rgb_colors: np.ndarray) -> None:
    """Assert RGB color array coordinates reside within [0.0, 1.0]^3."""
    assert rgb_colors.ndim == 2
    assert rgb_colors.shape[1] == 3
    assert np.all(rgb_colors >= 0.0)
    assert np.all(rgb_colors <= 1.0)


def assert_dominant_three_band_envelope(
    three_band: ThreeBandWaveformView,
    expected_channel: int,
    freq_hz: float,
    channel_name: str,
) -> None:
    """Assert steady-state energy is concentrated in expected band (0=low, 1=mid, 2=high)."""
    transient_cutoff = int(len(three_band.times) * 0.1)
    mean_low = float(np.mean(three_band.low[transient_cutoff:]))
    mean_mid = float(np.mean(three_band.mid[transient_cutoff:]))
    mean_high = float(np.mean(three_band.high[transient_cutoff:]))
    band_means = [mean_low, mean_mid, mean_high]

    dominant_band = int(np.argmax(band_means))
    assert dominant_band == expected_channel, (
        f"For {freq_hz} Hz expected dominant {channel_name}, but got means (low={mean_low:.3f}, mid={mean_mid:.3f}, high={mean_high:.3f})"
    )
