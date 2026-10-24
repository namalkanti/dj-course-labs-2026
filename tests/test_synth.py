"""Unit tests verifying physical DSP properties of synthetic audio fixtures.

Self-documenting assertion helpers encapsulate domain invariants and are grouped
at the bottom of the module:
- Sample count alignment with duration and sample rate.
- Peak amplitude bounds and normalization.
- Spectral flatness across linear frequency bands.
- Monotonic instantaneous frequency sweeps via zero-crossing rate.
- Bass and treble energy dominance across rhythmic segments.
- Brick-wall stopband attenuation in simulated codec cutoff filters.
"""

import numpy as np
import pytest
import scipy.signal as signal

from yourshot.synth import (
    apply_lossy_cutoff,
    generate_chirp,
    generate_drum_loop,
    generate_white_noise,
)


# ---------------------------------------------------------------------------
# Tests (Specifications)
# ---------------------------------------------------------------------------


def test_white_noise_spectral_flatness_and_level():
    fs = 44100
    duration_sec = 2.0
    audio = generate_white_noise(duration_sec=duration_sec, fs=fs)

    assert_sample_count_matches_duration(audio, duration_sec, fs)
    assert_peak_amplitude_equals(audio, 0.5)
    assert_spectral_flatness_between_bands(
        audio, fs, low_band_hz=(1000.0, 5000.0), high_band_hz=(15000.0, 19000.0)
    )


def test_chirp_frequency_progression():
    fs = 44100
    duration_sec = 1.0
    audio = generate_chirp(duration_sec=duration_sec, fs=fs, f_start=50.0, f_end=20000.0)

    assert_sample_count_matches_duration(audio, duration_sec, fs)
    assert_peak_amplitude_bounded_below(audio, 0.5)

    early_100ms = audio[: int(0.1 * fs)]
    late_100ms = audio[int(0.9 * fs) :]
    assert_zero_crossing_rate_increases(early_100ms, late_100ms, min_ratio=10.0)


def test_drum_loop_rhythm_and_frequency_split():
    fs = 44100
    bpm = 120.0
    duration_sec = 2.0
    audio = generate_drum_loop(duration_sec=duration_sec, bpm=bpm, fs=fs)

    assert_sample_count_matches_duration(audio, duration_sec, fs)
    assert_peak_amplitude_equals(audio, 0.7)

    kick_segment = audio[: int(0.1 * fs)]
    hat_segment = audio[int(0.24 * fs) : int(0.30 * fs)]

    assert_fractional_energy_in_frequency_band(
        kick_segment, fs, f_min=0.0, f_max=200.0, min_fraction=0.7
    )
    assert_fractional_energy_in_frequency_band(
        hat_segment, fs, f_min=5000.0, f_max=22050.0, min_fraction=0.5
    )


def test_apply_lossy_cutoff_brickwall_attenuation():
    fs = 44100
    noise = generate_white_noise(duration_sec=2.0, fs=fs)

    bypass = apply_lossy_cutoff(noise, cutoff_hz=22050.0, fs=fs)
    np.testing.assert_array_equal(bypass, noise)

    filtered = apply_lossy_cutoff(noise, cutoff_hz=16000.0, fs=fs)
    assert_stopband_attenuation_exceeds(
        filtered,
        fs,
        passband_hz=(5000.0, 14000.0),
        stopband_hz=(18000.0, 21000.0),
        min_attenuation_db=40.0,
    )


# ---------------------------------------------------------------------------
# Domain-Specific Assertion Helpers
# ---------------------------------------------------------------------------


def assert_sample_count_matches_duration(
    audio: np.ndarray, duration_sec: float, fs: int
) -> None:
    """Assert audio array length equals exact integer product of duration and sample rate."""
    expected_samples = int(duration_sec * fs)
    assert len(audio) == expected_samples, (
        f"Expected {expected_samples} samples for {duration_sec}s at {fs} Hz, got {len(audio)}"
    )


def assert_peak_amplitude_equals(
    audio: np.ndarray, expected_peak: float, tolerance: float = 1e-5
) -> None:
    """Assert maximum absolute sample value equals expected peak amplitude."""
    actual_peak = float(np.max(np.abs(audio)))
    assert actual_peak == pytest.approx(expected_peak, abs=tolerance), (
        f"Peak amplitude {actual_peak} did not match expected {expected_peak}"
    )


def assert_peak_amplitude_bounded_below(audio: np.ndarray, ceiling: float) -> None:
    """Assert maximum absolute sample value does not exceed ceiling."""
    actual_peak = float(np.max(np.abs(audio)))
    assert actual_peak <= ceiling, f"Peak amplitude {actual_peak} exceeded ceiling {ceiling}"


def assert_spectral_flatness_between_bands(
    audio: np.ndarray,
    fs: int,
    low_band_hz: tuple[float, float],
    high_band_hz: tuple[float, float],
    tolerance_db: float = 2.0,
) -> None:
    """Assert average power in two linear frequency bands matches within a decibel tolerance."""
    freqs, psd = signal.welch(audio, fs=fs, nperseg=2048)
    low_power = np.mean(psd[(freqs >= low_band_hz[0]) & (freqs <= low_band_hz[1])])
    high_power = np.mean(psd[(freqs >= high_band_hz[0]) & (freqs <= high_band_hz[1])])
    power_ratio_db = 10.0 * np.log10(high_power / low_power)
    assert power_ratio_db == pytest.approx(0.0, abs=tolerance_db), (
        f"Band power delta of {power_ratio_db:.2f} dB exceeds tolerance of +/- {tolerance_db} dB"
    )


def assert_zero_crossing_rate_increases(
    early_audio: np.ndarray,
    late_audio: np.ndarray,
    min_ratio: float = 10.0,
) -> None:
    """Assert later audio segment has a dramatically higher zero-crossing rate than earlier segment.

    Physical Meaning of Zero-Crossing Rate (ZCR):
    A "zero crossing" occurs whenever successive discrete samples change algebraic sign
    (i.e. x[n] * x[n+1] < 0). In time-domain signal processing:
    - For a pure or narrowband sinusoidal wave, each complete cycle crosses zero exactly twice.
    - Therefore, the average frequency f_0 relates directly to zero-crossing count N_zc over
      interval Delta_t: f_0 ~ N_zc / (2 * Delta_t).
    - A low-pitched bass tone (e.g. 50 Hz) oscillates slowly, yielding very few zero crossings
      per 100 ms (~10 crossings).
    - A high-pitched treble tone (e.g. 15 kHz) oscillates rapidly, producing thousands of zero
      crossings in the same 100 ms (~3000 crossings).

    In a frequency-swept chirp from 50 Hz to 20 kHz, comparing the zero-crossing rate between
    the first 100 ms and the final 100 ms provides a lightweight, robust time-domain proof
    of monotonic upward pitch progression without needing Fourier phase unwrapping.
    """
    early_crossings = int(np.sum(np.diff(np.signbit(early_audio))))
    late_crossings = int(np.sum(np.diff(np.signbit(late_audio))))
    assert late_crossings >= early_crossings * min_ratio, (
        f"Late crossings ({late_crossings}) not >= {min_ratio}x early crossings ({early_crossings})"
    )


def assert_fractional_energy_in_frequency_band(
    segment: np.ndarray,
    fs: int,
    f_min: float,
    f_max: float,
    min_fraction: float,
) -> None:
    """Assert power inside [f_min, f_max] constitutes at least min_fraction of total segment power."""
    freqs, psd = signal.welch(segment, fs=fs, nperseg=1024)
    band_power = float(np.sum(psd[(freqs >= f_min) & (freqs <= f_max)]))
    total_power = float(np.sum(psd))
    fraction = band_power / (total_power + 1e-12)
    assert fraction >= min_fraction, (
        f"Band [{f_min}, {f_max}] Hz power fraction {fraction:.2%} is below required {min_fraction:.2%}"
    )


def assert_stopband_attenuation_exceeds(
    filtered_audio: np.ndarray,
    fs: int,
    passband_hz: tuple[float, float],
    stopband_hz: tuple[float, float],
    min_attenuation_db: float = 40.0,
) -> None:
    """Assert passband power exceeds stopband power by at least min_attenuation_db."""
    freqs, psd = signal.welch(filtered_audio, fs=fs, nperseg=2048)
    pass_power = np.mean(psd[(freqs >= passband_hz[0]) & (freqs <= passband_hz[1])])
    stop_power = np.mean(psd[(freqs >= stopband_hz[0]) & (freqs <= stopband_hz[1])])
    attenuation_db = 10.0 * np.log10(pass_power / (stop_power + 1e-12))
    assert attenuation_db >= min_attenuation_db, (
        f"Stopband attenuation {attenuation_db:.1f} dB is below required {min_attenuation_db} dB"
    )
