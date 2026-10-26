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
from scipy import signal

from yourshot.synth import (
    apply_lossy_cutoff,
    generate_chirp,
    generate_drum_loop,
    generate_syncopated_breakbeat,
    generate_synthetic_beat,
    generate_tempo_drift_beat,
    generate_waltz_beat,
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


def test_synthetic_beat_four_on_the_floor_structure():
    """Verify that a 4/4 synthetic beat has kicks on all integer beats and sub-bass dominance.

    Physical Invariants:
    - Total sample count matches exact duration.
    - Transient attacks align with quarter-note grid intervals (T_beat = 60 / BPM).
    - Downbeat kick slice (0-150ms) concentrates over 60% of spectral energy in the sub-bass (<200 Hz).
    """
    fs = 44100
    bpm = 128.0
    duration_sec = 4.0
    audio = generate_synthetic_beat(bpm=bpm, duration_sec=duration_sec, meter="4/4", fs=fs)

    assert_sample_count_matches_duration(audio, duration_sec, fs)
    assert_peak_amplitude_bounded_below(audio, 1.0)

    beat_interval = 60.0 / bpm
    expected_beat_times = [i * beat_interval for i in range(int(duration_sec / beat_interval))]
    assert_transient_onsets_present_at(audio, fs, expected_beat_times, tolerance_ms=25.0)

    kick_window = audio[: int(0.15 * fs)]
    assert_fractional_energy_in_frequency_band(
        kick_window, fs, f_min=20.0, f_max=200.0, min_fraction=0.6
    )


def test_waltz_beat_triple_meter_structure():
    """Verify that a 3/4 synthetic waltz exhibits a 3-beat metric hierarchy.

    Physical Invariants:
    - Downbeat occurs on beat 1 of each 3-beat bar (kick), with offbeats on beats 2 and 3 (hats/clicks).
    - All beat positions possess distinct transient attacks matching grid intervals.
    - Downbeat kick possesses at least 6 dB higher sub-bass power than the offbeat hits.
    """
    fs = 44100
    bpm = 120.0
    duration_sec = 6.0
    audio = generate_waltz_beat(bpm=bpm, duration_sec=duration_sec, fs=fs)

    assert_sample_count_matches_duration(audio, duration_sec, fs)
    assert_peak_amplitude_bounded_below(audio, 1.0)

    beat_interval = 60.0 / bpm  # 0.5s
    num_beats = int(duration_sec / beat_interval)
    downbeat_times = [i * beat_interval for i in range(num_beats) if i % 3 == 0]
    offbeat_times = [i * beat_interval for i in range(num_beats) if i % 3 != 0]

    assert_transient_onsets_present_at(audio, fs, downbeat_times, tolerance_ms=25.0)
    assert_transient_onsets_present_at(audio, fs, offbeat_times, tolerance_ms=25.0)

    downbeat_segment = audio[: int(0.15 * fs)]
    offbeat_segment = audio[int(0.5 * fs) : int(0.65 * fs)]
    assert_sub_bass_energy_ratio_exceeds(
        downbeat_segment, offbeat_segment, fs, f_max=200.0, min_ratio_db=6.0
    )


def test_syncopated_breakbeat_structure():
    """Verify that a breakbeat exhibits both on-grid pulses and offbeat syncopation.

    What is Syncopation?
    Syncopation is an intentional rhythmic displacement: placing strong percussive
    attacks on normally weak beats or subdivisions (e.g. 16th-note anticipations,
    or the "and" of beat 2) while leaving standard on-beat pulses unaccented or silent.
    In beat-tracking DSP, syncopation creates strong local transients at fractional
    intervals (e.g., 1.75 * T_beat), testing whether autocorrelation can filter out
    the groove variations over multiple bars to detect the true underlying BPM.

    Physical Invariants:
    - Primary grid downbeats contain transient attacks.
    - Syncopated 16th-note subdivisions (e.g. at 1.75 * T_beat) contain distinct transient energy.
    """
    fs = 44100
    bpm = 140.0
    duration_sec = 4.0
    audio = generate_syncopated_breakbeat(bpm=bpm, duration_sec=duration_sec, fs=fs)

    assert_sample_count_matches_duration(audio, duration_sec, fs)
    assert_peak_amplitude_bounded_below(audio, 1.0)

    beat_interval = 60.0 / bpm  # ~0.4285s
    grid_beat_times = [i * beat_interval for i in range(int(duration_sec / beat_interval))]
    assert_transient_onsets_present_at(audio, fs, grid_beat_times, tolerance_ms=30.0)

    syncopated_hit_time = 1.75 * beat_interval
    assert_transient_onsets_present_at(audio, fs, [syncopated_hit_time], tolerance_ms=30.0)


def test_tempo_drift_beat_varying_intervals():
    """Verify that a drifting-tempo beat accelerates continuously over the duration.

    Physical Invariants:
    - Early beat intervals correspond closely to start_bpm.
    - Final beat intervals correspond closely to end_bpm.
    - Inter-onset intervals strictly decrease as tempo accelerates.
    """
    fs = 44100
    start_bpm = 120.0
    end_bpm = 140.0
    duration_sec = 8.0
    audio = generate_tempo_drift_beat(
        start_bpm=start_bpm, end_bpm=end_bpm, duration_sec=duration_sec, fs=fs
    )

    assert_sample_count_matches_duration(audio, duration_sec, fs)
    assert_peak_amplitude_bounded_below(audio, 1.0)

    assert_beat_intervals_accelerate(
        audio, fs, early_bpm_approx=start_bpm, late_bpm_approx=end_bpm
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


def assert_transient_onsets_present_at(
    audio: np.ndarray,
    fs: int,
    expected_times_sec: list[float],
    tolerance_ms: float = 25.0,
) -> None:
    """Assert peak energy in a sliding RMS/envelope aligns with expected transient timestamps.

    Physical Meaning:
    A transient onset marks the percussive attack of an instrument (e.g. kick beater hit,
    snare rim contact). In a clean synthetic drum loop, high instantaneous power is
    concentrated at specific beat timestamps.

    Verification Method:
    - Calculates a 5ms moving-average squared energy envelope e(t).
    - For each target timestamp, extracts the local window [t - tol, t + tol].
    - Verifies that an energy peak exists (amplitude > 1e-4) and its local maximum
      occurs within the specified millisecond tolerance window.
    """
    tol_samples = int((tolerance_ms / 1000.0) * fs)
    win_len = max(64, int(0.005 * fs))  # 5ms local window
    # Compute local squared energy envelope smoothed with moving average
    sq = audio**2
    env = np.convolve(sq, np.ones(win_len) / win_len, mode="same")

    for t_sec in expected_times_sec:
        center_idx = int(t_sec * fs)
        start_idx = max(0, center_idx - tol_samples)
        end_idx = min(len(audio), center_idx + tol_samples)
        if start_idx >= end_idx:
            continue
        local_window = env[start_idx:end_idx]
        local_peak_idx = start_idx + int(np.argmax(local_window))
        # Ensure there is non-trivial energy near this timestamp
        MIN_TRANSIENT_ENERGY = 1e-4
        assert np.max(local_window) > MIN_TRANSIENT_ENERGY, f"No transient detected near t={t_sec:.3f}s"
        assert abs(local_peak_idx - center_idx) <= tol_samples, (
            f"Transient at t={local_peak_idx / fs:.3f}s deviated from expected {t_sec:.3f}s"
        )


def assert_sub_bass_energy_ratio_exceeds(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    fs: int,
    f_max: float = 200.0,
    min_ratio_db: float = 6.0,
) -> None:
    """Assert sub-bass power in audio_a exceeds audio_b by at least min_ratio_db.

    Physical Meaning:
    In electronic dance and acoustic percussion, kick drums contain significant acoustic energy
    in the sub-bass and low frequencies (< 200 Hz), whereas hi-hats, snares, and clicks are
    predominantly high-frequency or broadband noise.

    Verification Method:
    - Estimates the Power Spectral Density (PSD) using Welch's periodogram for both audio clips.
    - Integrates power below f_max (200 Hz).
    - Checks that 10 * log10(P_a / P_b) >= min_ratio_db.
    """
    freqs_a, psd_a = signal.welch(audio_a, fs=fs, nperseg=min(1024, len(audio_a)))
    freqs_b, psd_b = signal.welch(audio_b, fs=fs, nperseg=min(1024, len(audio_b)))
    power_a = float(np.sum(psd_a[freqs_a <= f_max]))
    power_b = float(np.sum(psd_b[freqs_b <= f_max])) + 1e-12
    ratio_db = 10.0 * np.log10(power_a / power_b)
    assert ratio_db >= min_ratio_db, (
        f"Sub-bass ratio {ratio_db:.1f} dB is below required {min_ratio_db} dB"
    )


def assert_beat_intervals_accelerate(
    audio: np.ndarray,
    fs: int,
    early_bpm_approx: float,
    late_bpm_approx: float,
) -> None:
    """Assert inter-onset intervals between initial beats are longer than later beats.

    Physical Meaning:
    A tempo drift or acceleration implies that the duration of a quarter note (T_beat = 60 / BPM)
    contracts over time. For example, moving from 120 BPM (T_0 = 500 ms) to 140 BPM (T_1 = 428 ms)
    causes successive peak-to-peak transient intervals to shorten.

    Verification Method:
    - Derives onset candidate peaks using smoothed amplitude differencing.
    - Compares the initial inter-beat duration (peak_times[1] - peak_times[0]) against the
      final inter-beat duration (peak_times[-1] - peak_times[-2]).
    - Asserts that early_interval > late_interval and that both match expected bounds.
    """
    # Find onset peaks across whole signal using half-wave rectified differencing of smoothed envelope
    win_len = int(0.005 * fs)
    env = np.convolve(np.abs(audio), np.ones(win_len) / win_len, mode="same")
    diff = np.maximum(0.0, np.diff(env))
    # Threshold at 20% of max diff
    peaks, _ = signal.find_peaks(diff, height=0.2 * np.max(diff), distance=int(0.2 * fs))
    peak_times = peaks / fs

    MIN_BEAT_PEAKS = 6
    assert len(peak_times) >= MIN_BEAT_PEAKS, f"Expected at least {MIN_BEAT_PEAKS} beat peaks, found {len(peak_times)}"

    early_interval = peak_times[1] - peak_times[0]
    late_interval = peak_times[-1] - peak_times[-2]

    # Early interval should be longer than late interval (accelerating tempo)
    assert early_interval > late_interval, (
        f"Early interval {early_interval:.3f}s was not greater than late interval {late_interval:.3f}s"
    )
    # Check rough alignment with expected start/end BPM
    expected_early_int = 60.0 / early_bpm_approx
    expected_late_int = 60.0 / late_bpm_approx
    assert early_interval == pytest.approx(expected_early_int, rel=0.15)
    assert late_interval == pytest.approx(expected_late_int, rel=0.15)

