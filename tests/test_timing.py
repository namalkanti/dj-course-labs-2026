"""Unit tests verifying MIR timing, tempo, beat tracking, and meter discrimination.

Tests both the production backend (LIBROSA_BACKEND) and first-principles backend
(NAIVE_BACKEND) against deterministic synthetic audio fixtures.

Domain assertion helpers are grouped at the bottom of the module.
"""

import numpy as np
import pytest

from dj_dsp.synth import (
    generate_drum_loop,
    generate_syncopated_breakbeat,
    generate_tempo_drift_beat,
    generate_waltz_beat,
)
from dj_dsp.timing_data import Meter, MeterResult, OnsetEnvelope, TimingBackend
from dj_dsp.timing_librosa import LIBROSA_BACKEND
from dj_dsp.timing_naive import NAIVE_BACKEND

SAMPLE_RATE = 44100
HOP_LENGTH = 512
TEMPO_TOLERANCE_BPM = 2.0
BREAKBEAT_TOLERANCE_BPM = 3.0


# ---------------------------------------------------------------------------
# Test Audio Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_house_audio() -> tuple[np.ndarray, int]:
    audio = generate_drum_loop(bpm=128.0, duration_sec=8.0, fs=SAMPLE_RATE)
    return audio, SAMPLE_RATE


@pytest.fixture
def synthetic_waltz_audio() -> tuple[np.ndarray, int]:
    audio = generate_waltz_beat(bpm=120.0, duration_sec=8.0, fs=SAMPLE_RATE)
    return audio, SAMPLE_RATE


@pytest.fixture
def synthetic_breakbeat_audio() -> tuple[np.ndarray, int]:
    audio = generate_syncopated_breakbeat(bpm=140.0, duration_sec=8.0, fs=SAMPLE_RATE)
    return audio, SAMPLE_RATE


@pytest.fixture
def synthetic_drift_audio() -> tuple[np.ndarray, int]:
    audio = generate_tempo_drift_beat(
        start_bpm=120.0, end_bpm=130.0, duration_sec=8.0, fs=SAMPLE_RATE
    )
    return audio, SAMPLE_RATE


@pytest.fixture(params=[LIBROSA_BACKEND, NAIVE_BACKEND], ids=["librosa", "naive"])
def backend(request) -> TimingBackend:
    return request.param


# ---------------------------------------------------------------------------
# Tests (Specifications)
# ---------------------------------------------------------------------------


def test_onset_envelope_extraction(backend, synthetic_house_audio):
    """Verify onset envelope produces non-negative 1D flux aligned with duration."""
    y, sr = synthetic_house_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)

    assert_envelope_structure_valid(onset_env, y, sr, HOP_LENGTH)


def test_tempo_estimation_house(backend, synthetic_house_audio):
    """Verify 128 BPM house beat is detected within tolerance."""
    y, sr = synthetic_house_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    res = backend.estimate_tempo(onset_env)

    assert_tempo_close(res.bpm, 128.0, tolerance_bpm=TEMPO_TOLERANCE_BPM)
    assert_beat_interval_matches_bpm(res.beat_interval_frames, 128.0, sr, HOP_LENGTH)


def test_tempo_estimation_waltz(backend, synthetic_waltz_audio):
    """Verify 120 BPM 3/4 waltz tempo is detected within tolerance."""
    y, sr = synthetic_waltz_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    res = backend.estimate_tempo(onset_env)

    assert_tempo_close(res.bpm, 120.0, tolerance_bpm=TEMPO_TOLERANCE_BPM)


def test_tempo_estimation_breakbeat(backend, synthetic_breakbeat_audio):
    """Verify 140 BPM breakbeat tempo is detected despite syncopated snares."""
    y, sr = synthetic_breakbeat_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    res = backend.estimate_tempo(onset_env)

    assert_tempo_close(res.bpm, 140.0, tolerance_bpm=BREAKBEAT_TOLERANCE_BPM)


def test_meter_detection_four_on_floor(backend, synthetic_house_audio):
    """Verify four-on-the-floor kick/snare pattern is classified as 4/4 meter."""
    y, sr = synthetic_house_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    tempo_res = backend.estimate_tempo(onset_env)
    meter_res = backend.detect_meter(
        onset_env, beat_interval_frames=tempo_res.beat_interval_frames
    )

    assert_meter_verdict(meter_res, expected_meter=Meter.FOUR_FOUR)
    assert meter_res.ratio > 1.0


def test_meter_detection_waltz(backend, synthetic_waltz_audio):
    """Verify 3/4 waltz pattern is classified as 3/4 meter."""
    y, sr = synthetic_waltz_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    tempo_res = backend.estimate_tempo(onset_env)
    meter_res = backend.detect_meter(
        onset_env, beat_interval_frames=tempo_res.beat_interval_frames
    )

    assert_meter_verdict(meter_res, expected_meter=Meter.THREE_FOUR)
    assert meter_res.ratio < 1.0


def test_beat_tracking_monotonic_and_aligned(backend, synthetic_house_audio):
    """Verify dynamic beat tracking returns strictly increasing frames matching beat count."""
    y, sr = synthetic_house_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    track_res = backend.track_beats(onset_env)

    assert_beat_frames_valid(track_res.beat_frames, len(onset_env.values))
    assert_tempo_close(track_res.bpm, 128.0, tolerance_bpm=TEMPO_TOLERANCE_BPM)


def test_beat_tracking_adapts_to_tempo_drift(backend, synthetic_drift_audio):
    """Verify dynamic beat tracking reflects accelerating tempo across time."""
    y, sr = synthetic_drift_audio
    onset_env = backend.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    track_res = backend.track_beats(onset_env)

    assert_beat_intervals_shorten(track_res.beat_frames)


def test_backend_cross_agreement(synthetic_house_audio):
    """Verify librosa and naive backends agree on BPM within tolerance on standard audio."""
    y, sr = synthetic_house_audio
    env_librosa = LIBROSA_BACKEND.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)
    env_naive = NAIVE_BACKEND.compute_onset_envelope(y, sr, hop_length=HOP_LENGTH)

    res_librosa = LIBROSA_BACKEND.estimate_tempo(env_librosa)
    res_naive = NAIVE_BACKEND.estimate_tempo(env_naive)

    assert abs(res_librosa.bpm - res_naive.bpm) <= 1.5


# ---------------------------------------------------------------------------
# Domain-Specific Assertion Helpers
# ---------------------------------------------------------------------------


def assert_envelope_structure_valid(
    onset_env: OnsetEnvelope, y: np.ndarray, sr: int, hop_length: int
) -> None:
    assert isinstance(onset_env, OnsetEnvelope)
    assert onset_env.values.ndim == 1
    assert len(onset_env.values) > 0
    assert np.all(onset_env.values >= 0.0), "Onset envelope values must be non-negative"
    assert len(onset_env.times) == len(onset_env.values)
    expected_frames = int(np.ceil(len(y) / hop_length))
    assert abs(len(onset_env.values) - expected_frames) <= 2


def assert_tempo_close(
    actual_bpm: float, target_bpm: float, tolerance_bpm: float
) -> None:
    assert abs(actual_bpm - target_bpm) <= tolerance_bpm, (
        f"Expected BPM ~{target_bpm} ± {tolerance_bpm}, got {actual_bpm:.2f}"
    )


def assert_beat_interval_matches_bpm(
    beat_interval_frames: int,
    bpm: float,
    sr: int,
    hop_length: int,
    tolerance_frames: int = 2,
) -> None:
    expected_frames = round((60.0 / bpm) * (sr / hop_length))
    assert abs(beat_interval_frames - expected_frames) <= tolerance_frames, (
        f"Expected tau_0 ~{expected_frames} frames, got {beat_interval_frames}"
    )


def assert_meter_verdict(result: MeterResult, expected_meter: Meter) -> None:
    assert isinstance(result, MeterResult)
    assert result.meter == expected_meter, (
        f"Expected meter {expected_meter.value}, got {result.meter.value} (ratio={result.ratio:.3f})"
    )
    assert 0.0 <= result.confidence <= 1.0


def assert_beat_frames_valid(beat_frames: np.ndarray, total_frames: int) -> None:
    assert len(beat_frames) > 0
    assert np.all(beat_frames >= 0)
    assert np.all(beat_frames < total_frames)
    diffs = np.diff(beat_frames)
    assert np.all(diffs > 0), "Beat frame indices must be strictly increasing"


def assert_beat_intervals_shorten(beat_frames: np.ndarray) -> None:
    diffs = np.diff(beat_frames)
    assert len(diffs) >= 4, "Need at least 4 intervals to evaluate tempo drift"
    midpoint = len(diffs) // 2
    early_mean_interval = float(np.mean(diffs[:midpoint]))
    late_mean_interval = float(np.mean(diffs[midpoint:]))
    assert late_mean_interval < early_mean_interval, (
        f"Expected accelerating tempo: early mean interval ({early_mean_interval:.1f}) "
        f"should exceed late mean interval ({late_mean_interval:.1f})"
    )
