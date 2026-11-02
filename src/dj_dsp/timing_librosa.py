"""Production timing analysis using canonical MIR libraries (librosa).

Implements the TimingBackend protocol using librosa.onset, librosa.feature.tempo,
and librosa.beat.beat_track.
"""

import numpy as np

from dj_dsp.timing_data import (
    BeatTrackResult,
    MeterResult,
    OnsetEnvelope,
    TempoResult,
    TimingBackend,
)


def compute_onset_envelope(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> OnsetEnvelope:
    """Compute onset strength envelope using librosa."""
    raise NotImplementedError("Stub: librosa compute_onset_envelope")


def estimate_tempo(
    onset_env: OnsetEnvelope,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
) -> TempoResult:
    """Estimate global BPM and nominal beat interval using librosa."""
    raise NotImplementedError("Stub: librosa estimate_tempo")


def track_beats(
    onset_env: OnsetEnvelope,
    bpm: float | None = None,
) -> BeatTrackResult:
    """Track dynamic beat locations using librosa.beat.beat_track."""
    raise NotImplementedError("Stub: librosa track_beats")


def detect_meter(
    onset_env: OnsetEnvelope,
    beat_interval_frames: int | None = None,
) -> MeterResult:
    """Classify 3/4 vs 4/4 meter by sampling autocorrelation harmonics."""
    raise NotImplementedError("Stub: librosa detect_meter")


LIBROSA_BACKEND = TimingBackend(
    name="librosa",
    compute_onset_envelope=compute_onset_envelope,
    estimate_tempo=estimate_tempo,
    track_beats=track_beats,
    detect_meter=detect_meter,
)
