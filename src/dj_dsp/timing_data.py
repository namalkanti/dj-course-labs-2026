"""Pure data structures, domain enums, and functional interfaces for MIR timing analysis.

Pipeline Flow:
1. Raw Audio -> OnsetEnvelope: Transient extraction via spectral flux.
2. OnsetEnvelope -> TempoResult: Global tempo (BPM) and beat interval lag (tau_0).
3. OnsetEnvelope -> BeatTrackResult: Dynamic beat locations across time via DP trellis.
4. OnsetEnvelope -> MeterResult: Metric classification (Meter.FOUR_FOUR vs Meter.THREE_FOUR).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol
import numpy as np


class Meter(Enum):
    """Musical time signature classification."""

    FOUR_FOUR = "4/4"
    THREE_FOUR = "3/4"


# ---------------------------------------------------------------------------
# Chronological Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnsetEnvelope:
    """1D transient strength envelope extracted from audio via spectral flux.

    Fields:
        values: 1D array of onset strength per STFT frame (shape: N_frames).
        times: 1D array of frame timestamps in seconds.
        fs: Audio sample rate in Hz (e.g. 44100). Note: mirrors librosa's 'sr'.
        hop_length: Hop size between STFT frames in audio samples (e.g. 512).
    """

    values: np.ndarray
    times: np.ndarray
    fs: int
    hop_length: int

    @property
    def frame_rate(self) -> float:
        """Rate of envelope frames per second (fs / hop_length)."""
        return self.fs / self.hop_length

    @property
    def duration_sec(self) -> float:
        """Duration of the analyzed signal in seconds."""
        return float(self.times[-1]) if len(self.times) > 0 else 0.0


@dataclass(frozen=True)
class TempoResult:
    """Global tempo and nominal beat interval extracted from onset autocorrelation.

    Fields:
        bpm: Estimated beats per minute.
        beat_interval_frames: Integer lag tau_0 in STFT frames corresponding to one beat.
    """

    bpm: float
    beat_interval_frames: int


@dataclass(frozen=True)
class TemporalBeatAlignment:
    """Dynamic beat locations aligned across time via DP trellis tracking.

    Fields:
        beat_frames: 1D array of STFT frame indices for each detected beat.
        beat_times: 1D array of timestamps in seconds for each detected beat.
        bpm: Effective global BPM derived from the median inter-beat interval.
    """

    beat_frames: np.ndarray
    beat_times: np.ndarray
    bpm: float


@dataclass(frozen=True)
class MeterResult:
    """Time signature classification comparing 4-beat vs 3-beat bar harmonics.

    Fields:
        meter: Musical time signature enum (Meter.FOUR_FOUR or Meter.THREE_FOUR).
        ratio: Raw autocorrelation prominence ratio R(4 * tau_0) / R(3 * tau_0).
               Values > 1.0 indicate 4/4 meter; values < 1.0 indicate 3/4 meter.
        confidence: Normalized confidence metric in [0.0, 1.0].
    """

    meter: Meter
    ratio: float
    confidence: float


# ---------------------------------------------------------------------------
# Functional Interfaces (Protocols) & Backend Record
# ---------------------------------------------------------------------------


class ComputeOnsetFn(Protocol):
    """Compute 1D onset strength envelope from raw audio."""

    def __call__(
        self,
        audio: np.ndarray,
        fs: int,
        hop_length: int = 512,
    ) -> OnsetEnvelope: ...


class EstimateTempoFn(Protocol):
    """Estimate global BPM and nominal beat interval from an onset envelope."""

    def __call__(
        self,
        onset_env: OnsetEnvelope,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
    ) -> TempoResult: ...


class AlignBeatsFn(Protocol):
    """Align dynamic beat locations using dynamic programming (Ellis trellis)."""

    def __call__(
        self,
        onset_env: OnsetEnvelope,
        bpm: float | None = None,
    ) -> TemporalBeatAlignment: ...


class DetectMeterFn(Protocol):
    """Classify 3/4 vs 4/4 meter by sampling autocorrelation at bar harmonic lags."""

    def __call__(
        self,
        onset_env: OnsetEnvelope,
        beat_interval_frames: int | None = None,
    ) -> MeterResult: ...


@dataclass(frozen=True)
class TimingBackend:
    """Immutable bundle of pure functional timing algorithms."""

    name: str
    compute_onset_envelope: ComputeOnsetFn
    estimate_tempo: EstimateTempoFn
    align_beats: AlignBeatsFn
    detect_meter: DetectMeterFn
