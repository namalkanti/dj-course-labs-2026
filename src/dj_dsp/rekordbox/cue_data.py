"""Domain models for Rekordbox analysis files and cue registry synchronization.

All cue positions are stored as millisecond integers measured from audio track start.
Hot cues and memory cues are explicitly separated to preserve hardware semantics:
- HotCue: Performance pad trigger (A-H), optional color RGB, optional loop boundary.
- MemoryCue: Chronological reference marker with optional text comment, no pad assignment.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class HotCue:
    """Rekordbox Hot Cue trigger mapped to hardware performance pads A through H.

    Attributes:
        slot: Pad letter identifier ('A' through 'H'), mapped 1-to-1 from 1-indexed integers 1..8.
        time_ms: Timestamp in milliseconds from track start.
        comment: Optional descriptive label saved in Rekordbox.
        color_rgb: Optional 24-bit RGB tuple (red, green, blue) as configured in Rekordbox.
        is_loop: True if the cue is a saved loop rather than a single jump point.
        loop_end_ms: Timestamp in milliseconds of loop end if is_loop is True.
    """

    slot: str
    time_ms: int
    comment: str = ""
    color_rgb: Optional[Tuple[int, int, int]] = None
    is_loop: bool = False
    loop_end_ms: Optional[int] = None


@dataclass(frozen=True)
class MemoryCue:
    """Rekordbox Memory Cue navigation marker.

    Attributes:
        time_ms: Timestamp in milliseconds from track start.
        comment: Optional text label (e.g. 'Start', 'Drop', 'Vocal Hand-off').
    """

    time_ms: int
    comment: str = ""


@dataclass(frozen=True)
class TrackCues:
    """Container holding all parsed cues and analysis metadata for a single track.

    Attributes:
        path: Relative audio file path as recorded in the ANLZ PPTH tag.
        filename: Base audio file name.
        bpm: Rekordbox analyzed tempo from beatgrid.
        hot_cues: Sorted list of HotCue instances ordered by pad slot ('A' -> 'H').
        memory_cues: Sorted list of MemoryCue instances ordered chronologically by time_ms.
    """

    path: str
    filename: str
    bpm: float
    hot_cues: list[HotCue] = field(default_factory=list)
    memory_cues: list[MemoryCue] = field(default_factory=list)
    mtime: float = 0.0

    @property
    def total_cues(self) -> int:
        """Total count of hot cues and memory cues assigned to this track."""
        return len(self.hot_cues) + len(self.memory_cues)


@dataclass(frozen=True)
class TrackDiff:
    """Represents a markdown diff for a single track within the cue sheet.

    Attributes:
        title: Human-readable track header from markdown (e.g. '01. Illenium - Hearts on Fire').
        filename: Audio filename matched in the cue sheet.
        old_cue_lines: Existing markdown bullet lines under '- **Cues & Loops**:'.
        new_cue_lines: Proposed replacement markdown bullet lines derived from TrackCues.
    """

    title: str
    filename: str
    old_cue_lines: list[str]
    new_cue_lines: list[str]

    @property
    def has_changes(self) -> bool:
        """True if the proposed cue lines differ from existing markdown lines."""
        return self.old_cue_lines != self.new_cue_lines


@dataclass(frozen=True)
class CueSheetDiff:
    """Summary of cue sheet synchronization across all tracks.

    Attributes:
        matched_tracks: List of TrackDiff objects for all tracks found both in cue sheet and USB.
        missing_from_usb: List of track file names present in the cue sheet but not located on USB.
    """

    matched_tracks: list[TrackDiff]
    missing_from_usb: list[str] = field(default_factory=list)

    @property
    def changed_tracks(self) -> list[TrackDiff]:
        """Subset of matched tracks that have modified cue lines."""
        return [t for t in self.matched_tracks if t.has_changes]
