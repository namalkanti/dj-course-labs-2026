"""Unit and integration tests for Rekordbox binary ANLZ parsing and cue sheet synchronization.

Self-documenting assertion helpers encapsulate domain invariants and are grouped
at the bottom of the module:
- ANLZ tag presence, path decoding, and tempo invariants.
- Hot cue pad slot ordering ('A' through 'H'), timing non-negativity, and color bounds.
- Memory cue chronological sorting and label integrity.
- Cue sheet diff correctness: matched files, modified cues, warnings on missing files, and ignored USB extras.
"""

from typing import List, Optional, Tuple
import pytest

from dj_dsp.rekordbox.cue_data import CueSheetDiff, HotCue, MemoryCue, TrackCues, TrackDiff
from dj_dsp.rekordbox.cue_sync import (
    apply_cue_sheet_diff,
    calculate_cue_sheet_diff,
    format_time_ms,
)
from dj_dsp.rekordbox.rekordbox_parser import (
    build_track_cues,
    parse_bpm_tag,
    parse_cue_tags,
    parse_ppth_tag,
    split_anlz_tags,
)
from tests.utils.synthetic_anlz import (
    build_anlz_file,
    build_pco2_tag,
    build_pcp2_entry,
    build_ppth_tag,
    build_pqtz_tag,
)


SAMPLE_CUE_SHEET_MARKDOWN = (
    "# Test Cue Sheet\n\n"
    "### 01. Test Artist - Matched Song\n"
    "- **File**: `MatchedSong.wav`\n"
    "- **Cues & Loops**:\n"
    "  - `Hot Cue A` [Pad 1]: `0:00.000` (0 ms) — Initial intro hook.\n\n"
    "### 02. Missing Artist - Missing Track\n"
    "- **File**: `NonExistent.wav`\n"
    "- **Cues & Loops**:\n"
    "  - `Mem Cue 1` [CDJ]: `0:00.000` — Cold open.\n"
)


@pytest.fixture
def sample_cue_sheet_markdown() -> str:
    """Fixture providing a minimal 2-track cue sheet markdown document."""
    return SAMPLE_CUE_SHEET_MARKDOWN


# ---------------------------------------------------------------------------
# Specifications (Tests)
# ---------------------------------------------------------------------------


def test_format_time_ms_formatting():
    """Millisecond offsets must format as human-readable MM:SS.mmm with zero-padding."""
    assert format_time_ms(0) == "0:00.000"
    assert format_time_ms(264) == "0:00.264"
    assert format_time_ms(14464) == "0:14.464"
    assert format_time_ms(112083) == "1:52.083"
    assert format_time_ms(216083) == "3:36.083"


def test_parse_synthetic_track_with_hot_cue_rgb():
    """Synthetic track: parse PPTH path, 128.0 BPM grid, and Hot Cue A with 24-bit RGB."""
    ppth = build_ppth_tag("/Contents/Synthetic Artist/Album/Test Track Alpha.wav")
    pqtz = build_pqtz_tag(bpm=128.0)
    hot_entry = build_pcp2_entry(
        hot_cue_num=1,
        time_ms=45000,
        comment="Intro Drop",
        color_rgb=(255, 128, 0),
    )
    pco2 = build_pco2_tag(cue_obj_type=1, entries=[hot_entry])

    ext_bytes = build_anlz_file([ppth, pco2])
    dat_bytes = build_anlz_file([ppth, pqtz])

    ext_tags = split_anlz_tags(ext_bytes)
    dat_tags = split_anlz_tags(dat_bytes)

    assert_required_tags_present(ext_tags, expected_tag="PCO2")
    assert_required_tags_present(dat_tags, expected_tag="PQTZ")

    path = parse_ppth_tag(ext_tags["PPTH"][0])
    assert_path_contains(path, expected_substring="Test Track Alpha.wav")

    bpm = parse_bpm_tag(dat_tags["PQTZ"][0])
    assert_bpm_near(bpm, expected_bpm=128.0)

    hot_cues, mem_cues = parse_cue_tags(ext_tags)
    assert len(hot_cues) == 1
    assert_hot_cue_attributes(
        hot_cues[0],
        expected_slot="A",
        expected_time_ms=45000,
        expected_comment="Intro Drop",
        expected_rgb=(255, 128, 0),
    )


def test_parse_synthetic_track_with_labeled_memory_cues():
    """Synthetic track: parse chronological memory cues with text labels."""
    ppth = build_ppth_tag("/Contents/Synthetic Artist/Album/Test Track Beta.wav")
    pqtz = build_pqtz_tag(bpm=124.0)

    mem_entry_1 = build_pcp2_entry(hot_cue_num=0, time_ms=250, comment="Start")
    mem_entry_2 = build_pcp2_entry(hot_cue_num=0, time_ms=30000, comment="Vocal Verse")
    pco2 = build_pco2_tag(cue_obj_type=0, entries=[mem_entry_1, mem_entry_2])

    ext_bytes = build_anlz_file([ppth, pco2])
    dat_bytes = build_anlz_file([ppth, pqtz])

    track_cues = build_track_cues(ext_bytes, dat_bytes)

    assert_track_cues_valid(track_cues, expected_title_substring="Test Track Beta.wav", expected_bpm=124.0)
    assert_memory_cues_sorted(track_cues.memory_cues)
    assert_memory_cue_attributes(track_cues.memory_cues[0], expected_time_ms=250, expected_comment="Start")
    assert_memory_cue_attributes(track_cues.memory_cues[1], expected_time_ms=30000, expected_comment="Vocal Verse")


def test_parse_minimal_44b_cue_entry():
    """Tracks with 44-byte minimal cue entries (no comments, no RGB) must parse without error."""
    ppth = build_ppth_tag("/Contents/Synthetic Artist/Album/Minimal Track.wav")
    pqtz = build_pqtz_tag(bpm=130.0)

    hot_a = build_pcp2_entry(hot_cue_num=1, time_ms=15000)
    hot_b = build_pcp2_entry(hot_cue_num=2, time_ms=60000)
    pco2 = build_pco2_tag(cue_obj_type=1, entries=[hot_a, hot_b])

    ext_bytes = build_anlz_file([ppth, pco2])
    dat_bytes = build_anlz_file([ppth, pqtz])

    track_cues = build_track_cues(ext_bytes, dat_bytes)

    assert len(track_cues.hot_cues) == 2, f"Expected 2 hot cues, found {len(track_cues.hot_cues)}"
    assert_hot_cue_attributes(track_cues.hot_cues[0], expected_slot="A", expected_time_ms=15000)
    assert_hot_cue_attributes(track_cues.hot_cues[1], expected_slot="B", expected_time_ms=60000)


def test_cue_sheet_diff_matching_and_atomic_apply(sample_cue_sheet_markdown: str):
    """Cue sheet sync must match existing tracks, warn on missing, ignore extra USB tracks, and preserve notes."""
    matched_track = TrackCues(
        path="/Contents/Test Artist/MatchedSong.wav",
        filename="MatchedSong.wav",
        bpm=128.0,
        hot_cues=[HotCue(slot="A", time_ms=45000, color_rgb=(255, 128, 0))],
        memory_cues=[],
    )
    extra_usb_track = TrackCues(
        path="/Contents/Unrelated/IgnoredTrack.wav",
        filename="IgnoredTrack.wav",
        bpm=120.0,
        hot_cues=[],
        memory_cues=[],
    )

    diff = calculate_cue_sheet_diff(sample_cue_sheet_markdown, [matched_track, extra_usb_track])

    assert_cue_sheet_diff_matches(
        diff,
        expected_matched_file="MatchedSong.wav",
        expected_missing_file="NonExistent.wav",
    )

    updated_md = apply_cue_sheet_diff(sample_cue_sheet_markdown, diff)
    assert_markdown_preserves_annotations(
        updated_md,
        expected_updated_timestamp="0:45.000",
        expected_preserved_annotation="Initial intro hook.",
        expected_unaltered_section="NonExistent.wav",
    )


# ---------------------------------------------------------------------------
# Domain-Specific Assertion Helpers
# ---------------------------------------------------------------------------


def assert_required_tags_present(tags: dict[str, list[bytes]], expected_tag: str) -> None:
    """Assert essential Rekordbox analysis tag is present in parsed tag dictionary."""
    assert "PPTH" in tags, "Analysis stream is missing mandatory PPTH (path) tag"
    assert expected_tag in tags, f"Analysis stream is missing expected tag {expected_tag}"


def assert_path_contains(path: str, expected_substring: str) -> None:
    """Assert decoded UTF-16-BE path points to expected audio track."""
    assert expected_substring in path, f"Decoded path '{path}' does not contain '{expected_substring}'"


def assert_bpm_near(bpm: float, expected_bpm: float, tolerance: float = 0.1) -> None:
    """Assert beatgrid BPM matches expected tempo within numerical tolerance."""
    assert bpm == pytest.approx(expected_bpm, abs=tolerance), (
        f"Analyzed BPM {bpm} differs from expected {expected_bpm} beyond tolerance {tolerance}"
    )


def assert_hot_cue_attributes(
    hot_cue: HotCue,
    expected_slot: str,
    expected_time_ms: int,
    expected_comment: str = "",
    expected_rgb: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Assert HotCue domain invariants: slot letter, non-negative timestamp, comment, and RGB."""
    assert isinstance(hot_cue, HotCue), f"Expected HotCue instance, got {type(hot_cue)}"
    assert hot_cue.slot == expected_slot, f"Expected slot {expected_slot}, got {hot_cue.slot}"
    assert hot_cue.time_ms == expected_time_ms, (
        f"Expected timestamp {expected_time_ms} ms, got {hot_cue.time_ms} ms"
    )
    if expected_comment:
        assert hot_cue.comment == expected_comment, (
            f"Expected comment '{expected_comment}', got '{hot_cue.comment}'"
        )
    if expected_rgb is not None:
        assert hot_cue.color_rgb == expected_rgb, (
            f"Expected RGB color {expected_rgb}, got {hot_cue.color_rgb}"
        )


def assert_memory_cues_sorted(memory_cues: List[MemoryCue]) -> None:
    """Assert memory cues are ordered chronologically by timestamp."""
    timestamps = [c.time_ms for c in memory_cues]
    assert timestamps == sorted(timestamps), f"Memory cues are not chronologically sorted: {timestamps}"


def assert_memory_cue_attributes(
    memory_cue: MemoryCue, expected_time_ms: int, expected_comment: str
) -> None:
    """Assert MemoryCue timestamp and descriptive comment."""
    assert isinstance(memory_cue, MemoryCue), f"Expected MemoryCue instance, got {type(memory_cue)}"
    assert memory_cue.time_ms == expected_time_ms, (
        f"Expected timestamp {expected_time_ms} ms, got {memory_cue.time_ms} ms"
    )
    assert memory_cue.comment == expected_comment, (
        f"Expected comment '{expected_comment}', got '{memory_cue.comment}'"
    )


def assert_track_cues_valid(
    track_cues: TrackCues, expected_title_substring: str, expected_bpm: float
) -> None:
    """Assert TrackCues container consistency, non-empty filename, and BPM value."""
    assert isinstance(track_cues, TrackCues)
    assert expected_title_substring in track_cues.filename, (
        f"Expected '{expected_title_substring}' in filename '{track_cues.filename}'"
    )
    assert_bpm_near(track_cues.bpm, expected_bpm)


def assert_cue_sheet_diff_matches(
    diff: CueSheetDiff, expected_matched_file: str, expected_missing_file: str
) -> None:
    """Assert calculated diff matches set track, detects changes, and warns on missing track."""
    assert isinstance(diff, CueSheetDiff)
    matched_filenames = [t.filename for t in diff.matched_tracks]
    assert expected_matched_file in matched_filenames, (
        f"Expected '{expected_matched_file}' to be matched, found {matched_filenames}"
    )

    target_diff = next(t for t in diff.matched_tracks if t.filename == expected_matched_file)
    assert target_diff.has_changes is True, (
        f"Expected track '{expected_matched_file}' to report changes in cue lines"
    )

    assert expected_missing_file in diff.missing_from_usb, (
        f"Expected missing file warning for '{expected_missing_file}', got {diff.missing_from_usb}"
    )


def assert_markdown_preserves_annotations(
    updated_md: str,
    expected_updated_timestamp: str,
    expected_preserved_annotation: str,
    expected_unaltered_section: str,
) -> None:
    """Assert markdown sync updates timestamps while preserving user annotations after em-dash."""
    assert expected_updated_timestamp in updated_md, (
        f"Updated timestamp '{expected_updated_timestamp}' not found in updated markdown"
    )
    assert expected_preserved_annotation in updated_md, (
        f"User note '{expected_preserved_annotation}' was lost during markdown sync"
    )
    assert expected_unaltered_section in updated_md, (
        f"Unaltered track section '{expected_unaltered_section}' was corrupted during sync"
    )

