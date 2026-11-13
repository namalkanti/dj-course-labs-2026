"""Cue sheet markdown synchronization and atomic updates."""

import os
from pathlib import Path
import re
import tempfile
from typing import List, Optional

from dj_dsp.rekordbox.cue_data import CueSheetDiff, HotCue, MemoryCue, TrackCues, TrackDiff


def format_time_ms(time_ms: int) -> str:
    """Format millisecond integer into standard cue sheet timestamp string MM:SS.mmm.

    Args:
        time_ms: Milliseconds from audio start.

    Returns:
        Formatted timestamp string (e.g. 112083 -> '1:52.083', 264 -> '0:00.264').
    """
    total_seconds = time_ms // 1000
    rem_ms = time_ms % 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}.{rem_ms:03d}"


def _format_cue_line(cue: HotCue | MemoryCue, existing_annotation: str = "") -> str:
    """Format a single cue point line for the cue sheet."""
    suffix = f" — {existing_annotation}" if existing_annotation else ""
    if isinstance(cue, HotCue):
        pad_num = ord(cue.slot.upper()) - ord("A") + 1
        label = f"Hot Cue {cue.slot}"
        time_str = format_time_ms(cue.time_ms)
        ms_str = f"{cue.time_ms:,} ms"
        comment = cue.comment or existing_annotation
        if cue.is_loop and cue.loop_end_ms:
            end_time_str = format_time_ms(cue.loop_end_ms)
            line = f"  - `Saved Loop {cue.slot}` [Pad {pad_num}]: `{time_str}` – `{end_time_str}`"
        else:
            line = f"  - `{label}` [Pad {pad_num}]: `{time_str}` ({ms_str})"
        if comment:
            line += f" — {comment}." if not comment.endswith(".") else f" — {comment}"
        return line
    else:
        time_str = format_time_ms(cue.time_ms)
        comment = cue.comment or existing_annotation
        line = f"  - `Mem Cue` [CDJ]: `{time_str}`"
        if comment:
            line += f" — {comment}." if not comment.endswith(".") else f" — {comment}"
        return line


def _parse_track_sections(markdown_text: str) -> list[tuple[str, str, list[str], int, int]]:
    """Parse markdown text into track sections.

    Returns list of tuples: (title, filename, cue_lines, start_idx, end_idx)
    """
    lines = markdown_text.splitlines()
    sections = []

    header_re = re.compile(r"^###\s+(.*)")
    file_re = re.compile(r"^-\s+\*\*File\*\*:\s*`?([^`\n]+)`?")
    cue_header_re = re.compile(r"^-\s+\*\*Cues & Loops\*\*:")

    i = 0
    while i < len(lines):
        m_head = header_re.match(lines[i])
        if m_head:
            title = m_head.group(1).strip()
            section_start = i
            filename = ""
            cue_lines = []
            cue_start = -1
            cue_end = -1
            i += 1
            while i < len(lines) and not header_re.match(lines[i]):
                m_file = file_re.match(lines[i])
                if m_file:
                    filename = m_file.group(1).strip()
                elif cue_header_re.match(lines[i]):
                    cue_start = i + 1
                    j = cue_start
                    while j < len(lines) and lines[j].startswith("  - "):
                        cue_lines.append(lines[j])
                        j += 1
                    cue_end = j
                    i = j - 1
                i += 1
            if filename:
                sections.append((title, filename, cue_lines, cue_start, cue_end))
        else:
            i += 1

    return sections


def _split_stem_suffix(filename: str) -> tuple[str, int]:
    """Split filename into base stem and integer collision suffix.

    e.g. 'Hearts on Fire-4.aiff' -> ('Hearts on Fire.aiff', 4)
         'Hearts on Fire.aiff'   -> ('Hearts on Fire.aiff', 0)
    """
    path = Path(filename)
    stem = path.stem
    ext = path.suffix
    m = re.match(r"^(.*?)-(\d+)$", stem)
    if m:
        base_name = f"{m.group(1)}{ext}"
        suffix_num = int(m.group(2))
        return base_name, suffix_num
    return filename, 0


def calculate_cue_sheet_diff(
    current_markdown: str, extracted_tracks: List[TrackCues]
) -> CueSheetDiff:
    """Calculate cue updates against an existing markdown cue sheet."""
    # Group extracted tracks by normalized base filename
    # Sort candidates by (collision_suffix, mtime) ascending so latest is last
    tracks_by_base: dict[str, TrackCues] = {}
    for t in sorted(
        extracted_tracks,
        key=lambda x: (_split_stem_suffix(x.filename)[1], x.mtime),
    ):
        base_name, _ = _split_stem_suffix(t.filename)
        tracks_by_base[base_name] = t
        tracks_by_base[t.filename] = t

    sections = _parse_track_sections(current_markdown)

    matched_diffs: list[TrackDiff] = []
    missing_from_usb: list[str] = []

    for title, filename, old_cue_lines, _, _ in sections:
        if filename in tracks_by_base:
            track = tracks_by_base[filename]
            new_cue_lines = []

            # Extract existing annotations mapped by cue slot/label if present
            annotations: dict[str, str] = {}
            for line in old_cue_lines:
                # e.g. "  - `Hot Cue A` [Pad 1]: `0:00.000` (0 ms) — Initial intro hook."
                m_slot = re.search(r"Hot Cue ([A-H])", line)
                m_note = re.search(r"—\s*(.*)", line)
                if m_slot and m_note:
                    annotations[m_slot.group(1)] = m_note.group(1).strip()
                m_mem = re.search(r"Mem Cue", line)
                if m_mem and m_note:
                    annotations["MEM"] = m_note.group(1).strip()

            for hc in track.hot_cues:
                existing_note = annotations.get(hc.slot, "")
                new_cue_lines.append(_format_cue_line(hc, existing_annotation=existing_note))

            for mc in track.memory_cues:
                existing_note = annotations.get("MEM", "")
                new_cue_lines.append(_format_cue_line(mc, existing_annotation=existing_note))

            matched_diffs.append(
                TrackDiff(
                    title=title,
                    filename=filename,
                    old_cue_lines=old_cue_lines,
                    new_cue_lines=new_cue_lines,
                )
            )
        else:
            missing_from_usb.append(filename)

    return CueSheetDiff(matched_tracks=matched_diffs, missing_from_usb=missing_from_usb)


def apply_cue_sheet_diff(current_markdown: str, diff: CueSheetDiff) -> str:
    """Apply computed cue sheet diffs cleanly to generate updated markdown content."""
    lines = current_markdown.splitlines()
    sections = _parse_track_sections(current_markdown)

    diff_map = {d.filename: d for d in diff.matched_tracks if d.has_changes}

    # Process sections from bottom to top to preserve line indices
    sections_sorted = sorted(sections, key=lambda s: s[3], reverse=True)

    for _, filename, _, cue_start, cue_end in sections_sorted:
        if filename in diff_map and cue_start != -1:
            track_diff = diff_map[filename]
            lines[cue_start:cue_end] = track_diff.new_cue_lines

    trailing_newline = "\n" if current_markdown.endswith("\n") else ""
    return "\n".join(lines) + trailing_newline


def write_atomic(target_path: Path, content: str) -> None:
    """Safely overwrite target markdown file atomically using a temporary file."""
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=target_dir, delete=False, encoding="utf-8") as tf:
        tf.write(content)
        temp_name = tf.name

    os.replace(temp_name, target_path)
