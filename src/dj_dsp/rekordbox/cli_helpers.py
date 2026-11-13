"""CLI helpers for Rekordbox USB auto-discovery, serialization, and reporting."""

import getpass
import json
import os
from pathlib import Path
from typing import Any, List, Optional

from dj_dsp.rekordbox.cue_data import CueSheetDiff, HotCue, MemoryCue, TrackCues
from dj_dsp.rekordbox.cue_sync import format_time_ms


def find_pioneer_usb_roots() -> list[Path]:
    """Auto-detect mounted Pioneer USB media paths across common Linux and macOS mount points.

    Checks:
    - /run/media/$USER/*/PIONEER
    - /media/$USER/*/PIONEER
    - /run/media/$USER/* (direct check for USBANLZ or PIONEER subdir)
    - /media/$USER/*
    - /Volumes/*/PIONEER
    """
    candidates: list[Path] = []
    user = os.environ.get("USER") or getpass.getuser()

    search_roots = [
        Path(f"/run/media/{user}"),
        Path(f"/media/{user}"),
        Path("/Volumes"),
    ]

    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            for mount_entry in root.iterdir():
                if not mount_entry.is_dir():
                    continue
                # Case 1: mount/PIONEER exists
                pioneer_dir = mount_entry / "PIONEER"
                if pioneer_dir.is_dir() and pioneer_dir not in candidates:
                    candidates.append(pioneer_dir)
                    continue

                # Case 2: mount itself contains USBANLZ or rekordbox
                if (mount_entry / "USBANLZ").is_dir() and mount_entry not in candidates:
                    candidates.append(mount_entry)
        except (PermissionError, OSError):
            continue

    return sorted(candidates)


def resolve_pioneer_root(usb_path: Optional[Path] = None) -> Path:
    """Resolve Pioneer analysis root from optional user path or system discovery.

    Raises:
        FileNotFoundError: If path not found or no mounted media detected.
    """
    if usb_path is not None:
        path = usb_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Specified USB path does not exist: {path}")

        # If user passed mount point containing PIONEER
        if (path / "PIONEER").is_dir():
            return path / "PIONEER"
        return path

    detected = find_pioneer_usb_roots()
    if not detected:
        raise FileNotFoundError(
            "No Pioneer USB media detected. Please provide --usb-path explicitly."
        )
    return detected[0]


def serialize_hot_cue(cue: HotCue) -> dict[str, Any]:
    """Serialize HotCue to JSON-compatible dictionary."""
    return {
        "slot": cue.slot,
        "time_ms": cue.time_ms,
        "formatted_time": format_time_ms(cue.time_ms),
        "comment": cue.comment,
        "color_rgb": list(cue.color_rgb) if cue.color_rgb else None,
        "is_loop": cue.is_loop,
        "loop_end_ms": cue.loop_end_ms,
        "formatted_loop_end": format_time_ms(cue.loop_end_ms) if cue.loop_end_ms else None,
    }


def serialize_memory_cue(cue: MemoryCue) -> dict[str, Any]:
    """Serialize MemoryCue to JSON-compatible dictionary."""
    return {
        "time_ms": cue.time_ms,
        "formatted_time": format_time_ms(cue.time_ms),
        "comment": cue.comment,
    }


def serialize_track_cues(track: TrackCues, include_metadata: bool = False) -> dict[str, Any]:
    """Serialize TrackCues to JSON-compatible dictionary."""
    payload: dict[str, Any] = {
        "filename": track.filename,
        "path": track.path,
        "total_cues": track.total_cues,
        "hot_cues": [serialize_hot_cue(h) for h in track.hot_cues],
        "memory_cues": [serialize_memory_cue(m) for m in track.memory_cues],
    }
    if include_metadata:
        payload["bpm"] = track.bpm
    return payload


def serialize_diff_summary(diff: CueSheetDiff) -> dict[str, Any]:
    """Serialize CueSheetDiff for machine-readable status reporting."""
    return {
        "total_matched": len(diff.matched_tracks),
        "total_changed": len(diff.changed_tracks),
        "missing_from_usb": diff.missing_from_usb,
        "changes": [
            {
                "title": t.title,
                "filename": t.filename,
                "old_cues": t.old_cue_lines,
                "new_cues": t.new_cue_lines,
            }
            for t in diff.changed_tracks
        ],
    }


def format_text_report(
    tracks: List[TrackCues],
    diff: Optional[CueSheetDiff] = None,
    include_metadata: bool = False,
) -> str:
    """Format human-readable terminal output."""
    lines: list[str] = []

    for track in tracks:
        bpm_str = f" [BPM: {track.bpm:.2f}]" if include_metadata else ""
        lines.append(f"Track: {track.filename}{bpm_str}")
        lines.append(f"  Path: {track.path}")
        if not track.hot_cues and not track.memory_cues:
            lines.append("  (No cues found)")
        for h in track.hot_cues:
            pad_num = ord(h.slot.upper()) - ord("A") + 1
            if h.is_loop and h.loop_end_ms:
                loop_str = f"{format_time_ms(h.time_ms)} – {format_time_ms(h.loop_end_ms)}"
                desc = f"Saved Loop {h.slot} [Pad {pad_num}]: {loop_str}"
            else:
                desc = f"Hot Cue {h.slot} [Pad {pad_num}]: {format_time_ms(h.time_ms)} ({h.time_ms:,} ms)"
            if h.comment:
                desc += f" — {h.comment}"
            lines.append(f"  - {desc}")
        for m in track.memory_cues:
            desc = f"Mem Cue [CDJ]: {format_time_ms(m.time_ms)}"
            if m.comment:
                desc += f" — {m.comment}"
            lines.append(f"  - {desc}")
        lines.append("")

    if diff is not None:
        lines.append("--- Cue Sheet Synchronization ---")
        lines.append(f"Matched tracks: {len(diff.matched_tracks)}")
        lines.append(f"Changed tracks: {len(diff.changed_tracks)}")
        if diff.missing_from_usb:
            lines.append(f"Missing from USB: {', '.join(diff.missing_from_usb)}")
        for ch in diff.changed_tracks:
            lines.append(f"\n[Updated] {ch.title} ({ch.filename}):")
            for line in ch.new_cue_lines:
                lines.append(f"  + {line.strip()}")

    return "\n".join(lines).strip()
