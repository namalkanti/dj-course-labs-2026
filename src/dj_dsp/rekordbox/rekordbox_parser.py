"""Rekordbox ANLZ binary parser for cue points, beatgrids, and track metadata."""

from pathlib import Path
import struct
from typing import List, Optional, Tuple

from dj_dsp.rekordbox.cue_data import HotCue, MemoryCue, TrackCues

# Mapping 1-based hot cue index from Rekordbox binary to pad slot letter
HOT_CUE_SLOTS = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
}


def split_anlz_tags(raw_data: bytes) -> dict[str, list[bytes]]:
    """Parse binary ANLZ file stream and split into individual tagged chunks.

    Args:
        raw_data: Binary byte buffer of an ANLZ file starting with PMAI header.

    Returns:
        Dictionary mapping 4-character ASCII tag names (e.g. 'PPTH', 'PCO2', 'PQTZ')
        to a list of raw tag byte payloads (including tag headers).
    """
    if len(raw_data) < 28 or raw_data[:4] != b"PMAI":
        raise ValueError("Invalid ANLZ stream: missing PMAI header")

    pmai_hdr_len, total_len = struct.unpack_from(">II", raw_data, 4)
    offset = pmai_hdr_len
    tags: dict[str, list[bytes]] = {}

    limit = min(len(raw_data), total_len)
    while offset + 12 <= limit:
        tag_name_bytes, hdr_len, tag_len = struct.unpack_from(">4sII", raw_data, offset)
        if tag_len < 12 or offset + tag_len > len(raw_data):
            break
        try:
            tag_name = tag_name_bytes.decode("ascii")
        except UnicodeDecodeError:
            break
        tag_chunk = raw_data[offset : offset + tag_len]
        tags.setdefault(tag_name, []).append(tag_chunk)
        offset += tag_len

    return tags


def parse_ppth_tag(tag_bytes: bytes) -> str:
    """Extract relative audio path from an ANLZ PPTH tag chunk.

    Args:
        tag_bytes: Raw bytes of the PPTH tag chunk.

    Returns:
        Decoded relative path string (e.g. '/Contents/Artist/Album/Track.aiff').
    """
    if len(tag_bytes) < 16 or tag_bytes[:4] != b"PPTH":
        raise ValueError("Invalid PPTH tag")
    hdr_len, tag_len, path_len = struct.unpack_from(">III", tag_bytes, 4)
    path_bytes = tag_bytes[hdr_len : hdr_len + path_len]
    return path_bytes.decode("utf-16-be").rstrip("\x00")


def parse_bpm_tag(tag_bytes: bytes) -> float:
    """Extract Rekordbox analyzed tempo from a PQTZ beatgrid tag chunk.

    Args:
        tag_bytes: Raw bytes of the PQTZ tag chunk.

    Returns:
        Analyzed tempo in beats per minute (BPM).
    """
    if len(tag_bytes) < 24 or tag_bytes[:4] not in (b"PQTZ", b"PQT2"):
        raise ValueError("Invalid beatgrid tag")
    hdr_len, tag_len = struct.unpack_from(">II", tag_bytes, 4)
    count = struct.unpack_from(">I", tag_bytes, 20)[0]
    if count == 0 or len(tag_bytes) < hdr_len + 8:
        return 0.0

    # First beat grid entry: beat(2), tempo(2), time_ms(4)
    # tempo value is BPM * 100
    _, tempo_val, _ = struct.unpack_from(">HHI", tag_bytes, hdr_len)
    return round(tempo_val / 100.0, 2)


def _parse_pcp2_entries(tag_bytes: bytes) -> Tuple[list[HotCue], list[MemoryCue]]:
    """Parse PCP2 entries from a PCO2 tag chunk."""
    hdr_len, tag_len, cue_type, num_entries = struct.unpack_from(">IIII", tag_bytes, 4)
    offset = hdr_len
    hot_cues: list[HotCue] = []
    mem_cues: list[MemoryCue] = []

    fmt_hdr = ">4sII I B3sII B7sHH I"
    hdr_size = struct.calcsize(fmt_hdr)  # 44

    for _ in range(num_entries):
        if offset + hdr_size > len(tag_bytes):
            break
        (
            entry_magic,
            entry_hdr_len,
            entry_len,
            hot_cue_num,
            cue_kind,
            _,
            time_ms,
            loop_end_raw,
            color_id,
            _,
            _,
            _,
            len_comment,
        ) = struct.unpack_from(fmt_hdr, tag_bytes, offset)

        is_loop = (cue_kind == 2) or (loop_end_raw != 0xFFFFFFFF and loop_end_raw > time_ms)
        loop_end_ms = loop_end_raw if is_loop else None

        comment_start = offset + hdr_size
        comment = ""
        if len_comment > 0 and comment_start + len_comment <= offset + entry_len:
            comment_bytes = tag_bytes[comment_start : comment_start + len_comment]
            try:
                comment = comment_bytes.decode("utf-16-be").rstrip("\x00")
            except UnicodeDecodeError:
                comment = ""

        # Check for trailing RGB color tuple
        color_rgb: Optional[Tuple[int, int, int]] = None
        color_offset = comment_start + len_comment
        if color_offset + 4 <= offset + entry_len:
            color_code, r, g, b = struct.unpack_from(">BBBB", tag_bytes, color_offset)
            if (r, g, b) != (0, 0, 0) or color_code != 0:
                color_rgb = (r, g, b)

        if hot_cue_num in HOT_CUE_SLOTS:
            slot = HOT_CUE_SLOTS[hot_cue_num]
            hot_cues.append(
                HotCue(
                    slot=slot,
                    time_ms=time_ms,
                    comment=comment,
                    color_rgb=color_rgb,
                    is_loop=is_loop,
                    loop_end_ms=loop_end_ms,
                )
            )
        else:
            mem_cues.append(MemoryCue(time_ms=time_ms, comment=comment))

        offset += entry_len

    return hot_cues, mem_cues


def parse_cue_tags(tags: dict[str, list[bytes]]) -> Tuple[list[HotCue], list[MemoryCue]]:
    """Extract Hot Cues and Memory Cues from parsed ANLZ tags.

    Prefers modern extended PCO2 tags, falling back to legacy PCOB tags if PCO2 is absent.

    Args:
        tags: Dictionary of tag names to tag byte buffers produced by split_anlz_tags.

    Returns:
        Tuple of (hot_cues sorted by pad slot 'A'->'H', memory_cues sorted by time_ms).
    """
    all_hot_cues: dict[str, HotCue] = {}
    all_mem_cues: list[MemoryCue] = []

    target_tags = tags.get("PCO2") or tags.get("PCOB") or []
    for tag_bytes in target_tags:
        hot, mem = _parse_pcp2_entries(tag_bytes)
        for h in hot:
            all_hot_cues[h.slot] = h
        all_mem_cues.extend(mem)

    sorted_hot = sorted(all_hot_cues.values(), key=lambda h: h.slot)
    sorted_mem = sorted(all_mem_cues, key=lambda m: m.time_ms)
    return sorted_hot, sorted_mem


def build_track_cues(ext_data: bytes, dat_data: bytes, mtime: float = 0.0) -> TrackCues:
    """Construct TrackCues domain model from raw ANLZ0000.EXT and ANLZ0000.DAT byte buffers.

    Args:
        ext_data: Raw byte buffer of ANLZ0000.EXT (contains extended cues and waveforms).
        dat_data: Raw byte buffer of ANLZ0000.DAT (contains beatgrid and path).

    Returns:
        Fully populated TrackCues instance.
    """
    ext_tags = split_anlz_tags(ext_data)
    dat_tags = split_anlz_tags(dat_data)

    ppth_chunk = (ext_tags.get("PPTH") or dat_tags.get("PPTH") or [b""])[0]
    path = parse_ppth_tag(ppth_chunk) if ppth_chunk else ""
    filename = Path(path).name if path else ""

    pqtz_chunk = (dat_tags.get("PQTZ") or ext_tags.get("PQTZ") or dat_tags.get("PQT2") or [b""])[0]
    bpm = parse_bpm_tag(pqtz_chunk) if pqtz_chunk else 0.0

    hot_cues, memory_cues = parse_cue_tags(ext_tags)
    if not hot_cues and not memory_cues:
        hot_cues, memory_cues = parse_cue_tags(dat_tags)

    return TrackCues(
        path=path,
        filename=filename,
        bpm=bpm,
        hot_cues=hot_cues,
        memory_cues=memory_cues,
        mtime=mtime,
    )


def read_anlz_pair(track_dir: Path) -> Tuple[bytes, bytes]:
    """Read binary contents of ANLZ0000.EXT and ANLZ0000.DAT for a track directory."""
    ext_path = track_dir / "ANLZ0000.EXT"
    dat_path = track_dir / "ANLZ0000.DAT"

    ext_bytes = ext_path.read_bytes() if ext_path.exists() else b""
    dat_bytes = dat_path.read_bytes() if dat_path.exists() else b""
    return ext_bytes, dat_bytes


def scan_usb_anlz(pioneer_root: Path) -> List[TrackCues]:
    """Traverse USBANLZ directory hierarchy and extract cues for all analyzed tracks."""
    anlz_base = pioneer_root / "USBANLZ"
    if not anlz_base.exists():
        anlz_base = pioneer_root

    tracks: list[TrackCues] = []
    # Search for all directories containing ANLZ0000.DAT
    for dat_file in anlz_base.rglob("ANLZ0000.DAT"):
        track_dir = dat_file.parent
        ext_bytes, dat_bytes = read_anlz_pair(track_dir)
        if not dat_bytes:
            continue
        try:
            mtime = dat_file.stat().st_mtime
            track = build_track_cues(ext_bytes, dat_bytes, mtime=mtime)
            if track.filename:
                tracks.append(track)
        except Exception:
            continue

    return tracks
