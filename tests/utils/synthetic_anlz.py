"""Synthetic binary builders for generating minimal Rekordbox ANLZ test fixtures.

Allows constructing minimal, valid ANLZ0000.EXT and ANLZ0000.DAT byte streams in memory
without checking large proprietary binary files into source control.
"""

import struct
from typing import List, Optional, Tuple


def build_tag(tag_type: str, content: bytes, header_len: int = 12) -> bytes:
    """Build a standard Rekordbox tagged chunk (magic, len_header, len_tag, content)."""
    tag_magic = tag_type.encode("ascii")
    total_len = len(content) + 12
    return struct.pack(">4sII", tag_magic, header_len, total_len) + content


def build_ppth_tag(path_str: str) -> bytes:
    """Build a PPTH (file path) tag encoded in UTF-16-BE."""
    encoded_path = path_str.encode("utf-16-be") + b"\x00\x00"
    path_len = len(encoded_path)
    # PPTH header is 16 bytes: magic(4), len_hdr(4), len_tag(4), path_len(4)
    content = struct.pack(">I", path_len) + encoded_path
    return struct.pack(">4sIII", b"PPTH", 16, len(content) + 12, path_len) + encoded_path


def build_pqtz_tag(bpm: float) -> bytes:
    """Build a minimal PQTZ (beatgrid) tag with a constant tempo grid."""
    # PQTZ header is 24 bytes: magic(4), len_hdr(4), len_tag(4), u1(4), u2(4), count(4)
    # Entry is 8 bytes: beat(2), tempo(2), time_ms(4)
    tempo_val = int(round(bpm * 100))
    entry = struct.pack(">HHI", 1, tempo_val, 0)
    tag_hdr = struct.pack(">4sII III", b"PQTZ", 24, 24 + len(entry), 0, 0x80000, 1)
    return tag_hdr + entry


def build_pcp2_entry(
    hot_cue_num: int,
    time_ms: int,
    comment: str = "",
    color_rgb: Optional[Tuple[int, int, int]] = None,
    is_loop: bool = False,
    loop_end_ms: int = 0xFFFFFFFF,
) -> bytes:
    """Build a single PCP2 cue point entry for PCO2 tags."""
    cue_type = 2 if is_loop else 1
    encoded_comment = comment.encode("utf-16-be") if comment else b""
    len_comment = len(encoded_comment)

    # Base entry header before comment is 44 bytes
    fmt_hdr = ">4sII I B3sII B7sHH I"
    if len_comment == 0 and color_rgb is None:
        len_entry = 44
        return struct.pack(
            fmt_hdr,
            b"PCP2",
            16,
            len_entry,
            hot_cue_num,
            cue_type,
            b"\x00\x00\x00",
            time_ms,
            loop_end_ms,
            0,  # color_id
            b"\x00" * 7,
            0,  # loop_enum
            0,  # loop_denom
            0,  # len_comment
        )

    # With comment and optional color
    color_bytes = b""
    if color_rgb is not None:
        color_bytes = struct.pack(">BBBB", 0, color_rgb[0], color_rgb[1], color_rgb[2])

    base_len = 44 + len_comment + len(color_bytes)
    # Pad to multiple of 4 or 48 minimum if comment/color present
    len_entry = max(48 + len_comment, base_len)
    pad_len = len_entry - (44 + len_comment + len(color_bytes))

    entry_hdr = struct.pack(
        fmt_hdr,
        b"PCP2",
        16,
        len_entry,
        hot_cue_num,
        cue_type,
        b"\x00\x00\x00",
        time_ms,
        loop_end_ms,
        0,
        b"\x00" * 7,
        0,
        0,
        len_comment,
    )
    return entry_hdr + encoded_comment + color_bytes + (b"\x00" * pad_len)


def build_pco2_tag(cue_obj_type: int, entries: List[bytes]) -> bytes:
    """Build a PCO2 tag containing PCP2 entries.

    cue_obj_type: 0 for memory cues, 1 for hot cues.
    """
    total_entries_len = sum(len(e) for e in entries)
    total_tag_len = 20 + total_entries_len
    hdr = struct.pack(">4sII IHH", b"PCO2", 20, total_tag_len, cue_obj_type, len(entries), 0)
    return hdr + b"".join(entries)


def build_anlz_file(tags: List[bytes]) -> bytes:
    """Wrap tag chunks in a PMAI file header."""
    total_len = 28 + sum(len(t) for t in tags)
    pmai_hdr = struct.pack(">4sII IIII", b"PMAI", 28, total_len, 1, 1, 1, 0)
    return pmai_hdr + b"".join(tags)
