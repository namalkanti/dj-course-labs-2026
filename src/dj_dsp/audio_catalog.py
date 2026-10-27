"""Audio catalog providing synthetic signals and real reference audio fixtures.

Designed for reproducible audio quality analysis, transcode detection,
and DSP visualization across the DJ curriculum.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import soundfile as sf

from dj_dsp.synth import (
    generate_chirp,
    generate_drum_loop,
    generate_linear_chirp,
    generate_transient_clicks,
    generate_white_noise,
)


@dataclass(frozen=True)
class AudioEntry:
    """Metadata and read-only waveform for an audio catalog item."""
    label: str
    audio: np.ndarray
    fs: int
    is_synthetic: bool
    description: str


def get_default_fixtures_dir() -> Path:
    """Resolve default fixtures path at labs/audio_catalog."""
    # This file is at labs/src/dj_dsp/audio_catalog.py.
    # parents: [0]=dj_dsp, [1]=src, [2]=labs
    labs_root = Path(__file__).resolve().parents[2]
    return labs_root / "audio_catalog"


def load_audio_catalog(
    fs: int = 44100,
    fixtures_dir: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Load dictionary of clean audio signals (synthetic + reference fixtures).

    All audio arrays are returned as immutable 1D float32 arrays normalized
    to [-1.0, 1.0].

    Args:
        fs: Target sampling rate in Hz (default: 44100).
        fixtures_dir: Optional path to directory containing real audio files.
                      Defaults to labs/audio_catalog.

    Returns:
        Tuple of (catalog_dict, fs).
    """
    catalog: Dict[str, np.ndarray] = {}

    # Synthetic fixtures (deterministic, generated on demand)
    synthetic_defs = [
        (
            "Synthetic: Drum Groove (128 BPM)",
            lambda: generate_drum_loop(3.0, 128.0, fs),
        ),
        (
            "Synthetic: Isolated Transients (Castanet / Rimshot Pre-Echo Test)",
            lambda: generate_transient_clicks(2.0, (0.4, 0.9, 1.4), fs),
        ),
        (
            "Synthetic: Exponential Chirp (20 Hz - 22 kHz)",
            lambda: generate_chirp(3.0, fs),
        ),
        (
            "Synthetic: Linear High-Register Chirp (10 kHz - 20 kHz)",
            lambda: generate_linear_chirp(4.0, fs, 10000.0, 20000.0),
        ),
        (
            "Synthetic: White Noise (Wideband)",
            lambda: generate_white_noise(3.0, fs),
        ),
    ]

    for label, generator in synthetic_defs:
        arr = generator()
        arr.setflags(write=False)
        catalog[label] = arr

    # Real reference audio fixtures
    dir_path = fixtures_dir if fixtures_dir is not None else get_default_fixtures_dir()
    real_defs = [
        (
            "Real Audio: Persistence - Dabmasterars (Lossless FLAC)",
            dir_path / "persistence_by_dabmasterars_snippet.flac",
        ),
        (
            "Real Audio: High Technologic Beat Explosion - Loyalty Freak Music (320k MP3)",
            dir_path / "high_technologic_beat_explosion_by_loyalty_freak_music_snippet.mp3",
        ),
    ]

    for label, filepath in real_defs:
        if filepath.exists():
            data, file_fs = sf.read(str(filepath), dtype="float32")
            mono = data.mean(axis=1) if data.ndim > 1 else data
            mono.setflags(write=False)
            catalog[label] = mono

    return catalog, fs
