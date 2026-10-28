"""Unit tests for dj_dsp.audio_catalog.

Uses self-documenting assertion helpers grouped at the bottom to verify fixture
directory presence, track existence, non-emptiness, finiteness, and immutability.
"""

from pathlib import Path
import numpy as np

from dj_dsp.audio_catalog import (
    get_default_fixtures_dir,
    load_audio_catalog,
)


# ---------------------------------------------------------------------------
# Tests (Specifications)
# ---------------------------------------------------------------------------


def test_default_fixtures_dir_exists():
    """Verify that the default audio fixtures directory exists and contains a README.

    Invariants:
    - Fixtures directory path exists as a directory on disk.
    - README.md exists within the fixtures directory as standard documentation.
    """
    fixtures_dir = get_default_fixtures_dir()
    assert fixtures_dir.exists(), f"Fixtures dir not found at {fixtures_dir}"
    assert_file_exists(fixtures_dir / "README.md")


def test_load_audio_catalog_structure():
    """Verify that the audio catalog loads with expected keys, sample rate, and valid signals.

    Invariants:
    - Returned sampling rate matches requested sample rate (44100 Hz).
    - Catalog contains both synthetic benchmark signals and real audio fixtures.
    - Each catalog entry is a non-empty, 1D, finite, read-only NumPy array.
    """
    expected_fs = 44100
    catalog, fs = load_audio_catalog(fs=expected_fs)
    assert fs == expected_fs

    expected_keys = [
        "Synthetic: Drum Groove (128 BPM)",
        "Synthetic: Isolated Transients (Castanet / Rimshot Pre-Echo Test)",
        "Synthetic: Exponential Chirp (20 Hz - 22 kHz)",
        "Synthetic: White Noise (Wideband)",
        "Real Audio: Persistence - Dabmasterars (Lossless FLAC)",
        "Real Audio: High Technologic Beat Explosion - Loyalty Freak Music (320k MP3)",
    ]

    for key in expected_keys:
        assert key in catalog, f"Missing key in catalog: {key}"
        assert_valid_mono_audio_signal(catalog[key], track_name=key)


# ---------------------------------------------------------------------------
# Domain-Specific Assertion Helpers
# ---------------------------------------------------------------------------


def assert_file_exists(path: Path) -> None:
    """Assert filesystem path exists and is a regular file."""
    assert path.exists(), f"Path not found: {path}"
    assert path.is_file(), f"Expected file at: {path}"


def assert_valid_mono_audio_signal(audio: np.ndarray, track_name: str) -> None:
    """Assert audio array is a 1D, non-empty, finite, read-only NumPy array."""
    assert isinstance(audio, np.ndarray), f"Track '{track_name}' expected np.ndarray, got {type(audio)}"
    assert audio.ndim == 1, f"Track '{track_name}' must be mono 1D, got ndim={audio.ndim}"
    assert len(audio) > 0, f"Track '{track_name}' is empty (0 samples)"
    assert np.all(np.isfinite(audio)), f"Track '{track_name}' contains NaN or Inf values"
    assert not audio.flags.writeable, f"Track '{track_name}' must be immutable (read-only)"
