"""Unit tests for yourshot.audio_catalog."""

import unittest
from pathlib import Path
import numpy as np

from yourshot.audio_catalog import (
    get_default_fixtures_dir,
    load_audio_catalog,
)


class TestAudioCatalog(unittest.TestCase):

    def test_default_fixtures_dir_exists(self):
        fixtures_dir = get_default_fixtures_dir()
        self.assertTrue(fixtures_dir.exists(), f"Fixtures dir not found at {fixtures_dir}")
        self.assertTrue((fixtures_dir / "README.md").exists())

    def test_load_audio_catalog_structure(self):
        catalog, fs = load_audio_catalog(fs=44100)
        self.assertEqual(fs, 44100)

        # Ensure all expected keys are present
        expected_keys = [
            "Synthetic: Drum Groove (128 BPM)",
            "Synthetic: Isolated Transients (Castanet / Rimshot Pre-Echo Test)",
            "Synthetic: Exponential Chirp (20 Hz - 22 kHz)",
            "Synthetic: White Noise (Wideband)",
            "Real Audio: Persistence - Dabmasterars (Lossless FLAC)",
            "Real Audio: High Technologic Beat Explosion - Loyalty Freak Music (320k MP3)",
        ]
        for key in expected_keys:
            self.assertIn(key, catalog, f"Missing key in catalog: {key}")
            arr = catalog[key]
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.ndim, 1, f"Signal {key} should be mono 1D")
            self.assertFalse(arr.flags.writeable, f"Signal {key} should be read-only")
            self.assertTrue(len(arr) > 0, f"Signal {key} is empty")
            self.assertTrue(np.all(np.isfinite(arr)), f"Signal {key} contains non-finite values")


if __name__ == "__main__":
    unittest.main()
