"""Unit tests for Marimo lab UI and audio ingestion utilities."""

import io
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from dj_dsp.utils.lab import fig_to_image, parse_audio_upload


def test_fig_to_image():
    """Verify Matplotlib figure serializes to non-empty PNG byte stream."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    img_bytes = fig_to_image(fig)

    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0
    # PNG signature: \x89PNG\r\n\x1a\n
    assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_parse_audio_upload_stereo_to_mono():
    """Verify stereo audio bytes are parsed into 1D mono float32 array with downmixing and normalization."""
    fs = 44100
    # Create 1 second of stereo audio
    left = np.full(fs, 0.5, dtype=np.float32)
    right = np.full(fs, 0.25, dtype=np.float32)
    stereo = np.column_stack([left, right])

    buf = io.BytesIO()
    sf.write(buf, stereo, fs, format="WAV")
    raw_bytes = buf.getvalue()

    # Raw downmixed (no normalization): mean(0.5, 0.25) == 0.375
    raw_audio, parsed_fs = parse_audio_upload(raw_bytes, normalize=False)
    assert parsed_fs == fs
    assert raw_audio.ndim == 1
    assert len(raw_audio) == fs
    assert np.isclose(raw_audio[0], 0.375, atol=1e-3)

    # Normalized: peak scales to 1.0
    norm_audio, _ = parse_audio_upload(raw_bytes, normalize=True)
    assert np.isclose(np.max(np.abs(norm_audio)), 1.0)
