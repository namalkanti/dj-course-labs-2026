"""Utilities for Marimo interactive DSP lab notebooks."""

import io
from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from dj_dsp.utils.dsp import normalize_peak


def fig_to_image(fig: plt.Figure, dpi: int = 100) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes for Marimo display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def parse_audio_upload(
    file_bytes: Union[bytes, bytearray, io.BytesIO],
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Parse uploaded audio file bytes into a 1D float32 NumPy array and sample rate.

    Handles mono and multi-channel audio (downmixes stereo to mono via channel average)
    and optional peak normalization to 0 dBFS.
    """
    if isinstance(file_bytes, (bytes, bytearray)):
        buffer = io.BytesIO(file_bytes)
    else:
        buffer = file_bytes

    audio, fs = sf.read(buffer, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    if normalize:
        audio = normalize_peak(audio, headroom_db=0.0)

    return audio, fs
