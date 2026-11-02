"""LAME MP3 audio encoding and codec simulation.

Encodes audio arrays through the system LAME MP3 encoder (via FFmpeg),
falling back to pure mathematical brickwall/Butterworth filtering when FFmpeg
is unavailable.
"""

import numpy as np

from dj_dsp.ffmpeg import find_ffmpeg_binary, transcode_audio_subprocess
from dj_dsp.synth import DEFAULT_FS, apply_lossy_cutoff

# Default codec cutoff frequency thresholds in Hz per MP3 bitrate tier
LAME_FALLBACK_CUTOFF_MAP = {
    64: 11000.0,
    96: 14000.0,
    128: 16000.0,
    192: 18500.0,
    320: 20000.0,
}


def encode_mp3_lame(
    audio: np.ndarray,
    fs: int = DEFAULT_FS,
    bitrate_kbps: int = 128,
) -> np.ndarray:
    """Encode audio through genuine LAME MP3 encoder via FFmpeg with pure lowpass fallback.

    Parameters:
        audio: 1D float32 audio signal.
        fs: Sampling rate in Hz.
        bitrate_kbps: Target MP3 CBR bitrate in kbps (e.g. 64, 128, 192, 320).

    Returns:
        Transcoded audio array. If FFmpeg is unavailable, returns mathematically
        filtered audio simulating the encoder's lowpass brickwall cutoff.
    """
    if find_ffmpeg_binary() is None:
        fc = LAME_FALLBACK_CUTOFF_MAP.get(bitrate_kbps, 16000.0)
        return apply_lossy_cutoff(audio, cutoff_hz=fc, fs=fs)

    encoder_args = ["-codec:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k"]
    return transcode_audio_subprocess(
        audio, fs=fs, encoder_args=encoder_args, intermediate_suffix=".mp3"
    )
