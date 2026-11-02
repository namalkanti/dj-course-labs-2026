"""FFmpeg subprocess wrapper for audio transcoding operations.

Handles temporary file I/O, subprocess execution against the system FFmpeg binary,
and round-trip decoding back to standardized float32 NumPy arrays.
"""

import shutil
import subprocess
import tempfile
import numpy as np
import soundfile as sf


def find_ffmpeg_binary() -> str | None:
    """Locate the ffmpeg binary on the system PATH."""
    return shutil.which("ffmpeg")


def transcode_audio_subprocess(
    audio: np.ndarray,
    fs: int,
    encoder_args: list[str],
    intermediate_suffix: str = ".mp3",
) -> np.ndarray:
    """Transcode a float32 audio array through an external ffmpeg process.

    Parameters:
        audio: 1D float32 audio array.
        fs: Sampling rate in Hz.
        encoder_args: Arguments passed to the ffmpeg encoder (e.g. ['-codec:a', 'libmp3lame']).
        intermediate_suffix: File extension for intermediate lossy file (default '.mp3').

    Returns:
        Decoded 1D float32 audio array matched to the exact original length.
    """
    ffmpeg_bin = find_ffmpeg_binary()
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg binary not found on system PATH")

    with tempfile.NamedTemporaryFile(suffix=".wav") as f_in, \
         tempfile.NamedTemporaryFile(suffix=intermediate_suffix) as f_lossy, \
         tempfile.NamedTemporaryFile(suffix=".wav") as f_out:

        # Write input WAV (convert to 2D column if mono)
        stereo_audio = np.column_stack([audio, audio]) if audio.ndim == 1 else audio
        sf.write(f_in.name, stereo_audio, fs, subtype="PCM_16")

        # Encode pass
        cmd_enc = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", f_in.name,
            *encoder_args,
            f_lossy.name,
        ]
        subprocess.run(cmd_enc, capture_output=True, check=True)

        # Decode pass back to PCM
        cmd_dec = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", f_lossy.name,
            "-ar", str(fs),
            "-ac", "1",
            f_out.name,
        ]
        subprocess.run(cmd_dec, capture_output=True, check=True)

        decoded, _ = sf.read(f_out.name, dtype="float32")

        # Match length to original audio
        orig_len = len(audio)
        if len(decoded) > orig_len:
            decoded = decoded[:orig_len]
        elif len(decoded) < orig_len:
            decoded = np.pad(decoded, (0, orig_len - len(decoded)))

        return np.clip(decoded, -1.0, 1.0).astype(np.float32)
