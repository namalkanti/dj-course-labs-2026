"""Synthetic ground-truth audio signals and codec cutoff filter simulations.

Pure NumPy / SciPy implementation for validating transcode detection
and evaluating lossy compression artifacts.
"""

from typing import Tuple
import numpy as np
import scipy.signal as signal


def generate_white_noise(duration_sec: float = 3.0, fs: int = 44100) -> np.ndarray:
    """Generate uniform or Gaussian wideband white noise scaled to -6 dBFS.

    Args:
        duration_sec: Signal duration in seconds.
        fs: Sampling rate in Hz.

    Returns:
        1D NumPy float array with values normalized in range [-1.0, 1.0].
    """
    N = int(duration_sec * fs)
    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=0.2, size=N)

    # Normalize peak to 0.5 (-6 dBFS)
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak * 0.5
    return noise.astype(np.float32)

def generate_chirp(
    duration_sec: float = 3.0,
    fs: int = 44100,
    f_start: float = 20.0,
    f_end: float = 22050.0,
) -> np.ndarray:
    """Generate an exponential frequency sweep (chirp) from f_start to f_end.

    Exponential chirps spend equal time per octave, making them ideal
    for audio spectral roll-off visualization.

    Args:
        duration_sec: Signal duration in seconds.
        fs: Sampling rate in Hz.
        f_start: Starting frequency in Hz.
        f_end: Ending frequency in Hz (typically Nyquist = fs / 2).

    Returns:
        1D NumPy float array normalized in range [-1.0, 1.0].
    """
    N = int(duration_sec * fs)
    t = np.linspace(0.0, duration_sec, N, endpoint=False)
    sweep = signal.chirp(
            t,
            f0=f_start,
            t1=duration_sec,
            f1=f_end,
            method="logarithmic",
    )
    sweep *= 0.5
    return sweep.astype(np.float32)


def generate_drum_loop(
    duration_sec: float = 3.0,
    bpm: float = 128.0,
    fs: int = 44100,
) -> np.ndarray:
    """Synthesize a minimal electronic drum loop with kicks and hi-hats.

    - Kick: rapid exponential pitch drop (e.g. 150 Hz -> 45 Hz) with decay envelope.
    - Hi-hat: highpass-filtered noise burst with sharp exponential decay.

    Provides musical transients and high-frequency cymbal air to audition
    muffling and pre-echo artifacts.

    Args:
        duration_sec: Signal duration in seconds.
        bpm: Beats per minute.
        fs: Sampling rate in Hz.

    Returns:
        1D NumPy float array normalized in range [-1.0, 1.0].
    """
    # TODO: Synthesize kick drum and hi-hat patterns at the given BPM
    raise NotImplementedError("Implement generate_drum_loop")


def apply_lossy_cutoff(
    audio: np.ndarray,
    cutoff_hz: float,
    fs: int = 44100,
    order: int = 10,
) -> np.ndarray:
    """Apply a steep digital lowpass filter simulating lossy encoder cutoffs.

    - 16,000 Hz cutoff simulates ~128 kbps MP3.
    - 18,500 Hz cutoff simulates ~192 kbps MP3.
    - >= 22,050 Hz returns audio unmodified (lossless).

    Uses second-order sections (SOS) for numerical stability with high filter orders.

    Args:
        audio: 1D input audio array.
        cutoff_hz: Cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order (higher = steeper brick-wall transition).

    Returns:
        Filtered 1D NumPy float array.
    """
    # TODO: If cutoff_hz >= fs / 2, return audio.copy().
    # Otherwise, design a Butterworth lowpass filter with signal.butter(output='sos')
    # and filter using signal.sosfilt().
    raise NotImplementedError("Implement apply_lossy_cutoff")
