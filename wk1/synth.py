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
    t = time_samples_for_frequency(duration_sec, fs)
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
    total_samples = int(duration_sec * fs)
    output = np.zeros(total_samples, dtype=np.float32)

    # Physical parameters for drum hits
    kick_duration_sec = 0.25
    hat_duration_sec = 0.06
    hat_tau_sec = 0.025
    hat_highpass_hz = 7500.0

    # 1. Synthesize a single kick hit (~0.25 s)
    t_k = time_samples_for_frequency(kick_duration_sec, fs)
    kick_len = len(t_k)

    # Frequency drops from 150 Hz -> 45 Hz with tau = 20 ms
    f_start, f_end, tau_f = 150.0, 45.0, 0.02
    tau_a = kick_amplitude_decay_time_constant(f_end, bpm)
    phase_k = phase_integral(t_k, f_start, f_end, tau_f)
    amp_k = exponential_decay_envelope(kick_duration_sec, tau_a, fs)
    kick_hit = (np.sin(phase_k) * amp_k).astype(np.float32)

    # 2. Synthesize a single hi-hat hit (~0.06 s)
    t_h = time_samples_for_frequency(hat_duration_sec, fs)
    hat_len = len(t_h)
    rng = np.random.default_rng(seed=42)
    raw_noise = rng.normal(0.0, 0.5, hat_len)

    # 2nd-order Butterworth high-pass to isolate cymbal sizzle
    sos_hat = signal.butter(2, hat_highpass_hz, btype="highpass", fs=fs, output="sos")
    filtered_noise = signal.sosfilt(sos_hat, raw_noise)
    amp_h = exponential_decay_envelope(hat_duration_sec, hat_tau_sec, fs)
    hat_hit = (filtered_noise * amp_h).astype(np.float32)

    # 3. Schedule hits across duration
    beat_sec = 60.0 / bpm
    total_beats = int(duration_sec / beat_sec)

    for b in range(total_beats):
        # Kick on downbeat: b * beat_sec
        idx_k = int(b * beat_sec * fs)
        end_k = min(idx_k + kick_len, total_samples)
        output[idx_k:end_k] += kick_hit[: end_k - idx_k]

        # Hi-hat on offbeat: (b + 0.5) * beat_sec
        idx_h = int((b + 0.5) * beat_sec * fs)
        if idx_h < total_samples:
            end_h = min(idx_h + hat_len, total_samples)
            output[idx_h:end_h] += 0.6 * hat_hit[: end_h - idx_h]

    # Normalize peak to 0.7 (-3 dBFS)
    peak = np.max(np.abs(output))
    if peak > 0:
        output = (output / peak * 0.7).astype(np.float32)

    return output


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
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        return audio.copy().astype(np.float32)

    # High-order Butterworth designed as second-order sections for numerical stability
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    filtered = signal.sosfilt(sos, audio)
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)

def time_samples_for_frequency(duration_sec: float, fs: int) -> np.ndarray:
    """Return sampled time points [0, duration_sec) spaced at the sampling frequency fs.

    Examples:
        >>> t = time_samples_for_frequency(0.01, 1000)
        >>> len(t)
        10
        >>> float(t[0]), float(t[1]), float(t[-1])
        (0.0, 0.001, 0.009)
    """
    num_samples = int(duration_sec * fs)
    return np.arange(num_samples) / fs


def phase_integral(
    t: np.ndarray,
    f_start: float,
    f_end: float,
    tau: float,
) -> np.ndarray:
    """Compute continuous phase in radians for an exponential frequency drop.

    f(t) = f_end + (f_start - f_end) * exp(-t / tau)
    phi(t) = 2*pi * integral_0^t f(u) du
           = 2*pi * [f_end * t + (f_start - f_end) * tau * (1 - exp(-t / tau))]

    Examples:
        >>> # At t=0, phase must start at exactly 0.0
        >>> round(float(phase_integral(np.array([0.0]), 150.0, 45.0, 0.02)[0]), 6)
        0.0
        >>> # Phase increases monotonically
        >>> t = np.array([0.0, 0.01, 0.02])
        >>> phi = phase_integral(t, 150.0, 45.0, 0.02)
        >>> bool(np.all(np.diff(phi) > 0))
        True
    """
    integral = f_end * t + (f_start - f_end) * tau * (1.0 - np.exp(-t / tau))
    return 2.0 * np.pi * integral


def kick_amplitude_decay_time_constant(
    f_end_hz: float = 45.0,
    bpm: float = 128.0,
    min_cycles: float = 4.0,
) -> float:
    """Compute kick amplitude decay time constant tau_a.

    Balances two physical constraints:
    1. Pitch perception: human hearing needs ~4 cycles of the resting sub
       frequency f_end to perceive a defined bass pitch (tau_sub = min_cycles / f_end).
    2. Tempo clearance: the tail must drop >= 20 dB (~3 tau) before the offbeat
       hi-hat strikes at half a beat (tau_max = (60 / bpm) / 6).

    Examples:
        >>> # At 128 BPM and 45 Hz sub, bounded by tempo clearance (~78 ms)
        >>> round(kick_amplitude_decay_time_constant(45.0, 128.0), 3)
        0.078
        >>> # At slow 60 BPM, pitch perception dominates (~89 ms)
        >>> round(kick_amplitude_decay_time_constant(45.0, 60.0), 3)
        0.089
    """
    tau_sub = min_cycles / f_end_hz
    tau_max_tempo = (60.0 / bpm) / 6.0
    return float(min(tau_sub, tau_max_tempo))


def exponential_decay_envelope(
    duration_sec: float,
    tau_sec: float,
    fs: int,
) -> np.ndarray:
    """Generate an exponential amplitude decay envelope e^(-t / tau).

    Examples:
        >>> env = exponential_decay_envelope(0.1, 0.05, 1000)
        >>> len(env)
        100
        >>> # Peak starts at 1.0 (0 dBFS)
        >>> round(float(env[0]), 3)
        1.0
        >>> # At t = tau (index 50), amplitude drops to 1/e (~0.368)
        >>> round(float(env[50]), 3)
        0.368
    """
    t = time_samples_for_frequency(duration_sec, fs)
    return np.exp(-t / tau_sec).astype(np.float32)


def generate_transient_clicks(
    duration_sec: float = 2.0,
    click_times_sec: Tuple[float, ...] = (0.4, 0.9, 1.4),
    fs: int = 44100,
) -> np.ndarray:
    """Synthesize isolated dry transient clicks against silence (castanet/rimshot style).

    Designed specifically as an MDCT codec pre-echo torture test. Each click consists
    of a tight 12ms wideband burst (resonant metallic ping + filtered noise transient)
    separated by pure silence.

    Args:
        duration_sec: Total length in seconds.
        click_times_sec: Timestamps in seconds where transients occur.
        fs: Sampling rate in Hz.

    Returns:
        1D NumPy float32 array normalized to [-1.0, 1.0].
    """
    total_samples = int(duration_sec * fs)
    signal_out = np.zeros(total_samples, dtype=np.float32)

    click_duration = 0.012  # 12 ms
    click_len = int(click_duration * fs)
    t_click = np.linspace(0.0, click_duration, click_len, endpoint=False, dtype=np.float32)

    rng = np.random.default_rng(42)
    noise_burst = rng.normal(0.0, 0.4, click_len).astype(np.float32)
    envelope = np.exp(-t_click / 0.002).astype(np.float32)  # 2 ms tau decay
    ping = np.sin(2.0 * np.pi * 1400.0 * t_click, dtype=np.float32) * np.exp(-t_click / 0.003).astype(np.float32)

    single_click = (noise_burst * envelope + ping * 0.6).astype(np.float32)

    for click_time in click_times_sec:
        start_idx = int(click_time * fs)
        end_idx = min(start_idx + click_len, total_samples)
        valid_len = end_idx - start_idx
        if valid_len > 0:
            signal_out[start_idx:end_idx] += single_click[:valid_len]

    peak = np.max(np.abs(signal_out))
    if peak > 0:
        signal_out = (signal_out / peak * 0.9).astype(np.float32)

    return signal_out



