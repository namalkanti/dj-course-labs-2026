"""Synthetic ground-truth audio signals and codec cutoff filter simulations.

Pure NumPy / SciPy implementation for validating transcode detection
and evaluating lossy compression artifacts.
"""

from typing import Tuple
import numpy as np
from scipy import signal

DEFAULT_FS = 44100
DEFAULT_HEADROOM_RATIO = 0.85
MAX_PEAK_AMPLITUDE = 0.95
EPSILON_SLOPE = 1e-6


def generate_white_noise(duration_sec: float = 3.0, fs: int = DEFAULT_FS) -> np.ndarray:
    """Generate wideband Gaussian white noise scaled to -6 dBFS."""
    num_samples = int(duration_sec * fs)
    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=0.2, size=num_samples)
    return _normalize_peak(noise, target_peak=0.5)


def generate_chirp(
    duration_sec: float = 3.0,
    fs: int = DEFAULT_FS,
    f_start: float = 20.0,
    f_end: float = 22050.0,
) -> np.ndarray:
    """Generate an exponential frequency sweep (chirp) from f_start to f_end."""
    t = _time_samples_for_frequency(duration_sec, fs)
    sweep = signal.chirp(t, f0=f_start, t1=duration_sec, f1=f_end, method="logarithmic")
    return (sweep * 0.5).astype(np.float32)


def generate_linear_chirp(
    duration_sec: float = 4.0,
    fs: int = DEFAULT_FS,
    f_start: float = 10000.0,
    f_end: float = 20000.0,
) -> np.ndarray:
    """Generate a linear frequency sweep across the top register (10 kHz to 20 kHz)."""
    t = _time_samples_for_frequency(duration_sec, fs)
    sweep = signal.chirp(t, f0=f_start, t1=duration_sec, f1=f_end, method="linear")
    return (sweep * 0.5).astype(np.float32)


def generate_transient_clicks(
    duration_sec: float = 2.0,
    click_times_sec: Tuple[float, ...] = (0.4, 0.9, 1.4),
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize isolated dry transient clicks against silence (castanet/rimshot style)."""
    total_samples = int(duration_sec * fs)
    signal_out = np.zeros(total_samples, dtype=np.float32)

    click_duration = 0.012  # 12 ms
    click_len = int(click_duration * fs)
    t_click = np.linspace(0.0, click_duration, click_len, endpoint=False, dtype=np.float32)

    decay_env = np.exp(-t_click / 0.002).astype(np.float32)
    ping = np.sin(2.0 * np.pi * 3200.0 * t_click, dtype=np.float32) * decay_env

    rng = np.random.default_rng(42)
    noise_burst = rng.normal(0.0, 0.5, click_len).astype(np.float32) * decay_env
    click_waveform = (0.6 * ping + 0.4 * noise_burst).astype(np.float32)

    for click_t in click_times_sec:
        start_idx = int(click_t * fs)
        _overlay_drum_hit(signal_out, click_waveform, start_idx, amplitude=1.0)

    return _normalize_peak(signal_out, target_peak=0.9)


def generate_drum_loop(
    duration_sec: float = 3.0,
    bpm: float = 128.0,
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize a minimal electronic drum loop with kicks and offbeat hi-hats."""
    total_samples = int(duration_sec * fs)
    output = np.zeros(total_samples, dtype=np.float32)

    kick_hit = _synth_kick_drum(fs)
    hat_hit = _synth_hihat(fs)

    beat_sec = 60.0 / bpm
    total_beats = int(duration_sec / beat_sec)

    for b in range(total_beats):
        downbeat_pos = int(b * beat_sec * fs)
        _overlay_drum_hit(output, kick_hit, downbeat_pos, amplitude=1.0)

        offbeat_pos = int((b + 0.5) * beat_sec * fs)
        _overlay_drum_hit(output, hat_hit, offbeat_pos, amplitude=0.6)

    return _normalize_peak(output, target_peak=0.7)


def generate_synthetic_beat(
    bpm: float = 128.0,
    duration_sec: float = 8.0,
    meter: str = "4/4",
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize a clean four-on-the-floor dance beat with kicks, snares, and hi-hats."""
    total_samples = int(duration_sec * fs)
    buffer = np.zeros(total_samples, dtype=np.float32)

    beat_interval_sec = 60.0 / bpm
    beat_samples = int(beat_interval_sec * fs)

    kick = _synth_kick_drum(fs)
    snare = _synth_snare_drum(fs)
    hihat = _synth_hihat(fs)

    num_beats = int(duration_sec / beat_interval_sec)
    for beat_idx in range(num_beats):
        beat_pos = int(beat_idx * beat_samples)
        _overlay_drum_hit(buffer, kick, beat_pos, amplitude=1.0)

        if beat_idx % 2 == 1:
            _overlay_drum_hit(buffer, snare, beat_pos, amplitude=0.7)

        offbeat_pos = beat_pos + beat_samples // 2
        _overlay_drum_hit(buffer, hihat, offbeat_pos, amplitude=0.4)

    return _normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


def generate_waltz_beat(
    bpm: float = 120.0,
    duration_sec: float = 8.0,
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize a 3/4 waltz metric pattern (kick on 1, hats on 2 and 3)."""
    total_samples = int(duration_sec * fs)
    buffer = np.zeros(total_samples, dtype=np.float32)

    beat_interval_sec = 60.0 / bpm
    beat_samples = int(beat_interval_sec * fs)

    kick = _synth_kick_drum(fs)
    hihat = _synth_hihat(fs)

    num_beats = int(duration_sec / beat_interval_sec)
    for beat_idx in range(num_beats):
        beat_pos = int(beat_idx * beat_samples)
        if beat_idx % 3 == 0:
            _overlay_drum_hit(buffer, kick, beat_pos, amplitude=1.0)
        else:
            _overlay_drum_hit(buffer, hihat, beat_pos, amplitude=0.6)

    return _normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


def generate_syncopated_breakbeat(
    bpm: float = 140.0,
    duration_sec: float = 8.0,
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize a syncopated funk breakbeat with offbeat anticipations.

    Syncopation Definition:
        Syncopation is the deliberate placement of rhythmic accents where listeners
        expect weak beats or silence (the off-beats). In music information retrieval,
        syncopation challenges rigid tempo grids because strong transients land on
        fractional beat subdivisions rather than regular integer downbeats.

    Metric Layout (16-step grid of sixteenth notes per 4/4 bar):
        - Subdivisions: 1 e & a | 2 e & a | 3 e & a | 4 e & a
        - Steps 0, 4, 8, 12: On-beat quarter-note downbeats.
        - Steps 4, 12: Standard backbeat snare hits (beats 2 and 4).
        - Step 0: Downbeat kick (beat 1).
        - Step 7: Off-beat syncopated kick on the "a" of beat 2 (anticipates beat 3).
        - Step 10: Off-beat syncopated kick on the "&" of beat 3.
        - Even steps (0, 2, 4, ...): Steady eighth-note closed hi-hat pulse.
    """
    total_samples = int(duration_sec * fs)
    buffer = np.zeros(total_samples, dtype=np.float32)

    beat_interval_sec = 60.0 / bpm
    subdivision_16th_sec = beat_interval_sec / 4.0
    sub_samples = int(subdivision_16th_sec * fs)

    kick = _synth_kick_drum(fs)
    snare = _synth_snare_drum(fs)
    hihat = _synth_hihat(fs)

    total_16ths = int(duration_sec / subdivision_16th_sec)
    for step in range(total_16ths):
        step_pos = int(step * sub_samples)
        bar_step = step % 16

        # Continuous 8th-note hi-hat pulse
        if bar_step % 2 == 0:
            _overlay_drum_hit(buffer, hihat, step_pos, amplitude=0.35)

        # Kicks: downbeat (0), anticipated offbeat (7), and mid-bar syncopation (10)
        if bar_step in (0, 7, 10):
            _overlay_drum_hit(buffer, kick, step_pos, amplitude=0.95)

        # Snares: standard backbeat on 2 and 4
        if bar_step in (4, 12):
            _overlay_drum_hit(buffer, snare, step_pos, amplitude=0.8)

    return _normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


def generate_tempo_drift_beat(
    start_bpm: float = 120.0,
    end_bpm: float = 140.0,
    duration_sec: float = 8.0,
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize a beat with a continuously accelerating or decelerating tempo.

    Constraints:
        start_bpm and end_bpm must be strictly positive (> 0). In musical timing,
        negative or zero BPM is physically meaningless.
    """
    if start_bpm <= 0.0 or end_bpm <= 0.0:
        raise ValueError(
            f"BPM values must be strictly positive. Got start_bpm={start_bpm}, end_bpm={end_bpm}."
        )

    total_samples = int(duration_sec * fs)
    buffer = np.zeros(total_samples, dtype=np.float32)

    kick = _synth_kick_drum(fs)
    snare = _synth_snare_drum(fs)

    beat_times = _compute_drift_beat_times(start_bpm, end_bpm, duration_sec)

    for beat_idx, t_beat in enumerate(beat_times):
        sample_pos = int(t_beat * fs)
        drum = kick if beat_idx % 2 == 0 else snare
        amplitude = 1.0 if beat_idx % 2 == 0 else 0.8
        _overlay_drum_hit(buffer, drum, sample_pos, amplitude=amplitude)

    return _normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


def apply_lossy_cutoff(
    audio: np.ndarray,
    cutoff_hz: float,
    fs: int = DEFAULT_FS,
    order: int = 10,
) -> np.ndarray:
    """Apply a steep digital lowpass filter simulating lossy encoder cutoffs."""
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        return audio.copy().astype(np.float32)

    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    filtered = signal.sosfilt(sos, audio)
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _normalize_peak(audio: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """Scale audio to a target peak amplitude without clipping."""
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak > 0:
        return (audio / peak * target_peak).astype(np.float32)
    return audio.astype(np.float32)


def _time_samples_for_frequency(duration_sec: float, fs: int) -> np.ndarray:
    """Return sampled time points [0, duration_sec) spaced at sampling frequency fs."""
    num_samples = int(duration_sec * fs)
    return np.arange(num_samples) / fs


def _synth_kick_drum(fs: int) -> np.ndarray:
    """Synthesize a short, punchy electronic kick drum transient (808 style).

    Acoustic Model:
        - Downward exponential pitch sweep from 150 Hz to 45 Hz with decay tau = 20 ms.
        - Exponential amplitude decay with tau = 35 ms (120 ms total duration).
        - High-frequency 1.2 kHz transient click overlaid during first 5 ms for beater attack.
    """
    duration = 0.12  # 120 ms
    num_samples = int(duration * fs)
    t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float32)

    tau_f = 0.02
    freq_inst = 45.0 + 105.0 * np.exp(-t / tau_f)
    phase = 2.0 * np.pi * np.cumsum(freq_inst) / fs

    amplitude_env = np.exp(-t / 0.035).astype(np.float32)
    kick = np.sin(phase).astype(np.float32) * amplitude_env

    click = np.sin(2.0 * np.pi * 1200.0 * t[: int(0.005 * fs)], dtype=np.float32) * 0.3
    kick[: len(click)] += click

    return kick.astype(np.float32)


def _synth_snare_drum(fs: int) -> np.ndarray:
    """Synthesize an electronic snare drum hit (noise burst + resonant body).

    Acoustic Model:
        - High-frequency white noise burst with 40 ms exponential decay (snare wires).
        - Resonant sinusoidal fundamental at 185 Hz with 25 ms decay (drum head body resonance).
    """
    duration = 0.15  # 150 ms
    num_samples = int(duration * fs)
    t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float32)

    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 0.5, num_samples).astype(np.float32)
    noise_env = np.exp(-t / 0.04).astype(np.float32)

    body_tone = (
        np.sin(2.0 * np.pi * 185.0 * t, dtype=np.float32) * np.exp(-t / 0.025).astype(np.float32) * 0.4
    )
    return (noise * noise_env + body_tone).astype(np.float32)


def _synth_hihat(fs: int) -> np.ndarray:
    """Synthesize a crisp, short closed hi-hat hit.

    Acoustic Model:
        - Differentiated white noise (1st-order high-pass / pre-emphasis filter).
        - Ultra-short 8 ms exponential decay envelope (metallic transient click).
    """
    duration = 0.04  # 40 ms
    num_samples = int(duration * fs)
    t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float32)

    rng = np.random.default_rng(456)
    noise = rng.normal(0.0, 0.5, num_samples).astype(np.float32)
    env = np.exp(-t / 0.008).astype(np.float32)

    hp_noise = np.diff(noise, prepend=0.0).astype(np.float32)
    return (hp_noise * env).astype(np.float32)


def _overlay_drum_hit(
    buffer: np.ndarray, hit: np.ndarray, start_idx: int, amplitude: float = 1.0
) -> None:
    """In-place additive overlay of a single drum hit into an audio buffer."""
    if start_idx < 0 or start_idx >= len(buffer):
        return
    end_idx = min(start_idx + len(hit), len(buffer))
    valid_len = end_idx - start_idx
    if valid_len > 0:
        buffer[start_idx:end_idx] += hit[:valid_len] * amplitude


def _is_tempo_stationary(bpm_slope: float) -> bool:
    """Check if tempo change is negligible within floating-point tolerance."""
    return abs(bpm_slope) < EPSILON_SLOPE


def _solve_accelerated_beat_time(
    beat_idx: int, initial_beat_freq: float, beat_accel: float
) -> float | None:
    """Solve for beat timestamp t_k under constant tempo acceleration via kinematics.

    Kinematics formulation:
        k(t) = v_0 * t + 0.5 * a * t^2
        0.5 * a * t^2 + v_0 * t - k = 0

    When decelerating (beat_accel < 0), instantaneous tempo eventually drops to zero.
    Beyond that point, no further beats can occur (the drummer halted), so the
    discriminant becomes negative and this function returns None.
    """
    quad_a = 0.5 * beat_accel
    quad_b = initial_beat_freq
    quad_c = -float(beat_idx)

    discriminant = quad_b**2 - 4.0 * quad_a * quad_c
    if discriminant < 0:
        return None

    return float((-quad_b + np.sqrt(discriminant)) / (2.0 * quad_a))


def _compute_next_beat_time(
    beat_idx: int, bpm_slope: float, initial_freq: float, accel: float
) -> float | None:
    """Calculate the arrival time of beat k under stationary or drifting tempo."""
    if _is_tempo_stationary(bpm_slope):
        return beat_idx / initial_freq
    return _solve_accelerated_beat_time(beat_idx, initial_freq, accel)


def _is_beat_within_duration(t_beat: float, duration_sec: float) -> bool:
    """Check if the beat timestamp falls within the audio buffer boundary."""
    return t_beat < duration_sec


def _compute_drift_beat_times(
    start_bpm: float, end_bpm: float, duration_sec: float
) -> list[float]:
    """Compute timestamps in seconds for every beat under linear tempo drift.

    Iterates through sequential beats until either:
    1. The beat timestamp exceeds the audio buffer duration.
    2. Deceleration brings the tempo to zero (drummer halted).
    """
    bpm_slope = (end_bpm - start_bpm) / duration_sec
    initial_freq = start_bpm / 60.0
    accel = bpm_slope / 60.0

    beat_times: list[float] = []
    beat_idx = 0

    while True:
        t_beat = _compute_next_beat_time(beat_idx, bpm_slope, initial_freq, accel)
        if t_beat is None or not _is_beat_within_duration(t_beat, duration_sec):
            break

        beat_times.append(t_beat)
        beat_idx += 1

    return beat_times
