"""Synthetic ground-truth audio signals and codec cutoff filter simulations.

Pure NumPy / SciPy implementation for validating transcode detection
and evaluating lossy compression artifacts.
"""

from typing import Tuple
import shutil
import subprocess
import tempfile
import numpy as np
from scipy import signal
import soundfile as sf

DEFAULT_FS = 44100
DEFAULT_HEADROOM_RATIO = 0.85
MAX_PEAK_AMPLITUDE = 0.95
EPSILON_SLOPE = 1e-6

# Default codec cutoff frequency thresholds in Hz
LAME_FALLBACK_CUTOFF_MAP = {
    64: 11000.0,
    96: 14000.0,
    128: 16000.0,
    192: 18500.0,
    320: 20000.0,
}


def generate_white_noise(duration_sec: float = 3.0, fs: int = DEFAULT_FS) -> np.ndarray:
    """Generate wideband Gaussian white noise scaled to -6 dBFS."""
    num_samples = int(duration_sec * fs)
    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=0.2, size=num_samples)
    return normalize_peak(noise, target_peak=0.5)


def generate_chirp(
    duration_sec: float = 3.0,
    fs: int = DEFAULT_FS,
    f_start: float = 20.0,
    f_end: float = 22050.0,
) -> np.ndarray:
    """Generate an exponential frequency sweep (chirp) from f_start to f_end."""
    t = time_samples_for_frequency(duration_sec, fs)
    sweep = signal.chirp(t, f0=f_start, t1=duration_sec, f1=f_end, method="logarithmic")
    return (sweep * 0.5).astype(np.float32)


def generate_linear_chirp(
    duration_sec: float = 4.0,
    fs: int = DEFAULT_FS,
    f_start: float = 10000.0,
    f_end: float = 20000.0,
) -> np.ndarray:
    """Generate a linear frequency sweep across the top register (10 kHz to 20 kHz)."""
    t = time_samples_for_frequency(duration_sec, fs)
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

    return normalize_peak(signal_out, target_peak=0.9)


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

    return normalize_peak(output, target_peak=0.7)


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

    return normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


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

    return normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


def generate_syncopated_breakbeat(
    bpm: float = 140.0,
    duration_sec: float = 8.0,
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize a syncopated funk breakbeat with offbeat anticipations."""
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

        if bar_step % 2 == 0:
            _overlay_drum_hit(buffer, hihat, step_pos, amplitude=0.35)

        if bar_step in (0, 7, 10):
            _overlay_drum_hit(buffer, kick, step_pos, amplitude=0.95)

        if bar_step in (4, 12):
            _overlay_drum_hit(buffer, snare, step_pos, amplitude=0.8)

    return normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


def generate_tempo_drift_beat(
    start_bpm: float = 120.0,
    end_bpm: float = 140.0,
    duration_sec: float = 8.0,
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Synthesize a beat with a continuously accelerating or decelerating tempo."""
    total_samples = int(duration_sec * fs)
    buffer = np.zeros(total_samples, dtype=np.float32)

    kick = _synth_kick_drum(fs)
    snare = _synth_snare_drum(fs)

    slope = (end_bpm - start_bpm) / duration_sec
    beat_idx = 0
    while True:
        if abs(slope) < EPSILON_SLOPE:
            t_beat = beat_idx * (60.0 / start_bpm)
        else:
            a = 0.5 * slope / 60.0
            b = start_bpm / 60.0
            c = -float(beat_idx)
            discriminant = b**2 - 4.0 * a * c
            if discriminant < 0:
                break
            t_beat = (-b + np.sqrt(discriminant)) / (2.0 * a)

        if t_beat >= duration_sec:
            break

        sample_pos = int(t_beat * fs)
        if beat_idx % 2 == 0:
            _overlay_drum_hit(buffer, kick, sample_pos, amplitude=1.0)
        else:
            _overlay_drum_hit(buffer, snare, sample_pos, amplitude=0.8)

        beat_idx += 1

    return normalize_peak(buffer, target_peak=DEFAULT_HEADROOM_RATIO)


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


def encode_mp3_lame(
    audio: np.ndarray,
    fs: int = DEFAULT_FS,
    bitrate_kbps: int = 128,
) -> np.ndarray:
    """Encode audio through genuine LAME MP3 encoder via ffmpeg with lowpass fallback."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        fc = LAME_FALLBACK_CUTOFF_MAP.get(bitrate_kbps, 16000.0)
        return apply_lossy_cutoff(audio, cutoff_hz=fc, fs=fs)

    return encode_mp3_subprocess(audio, fs=fs, bitrate_kbps=bitrate_kbps, ffmpeg_bin=ffmpeg_bin)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def normalize_peak(audio: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """Scale audio to a target peak amplitude without clipping."""
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak > 0:
        return (audio / peak * target_peak).astype(np.float32)
    return audio.astype(np.float32)


def time_samples_for_frequency(duration_sec: float, fs: int) -> np.ndarray:
    """Return sampled time points [0, duration_sec) spaced at sampling frequency fs."""
    num_samples = int(duration_sec * fs)
    return np.arange(num_samples) / fs


def _synth_kick_drum(fs: int) -> np.ndarray:
    """Synthesize a short, punchy electronic kick drum transient."""
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
    """Synthesize an electronic snare drum hit (noise burst + resonant body)."""
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
    """Synthesize a crisp, short closed hi-hat hit."""
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


def encode_mp3_subprocess(
    audio: np.ndarray,
    fs: int,
    bitrate_kbps: int,
    ffmpeg_bin: str,
) -> np.ndarray:
    """Encode audio to MP3 and decode back using an external ffmpeg process."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f_in, \
         tempfile.NamedTemporaryFile(suffix=".mp3") as f_mp3, \
         tempfile.NamedTemporaryFile(suffix=".wav") as f_out:

        stereo_audio = np.column_stack([audio, audio]) if audio.ndim == 1 else audio
        sf.write(f_in.name, stereo_audio, fs, subtype="PCM_16")

        cmd_enc = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", f_in.name,
            "-codec:a", "libmp3lame",
            "-b:a", f"{bitrate_kbps}k",
            f_mp3.name,
        ]
        subprocess.run(cmd_enc, capture_output=True, check=True)

        cmd_dec = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", f_mp3.name,
            "-ar", str(fs),
            "-ac", "1",
            f_out.name,
        ]
        subprocess.run(cmd_dec, capture_output=True, check=True)

        decoded, _ = sf.read(f_out.name, dtype="float32")

        orig_len = len(audio)
        if len(decoded) > orig_len:
            decoded = decoded[:orig_len]
        elif len(decoded) < orig_len:
            decoded = np.pad(decoded, (0, orig_len - len(decoded)))

        return np.clip(decoded, -1.0, 1.0).astype(np.float32)
