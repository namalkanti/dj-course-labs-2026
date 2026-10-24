"""Pure data structures (views/lenses) over raw audio signals.

Each dataclass represents a distinct analytical projection or "lens" over the same
underlying audio signal without modifying the source audio:
- SpectrumView: Wide-angle frequency lens (0 to Nyquist) showing global tonal balance.
- SpectrogramView: 2D time-frequency lens (STFT) localizing spectral events in time.
- HighFrequencyShelfView: Telephoto inspection lens (10 kHz to Nyquist) exposing codec cutoffs.
- ThreeBandWaveformView: DJ 3-band performance lens (Low/Mid/High envelopes, CDJ-3000 style).
- RGBBlendedWaveformView: DJ blended-color performance lens (CDJ-2000 Nexus style).

All array fields are typed NumPy arrays with explicit dimensional invariants.
Derived scalar metrics (e.g. peak levels and display bounds) are exposed as
read-only properties to keep the underlying arrays as the single source of truth.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpectrumView:
    """Wide-angle frequency lens: 1D Power Spectral Density (PSD) estimate across discrete frequencies.

    Fields:
        frequencies: 1D array of frequency bin center values in Hz, spanning [0.0, fs / 2].
                     Length is typically (window_length // 2 + 1).
        power_db: 1D array of power levels in decibels relative to full scale (dBFS),
                  computed as 10 * log10(psd + eps). Same shape as frequencies.
    """

    frequencies: np.ndarray
    power_db: np.ndarray

    @property
    def peak_db(self) -> float:
        """Observed maximum power level across all frequency bins in dBFS (float(np.max(power_db)))."""
        return float(np.max(self.power_db))


@dataclass(frozen=True)
class SpectrogramView:
    """2D Time-Frequency lens: Short-Time Fourier Transform (STFT) power spectrogram.

    All decibel values are in dBFS (decibels relative to full scale, where 0 dBFS is
    digital clipping). The display window clamps values to an 80 dB span below peak
    (e.g. from -80 dBFS up to 0 dBFS) to prevent float quantization noise from
    diluting visual contrast.

    Fields:
        times: 1D array of segment center times in seconds, spanning [0, duration_sec].
               Length matches the number of time frames N_times.
        frequencies: 1D array of frequency bin center values in Hz, spanning [0.0, fs / 2].
                     Length is (window_length // 2 + 1).
        power_db: 2D array of power in dBFS with shape (N_frequencies, N_times).
                  Clamped to [vmin, vmax] (e.g. [-80 dBFS, 0 dBFS]).
        dynamic_range_db: Visual display dynamic range window width in decibels (defaults to 80.0 dB).
    """

    times: np.ndarray
    frequencies: np.ndarray
    power_db: np.ndarray
    dynamic_range_db: float = 80.0

    @property
    def vmax(self) -> float:
        """Observed maximum power level across the time-frequency matrix in dBFS (typically <= 0 dBFS)."""
        return float(np.max(self.power_db))

    @property
    def vmin(self) -> float:
        """Display floor in dBFS derived from vmax and dynamic_range_db (vmax - dynamic_range_db, e.g. -80 dBFS)."""
        return self.vmax - self.dynamic_range_db


@dataclass(frozen=True)
class ThreeBandWaveformView:
    """DJ 3-band performance lens: Pioneer CDJ-3000 / Rekordbox 6+ stacked waveform.

    Each frequency band is modeled as an independent amplitude envelope over time:
    - low: Bass register (< 250 Hz, rendered as Cyan/Blue in Rekordbox).
    - mid: Midrange register (250 Hz - 2.5 kHz, rendered as Orange/Yellow).
    - high: Treble register (>= 2.5 kHz, rendered as Blue/White).

    Fields:
        times: 1D array of frame center times in seconds. Length N_frames.
        low: 1D array of RMS/peak amplitude for the Bass band (>= 0.0).
        mid: 1D array of RMS/peak amplitude for the Mid band (>= 0.0).
        high: 1D array of RMS/peak amplitude for the Treble band (>= 0.0).
    """

    times: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray

    def to_rgb_blended(
        self, envelope_upper: np.ndarray, envelope_lower: np.ndarray
    ) -> "RGBBlendedWaveformView":
        """Convert 3-band envelopes into a classic CDJ-2000 blended RGB waveform lens."""
        total_energy = self.low + self.mid + self.high + 1e-9
        r = np.clip(self.low / total_energy, 0.0, 1.0)
        g = np.clip(self.mid / total_energy, 0.0, 1.0)
        b = np.clip(self.high / total_energy, 0.0, 1.0)
        rgb_colors = np.column_stack([r, g, b])

        return RGBBlendedWaveformView(
            times=self.times,
            envelope_upper=envelope_upper,
            envelope_lower=envelope_lower,
            rgb_colors=rgb_colors,
        )


@dataclass(frozen=True)
class RGBBlendedWaveformView:
    """DJ blended-color performance lens: Pioneer CDJ-2000 Nexus waveform.

    Renders a single overall waveform envelope whose fill color dynamically blends
    (R, G, B) based on relative low/mid/high energy balance.

    Fields:
        times: 1D array of frame center times in seconds. Length N_frames.
        envelope_upper: 1D array of overall full-signal positive peak envelope (>= 0.0).
        envelope_lower: 1D array of overall full-signal negative peak envelope (<= 0.0).
        rgb_colors: 2D array of shape (N_frames, 3) with normalized (R, G, B) float values in [0.0, 1.0].
    """

    times: np.ndarray
    envelope_upper: np.ndarray
    envelope_lower: np.ndarray
    rgb_colors: np.ndarray


@dataclass(frozen=True)
class HighFrequencyShelfView:
    """Telephoto inspection lens: isolated high-frequency spectral shelf (e.g. 10 kHz to Nyquist).

    Used for codec lowpass cutoff inspection (128k MP3 shelf at 16 kHz vs Lossless).

    Fields:
        frequencies: 1D array of frequency bin values in Hz, spanning [f_min, fs / 2].
        normalized_power_db: 1D array of power in decibels, normalized so the peak
                             in the [f_min, Nyquist] band is 0.0 dBFS.
    """

    frequencies: np.ndarray
    normalized_power_db: np.ndarray
