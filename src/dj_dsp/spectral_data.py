"""Pure data structures (views/lenses) over raw audio signals.

Each dataclass represents a distinct analytical projection or "lens" over the same
underlying audio signal without modifying the source audio:
- SpectrumView: 1D frequency lens (Power Spectral Density across frequencies).
- SpectrogramView: 2D time-frequency lens (STFT) localizing spectral events in time.
- ThreeBandWaveformView: DJ 3-band performance lens (Low/Mid/High envelopes, CDJ-3000 style).
- RGBBlendedWaveformView: DJ blended-color performance lens (CDJ-2000 Nexus style).

All array fields are typed NumPy arrays with explicit dimensional invariants.
Derived scalar metrics and sub-band projections are exposed as methods/properties.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB = 80.0
RGB_NORMALIZATION_EPSILON = 1e-9


@dataclass(frozen=True)
class SpectrumView:
    """1D Frequency lens: Power Spectral Density (PSD) estimate across discrete frequencies.

    Fields:
        frequencies: 1D array of frequency bin center values in Hz, spanning [0.0, fs / 2].
        power_db: 1D array of power levels in dBFS. Same shape as frequencies.
    """

    frequencies: np.ndarray
    power_db: np.ndarray

    @property
    def peak_db(self) -> float:
        """Observed maximum power level across all frequency bins in dBFS."""
        return float(np.max(self.power_db))

    def crop(
        self,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalize_peak: bool = False,
    ) -> "SpectrumView":
        """Return a sub-band SpectrumView cropped to [f_min, f_max]."""
        return crop_spectrum_view(self, f_min=f_min, f_max=f_max, normalize_peak=normalize_peak)


@dataclass(frozen=True)
class SpectrogramView:
    """2D Time-Frequency lens: Short-Time Fourier Transform (STFT) power spectrogram.

    Fields:
        times: 1D array of segment center times in seconds.
        frequencies: 1D array of frequency bin center values in Hz.
        power_db: 2D array of power in dBFS with shape (N_frequencies, N_times).
        dynamic_range_db: Visual display dynamic range window width in decibels.
    """

    times: np.ndarray
    frequencies: np.ndarray
    power_db: np.ndarray
    dynamic_range_db: float = DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB

    @property
    def vmax(self) -> float:
        """Observed maximum power level across the time-frequency matrix in dBFS."""
        return float(np.max(self.power_db))

    @property
    def vmin(self) -> float:
        """Display floor in dBFS derived from vmax and dynamic_range_db."""
        return self.vmax - self.dynamic_range_db


@dataclass(frozen=True)
class ThreeBandWaveformView:
    """DJ 3-band performance lens: Pioneer CDJ-3000 / Rekordbox 6+ stacked waveform.

    Fields:
        times: 1D array of frame center times in seconds. Length N_frames.
        low: 1D array of RMS amplitude for the Bass band (< 250 Hz).
        mid: 1D array of RMS amplitude for the Mid band (250 Hz - 2.5 kHz).
        high: 1D array of RMS amplitude for the Treble band (>= 2.5 kHz).
    """

    times: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray

    def to_rgb_blended(
        self, envelope_upper: np.ndarray, envelope_lower: np.ndarray
    ) -> "RGBBlendedWaveformView":
        """Convert 3-band envelopes into a classic CDJ-2000 blended RGB waveform lens."""
        return blend_envelopes_to_rgb(self, envelope_upper, envelope_lower)


@dataclass(frozen=True)
class RGBBlendedWaveformView:
    """DJ blended-color performance lens: Pioneer CDJ-2000 Nexus waveform.

    Fields:
        times: 1D array of frame center times in seconds. Length N_frames.
        envelope_upper: 1D array of overall positive peak envelope (>= 0.0).
        envelope_lower: 1D array of overall negative peak envelope (<= 0.0).
        rgb_colors: 2D array of shape (N_frames, 3) with normalized (R, G, B) values in [0.0, 1.0].
    """

    times: np.ndarray
    envelope_upper: np.ndarray
    envelope_lower: np.ndarray
    rgb_colors: np.ndarray


@dataclass(frozen=True)
class CutoffDetectionAnalysis:
    """Heuristic high-frequency cutoff profiling; not a provenance assessment.

    Attributes:
        detected_cutoff_hz: Estimated brick-wall lowpass cutoff frequency in Hz.
        peak_shelf_db: Reference peak power level in the upper register (10–22 kHz).
        noise_floor_db: Average power level in the post-cutoff stopband.
        stopband_attenuation_db: Difference in dB between shelf reference and stopband noise floor.
        verdict: Plain-English description of the heuristic's bandlimiting flag.
        is_suspicious_transcode: Bandwidth/attenuation flag.
        confidence: Legacy fixed heuristic score in [0.0, 1.0].
    """

    detected_cutoff_hz: float
    peak_shelf_db: float
    noise_floor_db: float
    stopband_attenuation_db: float
    verdict: str
    is_suspicious_transcode: bool
    confidence: float


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def crop_spectrum_view(
    view: SpectrumView,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    normalize_peak: bool = False,
) -> SpectrumView:
    """Filter spectrum frequency bins to [f_min, f_max] and optionally normalize peak to 0 dBFS."""
    upper_bound = float(view.frequencies[-1]) if f_max is None else f_max
    mask = (view.frequencies >= f_min) & (view.frequencies <= upper_bound)

    sub_freqs = view.frequencies[mask]
    sub_power = view.power_db[mask].copy()

    if normalize_peak and len(sub_power) > 0:
        sub_power -= float(np.max(sub_power))

    return SpectrumView(frequencies=sub_freqs, power_db=sub_power)


def blend_envelopes_to_rgb(
    view: ThreeBandWaveformView,
    envelope_upper: np.ndarray,
    envelope_lower: np.ndarray,
) -> RGBBlendedWaveformView:
    """Compute normalized RGB ratios from Low, Mid, High envelopes for CDJ-2000 style rendering."""
    total_energy = view.low + view.mid + view.high + RGB_NORMALIZATION_EPSILON
    r = np.clip(view.low / total_energy, 0.0, 1.0)
    g = np.clip(view.mid / total_energy, 0.0, 1.0)
    b = np.clip(view.high / total_energy, 0.0, 1.0)
    rgb_colors = np.column_stack([r, g, b])

    return RGBBlendedWaveformView(
        times=view.times,
        envelope_upper=envelope_upper,
        envelope_lower=envelope_lower,
        rgb_colors=rgb_colors,
    )
