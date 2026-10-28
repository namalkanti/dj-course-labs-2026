"""Minimal Matplotlib rendering functions for dj_dsp spectral views.

Follows clean plotting conventions:
- Pure rendering: functions consume typed *View dataclasses and draw onto provided or new Axes.
- Generous typography: 12-13pt titles, 11pt labels, 10pt ticks for presentation clarity.
- Color fidelity: accurate Rekordbox 3-band colors (Low=Cyan, Mid=Orange/Amber, High=Blue).
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from dj_dsp.spectral_data import (
    RGBBlendedWaveformView,
    SpectrogramView,
    SpectrumView,
    ThreeBandWaveformView,
)

REKORDBOX_CYAN = "#00bcd4"    # Low / Bass (< 250 Hz)
REKORDBOX_ORANGE = "#ff9800"  # Mid / Vocals & Synths (250 Hz - 2.5 kHz)
REKORDBOX_BLUE = "#3f51b5"    # High / Hi-Hats & Air (>= 2.5 kHz)
CUTOFF_RED = "#f44336"        # Lossy MP3 cutoff warning line
HUMAN_HEARING_COLOR = "#ab47bc"  # Purple accent for upper human hearing limit
HUMAN_HEARING_SPAN_HZ = (16000.0, 20000.0)
MIN_DISPLAY_DBFS = -120.0
STANDARD_LOG_TICKS_HZ = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]


def plot_three_band_waveform(
    view: ThreeBandWaveformView,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (11.0, 4.0),
    title: str = "Rekordbox 3-Band Waveform (CDJ-3000 Style)",
) -> Tuple[plt.Figure, plt.Axes]:
    """Render CDJ-3000 style stacked 3-band amplitude envelopes (Low, Mid, High)."""
    fig, ax = ensure_axes(ax, figsize)

    base_low = view.low
    base_mid = base_low + view.mid
    base_high = base_mid + view.high

    ax.fill_between(view.times, -base_high, base_high, color=REKORDBOX_BLUE, alpha=0.9, label="High (>= 2.5 kHz)")
    ax.fill_between(view.times, -base_mid, base_mid, color=REKORDBOX_ORANGE, alpha=0.95, label="Mid (250 Hz - 2.5 kHz)")
    ax.fill_between(view.times, -base_low, base_low, color=REKORDBOX_CYAN, alpha=1.0, label="Low (< 250 Hz)")
    ax.axhline(0.0, color="#ffffff", lw=0.5, alpha=0.4)

    ax.set_xlim(view.times[0], view.times[-1])
    max_val = float(np.max(base_high)) * 1.08 if len(base_high) > 0 else 1.0
    ax.set_ylim(-max_val, max_val)

    apply_standard_plot_styling(ax, title=title, xlabel="Time (seconds)", ylabel="Averaged (RMS) Amplitude")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    return fig, ax


def plot_welch_spectrum(
    view: SpectrumView,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (11.0, 4.0),
    title: str = "Averaged Power Spectrum (Welch PSD)",
    color: str = REKORDBOX_CYAN,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
    log_freq: bool = True,
    cutoff_marker_hz: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render Welch PSD on a logarithmic or linear frequency axis."""
    fig, ax = ensure_axes(ax, figsize)
    upper_bound = float(view.frequencies[-1]) if f_max is None else f_max

    if log_freq:
        ax.semilogx(view.frequencies, view.power_db, color=color, lw=1.5, alpha=0.95, label="Power")
        visible_ticks = [t for t in STANDARD_LOG_TICKS_HZ if f_min <= t <= upper_bound]
        ax.set_xticks(visible_ticks)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda val, pos: f"{int(val / 1000)}k" if val >= 1000.0 else f"{int(val)}")
        )
        ax.set_xlim(f_min, upper_bound)
    else:
        ax.plot(view.frequencies / 1000.0, view.power_db, color=color, lw=1.5, alpha=0.95, label="Power")
        ax.set_xlim(f_min / 1000.0, upper_bound / 1000.0)

    draw_cutoff_line(ax, cutoff_marker_hz, is_khz=(not log_freq), f_max_bound=upper_bound)

    min_y = max(MIN_DISPLAY_DBFS, float(np.min(view.power_db)) - 5.0)
    max_y = max(0.0, view.peak_db + 6.0)
    ax.set_ylim(min_y, max_y)

    apply_standard_plot_styling(ax, title=title, xlabel="Frequency (kHz)", ylabel="Power (dBFS)")
    if cutoff_marker_hz is not None and cutoff_marker_hz <= upper_bound:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    return fig, ax


def plot_stft_spectrogram(
    view: SpectrogramView,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (11.0, 4.5),
    title: str = "2D STFT Spectrogram (Time vs. Frequency)",
    cmap: str = "magma",
    cutoff_marker_hz: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render 2D STFT magnitude spectrogram with optional lossy codec cutoff marker."""
    fig, ax = ensure_axes(ax, figsize)

    mesh = ax.pcolormesh(
        view.times,
        view.frequencies / 1000.0,
        view.power_db,
        shading="gouraud",
        cmap=cmap,
        vmin=view.vmin,
        vmax=view.vmax,
    )

    if cutoff_marker_hz is not None and cutoff_marker_hz < view.frequencies[-1]:
        ax.axhline(
            cutoff_marker_hz / 1000.0,
            color=CUTOFF_RED,
            linestyle="--",
            lw=1.6,
            label=f"Cutoff Marker ({cutoff_marker_hz / 1000:.1f} kHz)",
        )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_ylim(0.0, view.frequencies[-1] / 1000.0)
    apply_standard_plot_styling(ax, title=title, xlabel="Time (seconds)", ylabel="Frequency (kHz)")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Power (dBFS)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    return fig, ax


def plot_shelf_spectrum(
    view: SpectrumView,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (11.0, 4.0),
    title: str = "High-Frequency Shelf (10–22 kHz)",
    cutoff_marker_hz: Optional[float] = None,
    color: str = REKORDBOX_CYAN,
    label: str = "Audio Spectrum",
    highlight_hearing_limit: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a single audio track's Welch power spectrum across the high shelf (linear scale)."""
    fig, ax = ensure_axes(ax, figsize)

    f_khz = view.frequencies / 1000.0
    f_min_khz, f_max_khz = float(f_khz[0]), float(f_khz[-1])

    if highlight_hearing_limit:
        draw_hearing_limit_span(ax, f_min_khz, f_max_khz)

    ax.plot(f_khz, view.power_db, color=color, lw=2.0, alpha=0.95, label=label, zorder=2)
    draw_cutoff_line(ax, cutoff_marker_hz, is_khz=True, f_max_bound=f_max_khz * 1000.0)

    ax.set_xlim(f_min_khz, f_max_khz)
    ax.set_ylim(max(MIN_DISPLAY_DBFS, float(np.min(view.power_db)) - 5.0), max(0.0, view.peak_db + 6.0))

    apply_standard_plot_styling(ax, title=title, xlabel="Frequency (kHz)", ylabel="Power (dBFS)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    return fig, ax


def plot_overlaid_welch_shelf(
    view_orig: SpectrumView,
    view_filt: SpectrumView,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (11.0, 4.0),
    title: str = "High-Frequency Shelf: Pristine vs. Filtered (10–22 kHz)",
    cutoff_marker_hz: Optional[float] = None,
    orig_label: str = "Pristine Master",
    filt_label: str = "Filtered Audio",
    highlight_hearing_limit: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render pristine vs filtered Welch power spectra overlaid on the same linear frequency axes."""
    fig, ax = ensure_axes(ax, figsize)

    f_orig_khz = view_orig.frequencies / 1000.0
    f_filt_khz = view_filt.frequencies / 1000.0
    f_min_khz = float(min(f_orig_khz[0], f_filt_khz[0]))
    f_max_khz = float(max(f_orig_khz[-1], f_filt_khz[-1]))

    if highlight_hearing_limit:
        draw_hearing_limit_span(ax, f_min_khz, f_max_khz)

    ax.plot(f_orig_khz, view_orig.power_db, color="#78909c", lw=2.0, alpha=0.85, label=orig_label, zorder=2)
    ax.plot(f_filt_khz, view_filt.power_db, color=REKORDBOX_ORANGE, lw=2.0, alpha=0.95, label=filt_label, zorder=2)

    draw_cutoff_line(ax, cutoff_marker_hz, is_khz=True, f_max_bound=f_max_khz * 1000.0)

    ax.set_xlim(f_min_khz, f_max_khz)
    min_db = min(float(np.min(view_orig.power_db)), float(np.min(view_filt.power_db)))
    max_db = max(float(view_orig.peak_db), float(view_filt.peak_db))
    ax.set_ylim(max(MIN_DISPLAY_DBFS, min_db - 5.0), max(0.0, max_db + 6.0))

    apply_standard_plot_styling(ax, title=title, xlabel="Frequency (kHz)", ylabel="Power (dBFS)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    return fig, ax


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def ensure_axes(ax: Optional[plt.Axes], figsize: Tuple[float, float]) -> Tuple[plt.Figure, plt.Axes]:
    """Return existing axes and figure or create a new Matplotlib figure canvas."""
    if ax is None:
        fig, created_ax = plt.subplots(figsize=figsize)
        return fig, created_ax
    return ax.figure, ax


def apply_standard_plot_styling(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent title, label, tick typography and subtle background grid."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, which="both", alpha=0.25)


def draw_cutoff_line(
    ax: plt.Axes, cutoff_marker_hz: Optional[float], is_khz: bool, f_max_bound: float
) -> None:
    """Annotate a detected cutoff frequency with a red dashed line if within bounds."""
    if cutoff_marker_hz is None or cutoff_marker_hz > f_max_bound:
        return

    x_pos = (cutoff_marker_hz / 1000.0) if is_khz else cutoff_marker_hz
    ax.axvline(
        x_pos,
        color=CUTOFF_RED,
        linestyle="--",
        lw=1.6,
        label=f"Cutoff ({cutoff_marker_hz / 1000.0:.1f} kHz)",
        zorder=3,
    )


def draw_hearing_limit_span(ax: plt.Axes, f_min_khz: float, f_max_khz: float) -> None:
    """Shade the upper human hearing limit (16–20 kHz) if within visible frequency bounds."""
    hearing_low_khz = HUMAN_HEARING_SPAN_HZ[0] / 1000.0
    hearing_high_khz = HUMAN_HEARING_SPAN_HZ[1] / 1000.0

    if f_max_khz < hearing_low_khz:
        return

    span_start = max(f_min_khz, hearing_low_khz)
    span_end = min(f_max_khz, hearing_high_khz)
    if span_end > span_start:
        ax.axvspan(
            span_start,
            span_end,
            color=HUMAN_HEARING_COLOR,
            alpha=0.12,
            label="Upper Hearing Limit (16–20 kHz)",
            zorder=0,
        )
