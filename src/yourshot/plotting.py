"""Minimal Matplotlib rendering functions for yourshot spectral views.

Follows clean plotting conventions:
- Pure rendering: functions consume typed *View dataclasses and draw onto provided or new Axes.
- Generous typography: 12-13pt titles, 11pt labels, 10pt ticks for presentation and notebook clarity.
- Vertical layout suitability: designed to render at full horizontal width (figsize 10-12 inches).
- Color fidelity: accurate Rekordbox 3-band colors (Low=Cyan, Mid=Orange/Amber, High=Blue).
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from yourshot.spectral_data import (
    RGBBlendedWaveformView,
    SpectrogramView,
    SpectrumView,
    ThreeBandWaveformView,
)

# Standard Rekordbox 3-Band Color Palette (hex)
REKORDBOX_CYAN = "#00bcd4"    # Low / Bass (< 250 Hz)
REKORDBOX_ORANGE = "#ff9800"  # Mid / Vocals & Synths (250 Hz - 2.5 kHz)
REKORDBOX_BLUE = "#3f51b5"    # High / Hi-Hats & Air (>= 2.5 kHz)
CUTOFF_RED = "#f44336"        # Lossy MP3 cutoff warning line


def plot_three_band_waveform(
    view: ThreeBandWaveformView,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (11.0, 4.0),
    title: str = "Rekordbox 3-Band Waveform (CDJ-3000 Style)",
) -> Tuple[plt.Figure, plt.Axes]:
    """Render CDJ-3000 style stacked 3-band amplitude envelopes (Low, Mid, High).

    Args:
        view: ThreeBandWaveformView containing times, low, mid, and high envelopes.
        ax: Existing Matplotlib Axes to draw on. If None, a new figure and axes are created.
        figsize: Canvas size in inches when creating a new figure.
        title: Plot title string.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Averaged (RMS) Amplitude", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, which="both", alpha=0.15, linestyle="--")
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
    """Render Welch PSD on a logarithmic or linear frequency axis.

    Args:
        view: SpectrumView containing frequencies and power_db.
        ax: Existing Matplotlib Axes. If None, creates a new figure.
        figsize: Canvas size in inches when creating a new figure.
        title: Plot title string.
        color: Line color.
        f_min: Minimum display frequency in Hz (defaults to 20 Hz).
        f_max: Maximum display frequency in Hz (defaults to Nyquist).
        log_freq: When True, plots frequency on a musical log scale. When False, linear.
        cutoff_marker_hz: Optional frequency to mark with a vertical dashed line.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if f_max is None:
        f_max = float(view.frequencies[-1])

    if log_freq:
        ax.semilogx(view.frequencies, view.power_db, color=color, lw=1.5, alpha=0.95, label="Power")
        ax.set_xlabel("Frequency (kHz)", fontsize=11)
        # Explicit standard musical log tick locations including sub-bass, mid bands, 10k, and 20k Nyquist edge
        standard_ticks = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
        visible_ticks = [t for t in standard_ticks if f_min <= t <= f_max]
        ax.set_xticks(visible_ticks)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda val, pos: f"{int(val / 1000)}k" if val >= 1000.0 else f"{int(val)}")
        )
    else:
        ax.plot(view.frequencies / 1000.0, view.power_db, color=color, lw=1.5, alpha=0.95, label="Power")
        ax.set_xlabel("Frequency (kHz)", fontsize=11)

    if cutoff_marker_hz is not None and cutoff_marker_hz <= f_max:
        marker_x = cutoff_marker_hz if log_freq else (cutoff_marker_hz / 1000.0)
        ax.axvline(
            marker_x,
            color=CUTOFF_RED,
            linestyle="--",
            lw=1.5,
            label=f"Cutoff ({cutoff_marker_hz / 1000:.1f} kHz)",
        )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    if log_freq:
        ax.set_xlim(f_min, f_max)
    else:
        ax.set_xlim(f_min / 1000.0, f_max / 1000.0)
    ax.set_ylim(max(-120.0, float(np.min(view.power_db)) - 5.0), max(0.0, view.peak_db + 6.0))

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Power (dBFS)", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, which="both", alpha=0.25)

    return fig, ax


def plot_stft_spectrogram(
    view: SpectrogramView,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (11.0, 4.5),
    title: str = "2D STFT Spectrogram (Time vs. Frequency)",
    cmap: str = "magma",
    cutoff_marker_hz: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render 2D STFT magnitude spectrogram with optional lossy codec cutoff marker.

    Args:
        view: SpectrogramView containing times, frequencies, power_db, vmin, and vmax.
        ax: Existing Matplotlib Axes. If None, creates a new figure.
        figsize: Canvas size in inches when creating a new figure.
        title: Plot title string.
        cmap: Matplotlib colormap (defaults to "magma").
        cutoff_marker_hz: Optional frequency in Hz to draw a horizontal dashed cutoff line.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Frequency (kHz)", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=10)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Power (dBFS)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    return fig, ax
