"""Command line interface for dj_dsp audio analysis, quality auditing, and shelf visualization."""

from pathlib import Path
import json
import os
import sys
import tempfile
import webbrowser
from typing import Optional
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from dj_dsp.plotting import plot_shelf_spectrum
from dj_dsp.rekordbox.cli_helpers import (
    format_text_report,
    resolve_pioneer_root,
    serialize_diff_summary,
    serialize_track_cues,
)
from dj_dsp.rekordbox.cue_sync import (
    apply_cue_sheet_diff,
    calculate_cue_sheet_diff,
    write_atomic,
)
from dj_dsp.rekordbox.rekordbox_parser import scan_usb_anlz
from dj_dsp.spectral import analyze_spectral_cutoff, compute_welch_spectrum
from dj_dsp.spectral_data import CutoffDetectionAnalysis

MIN_AUDIBLE_PEAK_THRESHOLD = 1e-6
DEFAULT_SHELF_F_MIN_HZ = 10000.0
INTERACTIVE_GUI_BACKENDS = ("qtagg", "tkagg", "gtk3agg", "gtk4agg", "wxagg")
NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "ps", "svg", "cairo")
PLOT_EXPORT_DPI = 150


@click.group()
def main() -> None:
    """DJ Audio DSP & MIR Toolkit CLI."""


@main.command(name="check-quality")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--duration",
    "-d",
    type=float,
    default=None,
    help="Analyze only the first N seconds (defaults to full track).",
)
@click.option(
    "--json-output",
    "-j",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON output.",
)
def check_quality(audio_file: Path, duration: float | None, json_output: bool) -> None:
    """Analyze AUDIO_FILE for high-frequency roll-off and transcode indicators."""
    audio, fs = load_audio_or_exit(audio_file, max_duration_sec=duration)
    analysis = analyze_spectral_cutoff(audio, fs=fs)

    if json_output:
        click.echo(format_quality_json(audio_file, fs, analysis))
    else:
        render_quality_report(audio_file, fs, analysis)

    if analysis.is_suspicious_transcode:
        sys.exit(1)


@main.group(name="rekordbox")
def rekordbox_group() -> None:
    """Rekordbox USB media analysis and cue synchronization commands."""


@rekordbox_group.command(name="cues")
@click.option(
    "--usb-path",
    "-u",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Path to mounted Pioneer USB root or PIONEER folder (auto-detected if omitted).",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format: human-readable 'text' or machine-readable 'json'.",
)
@click.option(
    "--update-cue-sheet",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to cue sheet markdown file to synchronize with extracted cues.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="When updating cue sheet, preview changes without writing to disk.",
)
@click.option(
    "--include-metadata",
    "-m",
    is_flag=True,
    default=False,
    help="Include Rekordbox internal analysis metadata (e.g. BPM).",
)
def rekordbox_cues(
    usb_path: Optional[Path],
    output_format: str,
    update_cue_sheet: Optional[Path],
    dry_run: bool,
    include_metadata: bool,
) -> None:
    """Extract cue points from Rekordbox USB and optionally update cue sheet markdown."""
    try:
        pioneer_root = resolve_pioneer_root(usb_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)

    tracks = scan_usb_anlz(pioneer_root)
    diff = None

    if update_cue_sheet is not None:
        if not update_cue_sheet.exists():
            click.echo(f"Error: Cue sheet file not found: {update_cue_sheet}", err=True)
            sys.exit(2)

        original_text = update_cue_sheet.read_text(encoding="utf-8")
        diff = calculate_cue_sheet_diff(original_text, tracks)

        if diff.changed_tracks and not dry_run:
            updated_text = apply_cue_sheet_diff(original_text, diff)
            write_atomic(update_cue_sheet, updated_text)

    if output_format.lower() == "json":
        result = {
            "pioneer_root": str(pioneer_root),
            "tracks": [serialize_track_cues(t, include_metadata=include_metadata) for t in tracks],
        }
        if diff is not None:
            sync_info = serialize_diff_summary(diff)
            sync_info["dry_run"] = dry_run
            sync_info["applied"] = bool(diff.changed_tracks and not dry_run)
            result["cue_sheet_sync"] = sync_info
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(format_text_report(tracks, diff=diff, include_metadata=include_metadata))
        if update_cue_sheet and dry_run and diff and diff.changed_tracks:
            click.echo("\n[Dry Run] No files modified on disk.")


@main.command(name="plot-shelf")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--duration",
    "-d",
    type=float,
    default=None,
    help="Analyze only the first N seconds (defaults to full track).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Save plot image to file instead of GUI window (e.g. shelf.png).",
)
@click.option(
    "--f-min",
    type=float,
    default=DEFAULT_SHELF_F_MIN_HZ,
    help=f"Minimum frequency to plot in Hz (default: {DEFAULT_SHELF_F_MIN_HZ:.0f}).",
)
def plot_shelf(audio_file: Path, duration: float | None, output: Path | None, f_min: float) -> None:
    """Plot high-frequency shelf (10–22 kHz) with 16–20 kHz hearing limit highlight."""
    if output is None:
        select_interactive_gui_backend()

    audio, fs = load_audio_or_exit(audio_file, max_duration_sec=duration)
    analysis = analyze_spectral_cutoff(audio, fs=fs)
    cutoff_marker_hz = analysis.detected_cutoff_hz if analysis.is_suspicious_transcode else None

    shelf_spec = compute_welch_spectrum(audio, fs=fs).crop(f_min=f_min, normalize_peak=True)
    title = f"High-Frequency Shelf: {audio_file.name} (Cutoff: {analysis.detected_cutoff_hz:,.0f} Hz)"
    fig, _ = plot_shelf_spectrum(
        shelf_spec,
        title=title,
        cutoff_marker_hz=cutoff_marker_hz,
        highlight_hearing_limit=True,
    )
    fig.tight_layout()

    display_or_save_figure(fig, output_path=output)


def load_audio_or_exit(file_path: Path, max_duration_sec: float | None = None) -> tuple[np.ndarray, int]:
    """Load normalized audio from disk or exit with status code 2 on read failure."""
    try:
        return load_normalized_mono_audio(file_path, max_duration_sec=max_duration_sec)
    except Exception as exc:
        click.echo(f"Error reading {file_path}: {exc}", err=True)
        sys.exit(2)


def load_normalized_mono_audio(
    file_path: Path, max_duration_sec: float | None = None
) -> tuple[np.ndarray, int]:
    """Read an audio file, downmix stereo channels to mono, slice duration, and peak normalize."""
    raw_data, fs = sf.read(file_path, dtype="float32")
    mono_audio = np.mean(raw_data, axis=1) if raw_data.ndim > 1 else raw_data

    if max_duration_sec is not None and max_duration_sec > 0:
        max_samples = int(max_duration_sec * fs)
        mono_audio = mono_audio[:max_samples]

    peak_amplitude = float(np.max(np.abs(mono_audio))) if len(mono_audio) > 0 else 0.0
    if peak_amplitude > MIN_AUDIBLE_PEAK_THRESHOLD:
        mono_audio = mono_audio / peak_amplitude

    return mono_audio.astype(np.float32), fs


def format_quality_json(audio_file: Path, fs: int, analysis: CutoffDetectionAnalysis) -> str:
    """Format spectral cutoff metrics and verdict into an indented JSON payload."""
    payload = {
        "file": str(audio_file),
        "sample_rate_hz": fs,
        "detected_cutoff_hz": round(analysis.detected_cutoff_hz, 1),
        "peak_shelf_db": round(analysis.peak_shelf_db, 2),
        "noise_floor_db": round(analysis.noise_floor_db, 2),
        "stopband_attenuation_db": round(analysis.stopband_attenuation_db, 2),
        "is_suspicious_transcode": analysis.is_suspicious_transcode,
        "verdict": analysis.verdict,
        "confidence": round(analysis.confidence, 2),
    }
    return json.dumps(payload, indent=2)


def render_quality_report(audio_file: Path, fs: int, analysis: CutoffDetectionAnalysis) -> None:
    """Print formatted terminal report showing cutoff frequency and passband/stopband power levels."""
    status_color = "red" if analysis.is_suspicious_transcode else "green"
    status_tag = "SUSPICIOUS TRANSCODE" if analysis.is_suspicious_transcode else "FULL BANDWIDTH"

    click.echo(f"\nFile: {click.style(str(audio_file), bold=True)}")
    click.echo(f"Sample Rate: {fs} Hz")
    click.echo(
        f"Status: [{click.style(status_tag, fg=status_color, bold=True)}] {analysis.verdict}"
    )
    click.echo(f"Detected Cutoff: {analysis.detected_cutoff_hz:,.0f} Hz")
    click.echo(f"Stopband Attenuation: {analysis.stopband_attenuation_db:.1f} dB")
    click.echo(f"Passband Ref Level: {analysis.peak_shelf_db:.1f} dBFS")
    click.echo(f"Stopband Noise Floor: {analysis.noise_floor_db:.1f} dBFS\n")


def display_or_save_figure(fig: plt.Figure, output_path: Path | None = None) -> None:
    """Render a Matplotlib figure: save to file, show via GUI backend, or open in system viewer."""
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_EXPORT_DPI, bbox_inches="tight")
        click.echo(f"Saved shelf plot to {output_path}")
        plt.close(fig)
        return

    current_backend = matplotlib.get_backend().lower()
    if current_backend not in NON_INTERACTIVE_BACKENDS:
        plt.show()
        return

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = Path(tmp.name)
    fig.savefig(temp_path, dpi=PLOT_EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Opening plot in default viewer: {temp_path}")
    webbrowser.open(temp_path.as_uri())


def select_interactive_gui_backend() -> None:
    """Probe and activate an available interactive GUI backend if a display server is present."""
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        return

    current = matplotlib.get_backend().lower()
    if current in INTERACTIVE_GUI_BACKENDS:
        return

    for backend in INTERACTIVE_GUI_BACKENDS:
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue


if __name__ == "__main__":
    main()
