"""Unit tests verifying CLI command behavior, status exit codes, and output formats."""

from pathlib import Path
import json
from click.testing import CliRunner, Result
import numpy as np
import soundfile as sf

from dj_dsp.cli import main
from dj_dsp.synth import apply_lossy_cutoff, generate_drum_loop
from tests.utils.synthetic_anlz import (
    build_anlz_file,
    build_pcp2_entry,
    build_pco2_tag,
    build_ppth_tag,
    build_pqtz_tag,
)

# ---------------------------------------------------------------------------
# Tests (Specifications)
# ---------------------------------------------------------------------------


def test_cli_help():
    """Verify that CLI root command displays help output and succeeds with exit code 0."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert_command_succeeded(result)
    assert "DJ Audio DSP & MIR Toolkit CLI" in result.output


def test_cli_check_quality_flac():
    """Verify that `check-quality` recognizes a lossless FLAC fixture as full bandwidth."""
    runner = CliRunner()
    flac_path = Path("audio_catalog/persistence_by_dabmasterars_snippet.flac")
    if not flac_path.exists():
        return

    result = runner.invoke(main, ["check-quality", str(flac_path)])
    assert_command_succeeded(result)
    assert "FULL BANDWIDTH" in result.output


def test_cli_check_quality_json_transcode(tmp_path: Path):
    """Verify that `check-quality -j` reports transcode status in JSON with non-zero exit code.

    Invariants:
    - CLI returns exit code 1 to indicate suspicious low-bitrate transcode.
    - JSON output contains `is_suspicious_transcode: true` and `detected_cutoff_hz < 19.5 kHz`.
    """
    fs = 44100
    groove = generate_drum_loop(duration_sec=3.0, fs=fs)
    cut_audio = apply_lossy_cutoff(groove, cutoff_hz=16000.0, fs=fs)
    test_file = tmp_path / "transcode_test.wav"
    sf.write(test_file, cut_audio, fs)

    runner = CliRunner()
    result = runner.invoke(main, ["check-quality", "-j", str(test_file)])
    assert_transcode_detected_exit_code(result)

    payload = json.loads(result.output)
    assert_transcode_payload_matches(payload, max_cutoff_hz=19500.0)


def test_cli_plot_shelf_to_file(tmp_path: Path):
    """Verify that `plot-shelf` writes a non-empty PNG spectral plot file to disk."""
    fs = 44100
    groove = generate_drum_loop(duration_sec=2.0, fs=fs)
    test_file = tmp_path / "audio.wav"
    sf.write(test_file, groove, fs)

    out_png = tmp_path / "shelf.png"
    runner = CliRunner()
    result = runner.invoke(main, ["plot-shelf", "-o", str(out_png), str(test_file)])
    assert_command_succeeded(result)
    assert_non_empty_file_created(out_png)


def test_cli_rekordbox_cues_json_output(tmp_path: Path):
    """Verify `rekordbox cues --format json` produces valid schema-compliant JSON output."""
    pioneer_dir = tmp_path / "PIONEER" / "USBANLZ" / "track1"
    pioneer_dir.mkdir(parents=True)

    ppth = build_ppth_tag("/Contents/Artist/CLI Track.wav")
    pqtz = build_pqtz_tag(bpm=128.0)
    hot = build_pcp2_entry(hot_cue_num=1, time_ms=10000, comment="Intro")
    pco2 = build_pco2_tag(cue_obj_type=0, entries=[hot])

    (pioneer_dir / "ANLZ0000.EXT").write_bytes(build_anlz_file([ppth, pco2]))
    (pioneer_dir / "ANLZ0000.DAT").write_bytes(build_anlz_file([ppth, pqtz]))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["rekordbox", "cues", "--usb-path", str(tmp_path), "--format", "json", "--include-metadata"],
    )

    assert_command_succeeded(result)
    data = json.loads(result.output)
    assert "tracks" in data
    assert len(data["tracks"]) == 1
    track_data = data["tracks"][0]
    assert track_data["filename"] == "CLI Track.wav"
    assert track_data["bpm"] == 128.0
    assert len(track_data["hot_cues"]) == 1
    assert track_data["hot_cues"][0]["slot"] == "A"
    assert track_data["hot_cues"][0]["time_ms"] == 10000
    assert track_data["hot_cues"][0]["formatted_time"] == "0:10.000"


def test_cli_rekordbox_cues_sync_dry_run(tmp_path: Path):
    """Verify `rekordbox cues --dry-run` calculates diff without modifying target file on disk."""
    pioneer_dir = tmp_path / "PIONEER" / "USBANLZ" / "track1"
    pioneer_dir.mkdir(parents=True)

    ppth = build_ppth_tag("/Contents/TestSong.wav")
    hot = build_pcp2_entry(hot_cue_num=1, time_ms=30000, comment="New Drop")
    pco2 = build_pco2_tag(cue_obj_type=0, entries=[hot])

    (pioneer_dir / "ANLZ0000.EXT").write_bytes(build_anlz_file([ppth, pco2]))
    (pioneer_dir / "ANLZ0000.DAT").write_bytes(build_anlz_file([ppth]))

    cue_sheet_path = tmp_path / "cue-sheet.md"
    initial_content = (
        "### 01. Test Artist - TestSong\n"
        "- **File**: `TestSong.wav`\n"
        "- **Cues & Loops**:\n"
        "  - `Hot Cue A` [Pad 1]: `0:00.000` (0 ms) — Old Drop.\n"
    )
    cue_sheet_path.write_text(initial_content, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rekordbox",
            "cues",
            "--usb-path",
            str(tmp_path),
            "--format",
            "json",
            "--update-cue-sheet",
            str(cue_sheet_path),
            "--dry-run",
        ],
    )

    assert_command_succeeded(result)
    data = json.loads(result.output)
    assert data["cue_sheet_sync"]["dry_run"] is True
    assert data["cue_sheet_sync"]["applied"] is False
    assert data["cue_sheet_sync"]["total_changed"] == 1
    assert cue_sheet_path.read_text(encoding="utf-8") == initial_content


def test_cli_rekordbox_cues_sync_atomic_apply(tmp_path: Path):
    """Verify `rekordbox cues --update-cue-sheet` applies updates to disk when not in dry-run."""
    pioneer_dir = tmp_path / "PIONEER" / "USBANLZ" / "track1"
    pioneer_dir.mkdir(parents=True)

    # Cue point has comment="" so existing annotation in markdown should be preserved
    ppth = build_ppth_tag("/Contents/TestSong.wav")
    hot = build_pcp2_entry(hot_cue_num=1, time_ms=30000, comment="")
    pco2 = build_pco2_tag(cue_obj_type=0, entries=[hot])

    (pioneer_dir / "ANLZ0000.EXT").write_bytes(build_anlz_file([ppth, pco2]))
    (pioneer_dir / "ANLZ0000.DAT").write_bytes(build_anlz_file([ppth]))

    cue_sheet_path = tmp_path / "cue-sheet.md"
    initial_content = (
        "### 01. Test Artist - TestSong\n"
        "- **File**: `TestSong.wav`\n"
        "- **Cues & Loops**:\n"
        "  - `Hot Cue A` [Pad 1]: `0:00.000` (0 ms) — Note.\n"
    )
    cue_sheet_path.write_text(initial_content, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rekordbox",
            "cues",
            "--usb-path",
            str(tmp_path),
            "--update-cue-sheet",
            str(cue_sheet_path),
        ],
    )

    assert_command_succeeded(result)
    updated_content = cue_sheet_path.read_text(encoding="utf-8")
    assert "0:30.000" in updated_content
    assert "Note." in updated_content


def test_cli_rekordbox_cues_missing_path_exit_code():
    """Verify `rekordbox cues` exits with status 2 when supplied an invalid USB path."""
    runner = CliRunner()
    result = runner.invoke(main, ["rekordbox", "cues", "--usb-path", "/nonexistent/pioneer/path"])
    assert result.exit_code == 2
    assert "Error:" in result.output


# ---------------------------------------------------------------------------
# Domain-Specific Assertion Helpers
# ---------------------------------------------------------------------------


def assert_command_succeeded(result: Result) -> None:
    """Assert CLI execution completed with return code 0."""
    assert result.exit_code == 0, f"Command failed (exit {result.exit_code}): {result.output}"


def assert_transcode_detected_exit_code(result: Result) -> None:
    """Assert CLI returned status code 1 indicating a detected low-bitrate transcode."""
    assert result.exit_code == 1, f"Expected transcode exit code 1, got {result.exit_code}: {result.output}"


def assert_transcode_payload_matches(payload: dict, max_cutoff_hz: float) -> None:
    """Assert JSON output correctly identifies suspicious transcode metrics."""
    assert payload["is_suspicious_transcode"] is True
    assert payload["detected_cutoff_hz"] < max_cutoff_hz


def assert_non_empty_file_created(path: Path) -> None:
    """Assert file was created on disk and has non-zero byte size."""
    assert path.exists(), f"Expected file not found: {path}"
    assert path.stat().st_size > 0, f"Generated file is empty: {path}"
