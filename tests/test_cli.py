"""Unit tests verifying CLI command behavior, status exit codes, and output formats."""

from pathlib import Path
import json
from click.testing import CliRunner, Result
import numpy as np
import soundfile as sf

from dj_dsp.cli import main
from dj_dsp.synth import apply_lossy_cutoff, generate_drum_loop

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
