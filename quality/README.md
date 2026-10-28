# Audio Quality Lab: Codec Analysis & Transcode Detection

An interactive digital signal processing lab analyzing lossy vs lossless audio compression, psychoacoustic filtering, and automated detection of upsampled/transcoded files.

## Background & Lab Outline

When preparing tracks for high-SPL sound systems, lossy compression artifacts and fake upsampled audio (e.g. 128 kbps MP3 re-encoded to 320 kbps or WAV) can cause transient smearing, muffled high end, and phase issues. This lab examines:
1. **Audio Quality Fundamentals for DJs**: Sample rates (44.1 kHz vs 48 kHz), bit depth dynamics (16-bit vs 24-bit), container formats (WAV vs AIFF vs FLAC vs MP3), psychoacoustic masking, and the mechanics of slicing compressed audio.
2. **Pristine Signal Analysis & Codec Filter Analysis**: Welch Power Spectral Density (PSD), STFT spectrograms, three-band Pioneer-style waveform separation, psychoacoustic cutoff shelves (~16 kHz for 128k, ~18.5 kHz for 192k, ~20+ kHz for 320k/lossless), and automated transcode heuristic classification.
3. **DJ vs DSP Vocabulary & Visual Translation**: Connecting DJ ear observations (transient smear, dull hats, watery cymbals) to physical DSP metrics and Rekordbox display artifacts.

## Launch

Run using `uv` from the workspace root or directory:

```bash
uv run marimo edit quality/quality_lab.py
```

Or view as a standalone read-only app:

```bash
uv run marimo run quality/quality_lab.py
```

## CLI Quality Inspection & Shelf Plotting

The `dj-dsp` CLI provides standalone terminal-based transcode detection and high-frequency shelf visualization without launching Marimo:

```bash
# Formatted quality audit with pass/fail transcode exit codes (0 = pass, 1 = transcode)
uv run dj-dsp check-quality audio_catalog/persistence_by_dabmasterars_snippet.flac

# JSON output for automated scripting or batch pipeline inspection
uv run dj-dsp check-quality -j audio_catalog/persistence_by_dabmasterars_snippet.flac

# Analyze only the first 15 seconds of a track
uv run dj-dsp check-quality -d 15.0 path/to/track.mp3

# Plot high-frequency shelf (10–22 kHz) with 16–20 kHz hearing limit highlight
# Opens an interactive GUI window, or falls back to system viewer
uv run dj-dsp plot-shelf path/to/track.mp3

# Save shelf plot directly to image without GUI interaction
uv run dj-dsp plot-shelf -o shelf.png path/to/track.mp3
```

