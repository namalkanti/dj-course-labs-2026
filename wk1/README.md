# Week 1 Lab: Audio Quality, Codec Analysis & Transcode Detection

An interactive digital signal processing lab analyzing lossy vs lossless audio compression, psychoacoustic filtering, and automated detection of upsampled/transcoded files.

## Background & Lab Outline

When preparing tracks for high-SPL sound systems, lossy compression artifacts and fake upsampled audio (e.g. 128 kbps MP3 re-encoded to 320 kbps or WAV) can cause transient smearing, muffled high end, and phase issues. This lab examines:
1. **Audio Quality Fundamentals for DJs**: Sample rates (44.1 kHz vs 48 kHz), bit depth dynamics (16-bit vs 24-bit), container formats (WAV vs AIFF vs FLAC vs MP3), psychoacoustic masking, and the mechanics of slicing compressed audio.
2. **Pristine Signal Analysis & Codec Filter Analysis**: Welch Power Spectral Density (PSD), STFT spectrograms, three-band Pioneer-style waveform separation, psychoacoustic cutoff shelves (~16 kHz for 128k, ~18.5 kHz for 192k, ~20+ kHz for 320k/lossless), and automated transcode heuristic classification.
3. **DJ vs DSP Vocabulary & Visual Translation**: Connecting DJ ear observations (transient smear, dull hats, watery cymbals) to physical DSP metrics and Rekordbox display artifacts.

## Launch

Run using `uv` from the workspace root or directory:

```bash
uv run marimo edit wk1/quality_lab.py
```

Or view as a standalone read-only app:

```bash
uv run marimo run wk1/quality_lab.py
```

