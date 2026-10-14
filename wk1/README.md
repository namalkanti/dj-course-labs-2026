# Week 1 Lab: Audio Quality, Codec Analysis & Transcode Detection

An interactive digital signal processing lab analyzing lossy vs lossless audio compression, psychoacoustic filtering, and automated detection of upsampled/transcoded files.

## Background

When preparing tracks for high-SPL sound systems, lossy compression artifacts and fake upsampled audio (e.g. 128 kbps MP3 re-encoded to 320 kbps or WAV) can cause transient smearing, muffled high end, and phase issues. This lab examines:
1. **Codec Mechanics**: PCM vs lossless (FLAC) vs perceptual lossy coding (MP3, AAC).
2. **Spectral Cutoffs**: Measuring frequency roll-off shelves across bitrate tiers (~16 kHz for 128k, ~18.5 kHz for 192k, ~20+ kHz for 320k/lossless).
3. **Automated Transcode Detection**: Combining container metadata parsing with STFT high-frequency energy profiling to identify fake 320 kbps rips.

## Requirements & Launch

Run using `uv`:

```bash
uv run --with marimo --with numpy --with scipy --with matplotlib --with soundfile --with mutagen marimo edit quality_lab.py
```
