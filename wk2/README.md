# Week 2 Lab: Timing, Tempo & Meter Estimation

An interactive digital signal processing lab exploring spectral flux onset envelopes, autocorrelation tempo estimation, and time signature discrimination.

## Background

When preparing DJ sets, commercial software (e.g. Rekordbox) can stumble on tracks with ambiguous timing, floating intros, non-4/4 meters, or dense harmonic layering. This lab breaks down classical Music Information Retrieval (MIR) pipelines to understand:
1. **Spectral Flux**: Extracting onset strength via half-wave rectified STFT differences.
2. **Tempo Extraction**: Autocorrelation of onset envelopes to estimate primary inter-beat intervals ($T_{\text{beat}}$).
3. **Meter Discrimination**: Evaluating autocorrelation lag ratios ($3\times$ vs $4\times$ beat interval) to classify duple/quadruple ($4/4$) vs triple/compound ($3/4, 6/8$) time signatures.
4. **Library Parity**: Comparing from-scratch NumPy implementations against standard `librosa` APIs.

## Requirements & Launch

Run using `uv`:

```bash
uv run --with marimo --with librosa --with matplotlib --with soundfile marimo edit timing_lab.py
```
