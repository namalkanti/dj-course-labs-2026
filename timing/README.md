# Timing Lab: Tempo, Meter & Phrase Architecture

An interactive digital signal processing lab exploring the DJ metric hierarchy—beats, bars, phrases, and meters—using spectral flux onset envelopes, autocorrelation tempo estimation, and time signature discrimination.

## Background & Lab Outline

When preparing DJ sets, commercial software (e.g. Rekordbox) can stumble on tracks with ambiguous timing, floating intros, non-4/4 meters, or dense harmonic layering. This lab breaks down classical Music Information Retrieval (MIR) pipelines to understand:
1. **DJ Timing Fundamentals & The Metric Hierarchy**: Connecting the DJ timing hierarchy (quarter-note beats, 4-beat bars, 16/32-beat phrases) to physical DSP signal representations.
2. **Transient Extraction via Spectral Flux**: Converting raw audio into half-wave rectified STFT onset strength envelopes.
3. **Beat & Tempo Extraction via Autocorrelation**: Applying the Wiener–Khinchin theorem via FFT-based autocorrelation to isolate primary beat intervals ($T_{\text{beat}}$) and handle octave/subharmonic BPM ambiguities.
4. **Bar & Meter Discrimination ($4/4$ vs $3/4$ / $6/8$)**: Evaluating autocorrelation lag ratios ($3\times$ vs $4\times$ beat interval) to classify duple/quadruple vs triple/compound time signatures.
5. **Phrase Architecture & Visualization**: Extracting long-lag periodicities (16 and 32 beats) to identify drops, breakdowns, and phrase boundaries.
6. **Rekordbox Failure Modes & Verification**: Analyzing why DJ software beatgrids drift, and comparing from-scratch NumPy implementations against production `librosa` APIs.

## Launch

Run using `uv` from the workspace root or directory:

```bash
uv run marimo edit timing/timing_lab.py
```

Or view as a standalone read-only app:

```bash
uv run marimo run timing/timing_lab.py
```
