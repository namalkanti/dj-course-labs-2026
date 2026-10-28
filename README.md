# DJ DSP Labs

Hands-on digital signal processing and music information retrieval (MIR) lab modules for a DJ curriculum.

Modeled after UC Berkeley EE123 lab formats: combining foundational mathematical formulations, from-scratch implementations, standard production library comparisons (`librosa`, `scipy`), and reactive interactive inspection via [marimo](https://marimo.io/).

## Structure

```
labs/
├── README.md
├── primer/                   # Marimo DAG & reactive UI primer
├── quality/                  # Audio quality, codec analysis & transcode detection
│   ├── quality_lab.py        # Marimo interactive notebook
│   └── README.md
├── timing/                   # Timing, tempo, meter & phrase architecture
│   ├── timing_lab.py         # Marimo interactive notebook
│   └── README.md
├── eq/                       # EQ, filterbank, and crossover study guide
│   └── README.md
├── harmonics_transitions/    # Harmonic mixing and transition study guide
│   └── README.md
├── src/dj_dsp/               # Reusable DSP, plotting, synthesis, and catalog modules
│   ├── audio_catalog.py      # Reference fixture management and validation
│   ├── plotting.py           # Matplotlib figures for spectra, spectrograms, and 3-band waveforms
│   ├── spectral.py           # Pure Welch PSD, STFT, cutoff detection, and band separation
│   ├── spectral_data.py      # Immutable dataclass views for spectral data
│   └── synth.py              # Synthetic signal generation, LAME encoding, and filtering
├── audio_catalog/            # Public-domain and CC-licensed audio test fixtures
│   └── README.md
└── tests/                    # Pytest test suite for core library modules
```

## Setup & Environment

All dependencies and Python runtimes are managed reproducibly via [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

To run the test suite:

```bash
uv run pytest
```

## Running Labs

Launch any interactive Marimo lab using `uv run marimo edit`:

```bash
# Audio Quality & Transcode Detection
uv run marimo edit quality/quality_lab.py

# Timing, Tempo & Meter Estimation
uv run marimo edit timing/timing_lab.py

# Primer: Marimo DAG & UI Pattern Guide
uv run marimo edit primer/marimo_primer.py
```

Or run as a read-only app:

```bash
uv run marimo run quality/quality_lab.py
```

## CLI Tools

A Click-powered CLI (`dj-dsp`) is installed for terminal audio quality audits and spectral plotting:

```bash
# Check track audio quality & transcode detection
uv run dj-dsp check-quality path/to/track.flac

# JSON output for pipelines
uv run dj-dsp check-quality -j path/to/track.mp3

# Plot high-frequency shelf (10–22 kHz) with 16–20 kHz hearing limit highlight
uv run dj-dsp plot-shelf path/to/track.mp3
uv run dj-dsp plot-shelf -o shelf.png path/to/track.mp3
```

