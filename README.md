# YourShot DSP Labs

Hands-on digital signal processing and music information retrieval (MIR) lab modules complementing the YourShot DJ curriculum.

Modeled after UC Berkeley EE123 lab formats: combining foundational mathematical formulations, from-scratch implementations, standard production library comparisons (`librosa`, `scipy`), and reactive interactive inspection via [marimo](https://marimo.io/).

## Structure

```
labs/
├── README.md
├── primer/                   # Marimo DAG & reactive UI primer
├── wk1/                      # Week 1: Audio quality, codec analysis & transcode detection
│   ├── quality_lab.py        # Marimo interactive notebook
│   └── README.md
├── wk2/                      # Week 2: Timing, tempo, meter & phrase architecture
│   ├── timing_lab.py         # Marimo interactive notebook
│   └── README.md
├── src/yourshot/             # Reusable DSP, plotting, synthesis, and catalog modules
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
# Week 1: Audio Quality & Transcode Detection
uv run marimo edit wk1/quality_lab.py

# Week 2: Timing, Tempo & Meter Estimation
uv run marimo edit wk2/timing_lab.py

# Primer: Marimo DAG & UI Pattern Guide
uv run marimo edit primer/marimo_primer.py
```

Or run as a read-only app:

```bash
uv run marimo run wk1/quality_lab.py
```

