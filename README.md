# YourShot DSP Labs

Hands-on digital signal processing and music information retrieval (MIR) lab modules complementing the YourShot DJ curriculum.

Modeled after UC Berkeley EE123 lab formats: combining foundational mathematical formulations, from-scratch implementations, standard production library comparisons (`librosa`), and reactive interactive inspection via [marimo](https://marimo.io/).

## Structure

```
labs/
├── README.md
└── wk1/                      # Week 1: Onset detection, BPM estimation & meter analysis
    ├── timing_lab.py         # Marimo interactive notebook
    └── README.md
```

## Running Labs

All labs manage dependencies reproducibly via [`uv`](https://docs.astral.sh/uv/).

To launch an interactive Marimo lab:

```bash
cd wk1
uv run --with marimo --with librosa --with matplotlib --with soundfile marimo edit timing_lab.py
```

Or view as an app:

```bash
cd wk1
uv run --with marimo --with librosa --with matplotlib --with soundfile marimo run timing_lab.py
```
