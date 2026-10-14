# YourShot DSP Labs

Hands-on digital signal processing and music information retrieval (MIR) lab modules complementing the YourShot DJ curriculum.

Modeled after UC Berkeley EE123 lab formats: combining foundational mathematical formulations, from-scratch implementations, standard production library comparisons (`librosa`), and reactive interactive inspection via [marimo](https://marimo.io/).

## Structure

```
labs/
├── README.md
├── primer/                   # Marimo DAG & UI primer
├── wk1/                      # Week 1: Audio quality, codec analysis & transcode detection
│   ├── quality_lab.py        # Marimo interactive notebook
│   └── README.md
└── wk2/                      # Week 2: Timing, tempo & meter estimation
    ├── timing_lab.py         # Marimo interactive notebook
    └── README.md
```

## Running Labs

All labs manage dependencies reproducibly via [`uv`](https://docs.astral.sh/uv/).

To launch an interactive Marimo lab (e.g. Week 1):

```bash
cd wk1
uv run --with marimo --with numpy --with scipy --with matplotlib --with soundfile --with mutagen marimo edit quality_lab.py
```

Or view as an app:

```bash
cd wk1
uv run --with marimo --with numpy --with scipy --with matplotlib --with soundfile --with mutagen marimo run quality_lab.py
```
