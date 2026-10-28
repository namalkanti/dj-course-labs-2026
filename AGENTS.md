# Project Guidelines — DJ Audio DSP Labs (`dj_dsp`)

## Repository Scope & Architecture
- Standalone Python project for digital signal processing (DSP) and music information retrieval (MIR) applied to DJ techniques.
- Modeled after UC Berkeley EE123 lab format: mathematical formulations, from-scratch NumPy/SciPy algorithms, standard MIR comparisons (`librosa`), and reactive interactive inspection via [Marimo](https://marimo.io/).
- Package: `dj_dsp` under `src/dj_dsp/`.
- Lab directories:
  - `quality/` — Audio quality, codec analysis & transcode detection (`quality_lab.py`).
  - `timing/` — Tempo, meter & phrase architecture (`timing_lab.py`).
  - `eq/` — Crossovers, filterbanks, and 3-band isolators.
  - `harmonics_transitions/` — Chromagrams, key classification, and transition gain/crossfader curves.

## Coding & DSP Principles
- **Pure functional DSP core**: Core analysis functions in `src/dj_dsp/spectral.py` and `synth.py` must remain pure functions taking arrays/parameters and returning frozen dataclasses defined in `spectral_data.py`.
- **Synthetic vs. Real Audio**:
  - Never commit proprietary copyrighted tracks to Git (e.g. Amazon MP3 downloads or personal library rips).
  - Use pure mathematical synthesizers (`synth.py`) or CC0 / public-domain fixtures in `audio_catalog/`.
  - For tests modeling natural rolloff or codec behavior, use shaped noise (e.g. pink noise / 1st-order Butterworth lowpass + brickwall cutoff) to keep tests fast, fully reproducible, and free of copyright entanglements.
- **Binary Forensic Thresholding**:
  - Cutoff detection is binary: full-bandwidth high-quality audio ($\ge 19.5\text{ kHz}$) vs. low-bitrate transcode ($< 19.5\text{ kHz}$).
  - A threshold drop of $22\text{ dB}$ below the $11\text{–}14\text{ kHz}$ passband reference cleanly separates natural acoustic rolloff ($10\text{–}15\text{ dB}$ across $14\text{–}20\text{ kHz}$) from digital brickwall codec filters ($>20\text{ dB}$ into the noise floor).

## Verification Commands
Always run verification using `uv`:
```bash
# Full unit test suite (25 tests)
uv run pytest

# Check marimo syntax and DAG parsing
uv run python -m py_compile quality/quality_lab.py timing/timing_lab.py

# Launch interactive labs
uv run marimo edit quality/quality_lab.py
uv run marimo edit timing/timing_lab.py
```
