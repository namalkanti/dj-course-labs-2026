import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marimo Primer & Pattern Guide

    Interactive reference for reactive DSP notebooks in `labs/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reactive DAG & Dataflow

    In Marimo, execution order is determined by variable dependencies, not visual order.
    If an upstream variable changes, downstream cells re-execute automatically.
    """)
    return


@app.cell
def _():
    import numpy as np

    #Upstream cell: Define base signal parameters
    sampling_rate = 44100
    duration_sec = 0.5
    return duration_sec, np, sampling_rate


@app.cell
def _(duration_sec, mo, np, sampling_rate):
    #Downstream cell: Automatically re-evaluates if duration_sec or sampling_rate change
    total_samples = int(sampling_rate * duration_sec)
    time_vector = np.linspace(0.0, duration_sec, total_samples, endpoint=False)
    mo.md(f"Computed `{total_samples}` samples spanning `{duration_sec}` seconds.")
    return


@app.cell
def _():
    _local_freq = 440.0
    _note = "A4"
    f"Synthesizing {_note} at {_local_freq} Hz"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive UI Elements & State Binding

    Widgets created via `mo.ui.*` act as reactive sources.
    - **Rule**: Instantiate widgets in one cell; access `.value` in downstream cells.
    - Reading `.value` in the cell that created the widget breaks reactivity or creates a cycle.
    """)
    return


@app.cell
def _(mo):
    freq_slider = mo.ui.slider(
        start=20, stop=2000, step=10, value=440, label="Frequency (Hz)"
    )
    amp_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.5, label="Amplitude"
    )
    waveform_dropdown = mo.ui.dropdown(
        options=["sine", "square", "sawtooth"],
        value="sine",
        label="Waveform",
    )
    audio_file = mo.ui.file(
        filetypes=[".wav", ".mp3", ".flac"],
        label="Upload Reference Audio",
    )

    widget_panel = mo.vstack([
        mo.hstack([freq_slider, amp_slider, waveform_dropdown]),
        audio_file,
    ])
    widget_panel
    return amp_slider, audio_file, freq_slider, waveform_dropdown


@app.cell(hide_code=True)
def _(amp_slider, audio_file, freq_slider, mo, waveform_dropdown):
    file_status = (
        f"**File loaded**: `{audio_file.name}` ({len(audio_file.contents)} bytes)"
        if audio_file.value
        else "_No file selected._"
    )

    mo.md(f"""
    ### Live Parameter State

    - **Waveform**: `{waveform_dropdown.value}`
    - **Frequency**: `{freq_slider.value} Hz`
    - **Amplitude**: `{amp_slider.value}`
    - **Reference Audio**: {file_status}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Plotting & Layout Patterns

    Reactive visualization in Marimo requires no explicit `plt.draw()` or canvas callbacks.
    When parameters change, downstream computation generates new arrays, and figure cells re-render.
    """)
    return


@app.cell(hide_code=True)
def _(amp_slider, freq_slider, np, sampling_rate, waveform_dropdown):
    f0 = float(freq_slider.value)
    amp = float(amp_slider.value)
    wtype = waveform_dropdown.value

    # Generate 50 ms chunk for time-domain visualization
    vis_duration = 0.05
    t_vis = np.linspace(
        0.0, vis_duration, int(sampling_rate * vis_duration), endpoint=False
    )
    phase = 2.0 * np.pi * f0 * t_vis

    if wtype == "sine":
        sig_vis = amp * np.sin(phase)
    elif wtype == "square":
        sig_vis = amp * np.sign(np.sin(phase))
    elif wtype == "sawtooth":
        sig_vis = amp * (2.0 * (t_vis * f0 - np.floor(t_vis * f0 + 0.5)))
    else:
        sig_vis = np.zeros_like(t_vis)

    # Compute single-sided magnitude spectrum (FFT with Hann window to reduce leakage)
    n_fft = 2048
    t_fft = np.linspace(0.0, n_fft / sampling_rate, n_fft, endpoint=False)
    phase_fft = 2.0 * np.pi * f0 * t_fft
    if wtype == "sine":
        sig_fft = amp * np.sin(phase_fft)
    elif wtype == "square":
        sig_fft = amp * np.sign(np.sin(phase_fft))
    else:
        sig_fft = amp * (2.0 * (t_fft * f0 - np.floor(t_fft * f0 + 0.5)))

    window = np.hanning(n_fft)
    fft_vals = np.fft.rfft(sig_fft * window)
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    mag_db = 20.0 * np.log10(np.abs(fft_vals) / (n_fft / 2.0) + 1e-9)
    return f0, freq_bins, mag_db, n_fft, sig_fft, sig_vis, t_vis, wtype


@app.cell(hide_code=True)
def _(f0, freq_bins, mag_db, sig_vis, t_vis, wtype):
    import matplotlib.pyplot as plt

    fig, (ax_time, ax_freq) = plt.subplots(
        2, 1, figsize=(9, 5), constrained_layout=True
    )

    # Time-domain waveform
    ax_time.plot(t_vis * 1000.0, sig_vis, color="#1f77b4", lw=1.5)
    ax_time.set_title(f"Time Domain: {wtype.capitalize()} Wave ({f0:.1f} Hz)")
    ax_time.set_xlabel("Time (ms)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_ylim(-1.1, 1.1)
    ax_time.grid(True, alpha=0.3)

    # Frequency-domain magnitude spectrum
    ax_freq.plot(freq_bins, mag_db, color="#ff7f0e", lw=1.2)
    ax_freq.set_title("Magnitude Spectrum (FFT with Hann Window)")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude (dBFS)")
    ax_freq.set_xlim(0, 4000)
    ax_freq.set_ylim(-80, 6)
    ax_freq.grid(True, alpha=0.3)

    fig
    return fig, plt


@app.cell
def _(f0, fig, mo, sig_vis, wtype):
    # Layout orchestration: mo.tabs and mo.accordion for lab sections
    summary_markdown = mo.md(f"""
    **Synthesis Summary:**
    - Type: `{wtype}`
    - Fundamental $f_0$: `{f0:.1f} Hz`
    - Max sample peak: `{abs(sig_vis).max():.4f}`
    """)

    lab_view = mo.ui.tabs({
        "Interactive Plots": fig,
        "Numeric Summary": summary_markdown,
    })

    explanation_accordion = mo.accordion({
        "Spectral Characteristics Reference": mo.md(r"""
        - **Sine**: Single spectral peak at $f_0$.
        - **Square**: Odd harmonics ($f_0, 3f_0, 5f_0, \dots$) with $1/n$ amplitude roll-off.
        - **Sawtooth**: All integer harmonics ($f_0, 2f_0, 3f_0, \dots$) with $1/n$ roll-off.
        """)
    })

    mo.vstack([lab_view, explanation_accordion])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of Windowing: Spectral Leakage

    When analyzing a finite block of samples without explicitly applying a window function, an implicit **rectangular window** (boxcar) is applied.

    Unless the signal frequency is an exact integer multiple of the FFT bin spacing ($$\Delta f = f_s / N$$), truncating the periodic signal causes a sharp discontinuity at the frame boundaries. In the frequency domain:
    - **Rectangular window (No window)**: Exhibits a narrow main lobe, but very high sidelobes (first sidelobe is only $$-13\text{ dB}$$ down). Energy smears across distant frequency bins (**spectral leakage**), masking weak signals and raising the noise floor.
    - **Hann window**: Gently tapers signal boundaries to zero. While slightly widening the main lobe (lowering frequency resolution), sidelobes decay rapidly (first sidelobe at $$-31.5\text{ dB}$$, rolling off at $$-18\text{ dB/octave}$$), suppressing false spectral artifacts.
    """)
    return


@app.cell
def _(f0, freq_bins, mag_db, n_fft, np, plt, sampling_rate, sig_fft):
    # Compute FFT without a window (implicit rectangular window)
    fft_rect = np.fft.rfft(sig_fft)
    mag_db_rect = 20.0 * np.log10(np.abs(fft_rect) / (n_fft / 2.0) + 1e-9)

    fig_leakage, (ax_wide, ax_zoom) = plt.subplots(
        2, 1, figsize=(9, 6), constrained_layout=True
    )

    # Wideband comparison: highlighting spectral leakage floor
    ax_wide.plot(
        freq_bins, mag_db_rect, label="No Window (Rectangular)", color="#d62728", lw=1.2, alpha=0.85
    )
    ax_wide.plot(
        freq_bins, mag_db, label="Hann Window", color="#1f77b4", lw=1.4
    )
    ax_wide.set_title("Wideband Comparison: Spectral Leakage Noise Floor", fontsize=11, fontweight="bold")
    ax_wide.set_xlabel("Frequency (Hz)")
    ax_wide.set_ylabel("Magnitude (dBFS)")
    ax_wide.set_xlim(0, 4000)
    ax_wide.set_ylim(-90, 6)
    ax_wide.grid(True, alpha=0.3)
    ax_wide.legend(loc="upper right")

    # Zoomed comparison: close-up around the fundamental frequency
    f_min = max(0.0, f0 - 350.0)
    f_max = min(sampling_rate / 2.0, f0 + 350.0)

    ax_zoom.plot(
        freq_bins, mag_db_rect, label="No Window (High Sidelobes)", color="#d62728", lw=1.3, marker=".", markersize=4
    )
    ax_zoom.plot(
        freq_bins, mag_db, label="Hann Window (Suppressed Sidelobes)", color="#1f77b4", lw=1.4, marker=".", markersize=4
    )
    ax_zoom.set_title(
        f"Zoomed View around f₀ = {f0:.1f} Hz: Sidelobe Roll-off vs. Main Lobe Width",
        fontsize=11,
        fontweight="bold",
    )
    ax_zoom.set_xlabel("Frequency (Hz)")
    ax_zoom.set_ylabel("Magnitude (dBFS)")
    ax_zoom.set_xlim(f_min, f_max)
    ax_zoom.set_ylim(-70, 6)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc="upper right")

    fig_leakage
    return


if __name__ == "__main__":
    app.run()
