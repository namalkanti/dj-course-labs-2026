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


if __name__ == "__main__":
    app.run()
