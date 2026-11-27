import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def init_marimo():
    import io
    import inspect
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from dj_dsp.synth import generate_synthetic_beat
    from dj_dsp.timing_librosa import LIBROSA_BACKEND
    from dj_dsp.timing_naive import NAIVE_BACKEND

    def fig_to_image(fig, dpi=100):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return mo.image(buf.read(), rounded=True)

    return (
        LIBROSA_BACKEND,
        NAIVE_BACKEND,
        fig_to_image,
        generate_synthetic_beat,
        inspect,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def render_header(mo):
    mo.md(r"""
    # Timing Lab: Tempo, Meter & Phrase Architecture

    *An interactive digital signal processing lab exploring the DJ timing hierarchy—Beats, Bars, Phrases, and Time Signatures—using Spectral Flux, Envelope Autocorrelation, and Tempograms.*
    """)
    return


@app.cell(hide_code=True)
def render_part1_overview(mo):
    mo.md(r"""
    ## Part 1: DJ Timing Fundamentals & The Metric Hierarchy

    In DJing, track structure is organized into a strict hierarchical pyramid:
    1. **Beats** (Quarter notes): The atomic grid pulses (e.g. 128 BPM = 128 beats/min $\approx 469$ ms per beat).
    2. **Bars** (Measures): Groupings of beats. In standard electronic music, 1 Bar = 4 Beats (4/4 time signature).
    3. **Phrases**: Musical blocks of **4 bars (16 beats)** or **8 bars (32 beats)**. Drops, bassline entries, vocal intros, and breakdowns strictly snap to phrase boundaries.
    4. **Meter / Time Signature**: Simple duple/quadruple ($4/4$) vs triple/compound ($3/4, 6/8$). Mixing a 3/4 track into a 4/4 set causes immediate rhythmic dissonance because phrases do not align.

    ---

    ### The DJ-to-DSP Timing Translation Table

    | DJ Concept | Physical Musical Meaning | DSP / MIR Mathematical Implementation |
    | :--- | :--- | :--- |
    | **Transient / Attack / "The Hit"** | The percussive attack of a kick, snare, or hi-hat (ignoring decaying/sustained sound) | Positive jump in spectral energy (**Spectral Flux**) with half-wave rectification |
    | **Track BPM** | Frequency of recurring quarter notes (the pulse you tap your foot to) | Primary lag peak $\tau_{\text{beat}}$ in autocorrelation $R(\tau)$ |
    | **Beatgrid Phase ("The ONE")**| Alignment of the initial downbeat where the bar begins | Offset $\phi_0$ maximizing cross-correlation with beat pulse train |
    | **Halftime / Doubletime Glitch**| Rekordbox detecting 70 BPM instead of 140 BPM | Subharmonic / octave ambiguity peak selection ($2\tau_0$ vs $\tau_0$) |
    | **Meter ($4/4$ vs $3/4$)** | Downbeat every 4 beats (house/techno) vs every 3 beats (waltz) | Energy ratio of autocorrelation peaks: $R(4\tau_0) / R(3\tau_0)$ |
    | **Syncopation / "Groove" Offbeats**| Accents on weak subdivisions (e.g. 16th notes, the "and" of 2) while leaving main beats empty | High spectral flux on sub-intervals; resolved via multi-bar autocorrelation |
    | **Phrasing (16/32-Beat Blocks)**| Major section changes, drop transitions, structural energy shifts | Long-lag envelope modulation ($16\tau_0, 32\tau_0$) or novelty curves |
    """)
    return


@app.cell(hide_code=True)
def render_part2_theory(mo):
    mo.md(r"""
    ## Part 2: The Hit (Transient Extraction via Spectral Flux)

    To locate rhythmic pulses, we must convert raw continuous waveform oscillations into a positive **onset strength envelope**.

    Simple time-domain envelope followers fail on club music because sustained sub-bass notes carry huge energy but zero rhythmic attack. Instead, we use **Spectral Flux**: computing the Short-Time Fourier Transform (STFT), taking first-order temporal differences across frequency bins, and half-wave rectifying (keeping only positive energy surges):

    $$SF(n) = \sum_{k=0}^{K-1} \max\left(0, |X(n, k)| - |X(n-1, k)|\right)$$

    Where:
    - $X(n, k)$ is the complex STFT frame $n$ at frequency bin $k$.
    - $|X(n, k)|$ is the magnitude spectrum.
    - $\max(0, \Delta)$ ensures that decaying sounds (the sustain and release of a cymbal) do not produce spurious negative onsets.
    """)
    return


@app.cell
def part2_audio_fixture(generate_synthetic_beat):
    # 4 bars of 128 BPM house beat (kick on beats 1-4, snare on 2 & 4, hi-hats on offbeats)
    fs_fixture = 44100
    bpm_fixture = 128.0
    audio_fixture = generate_synthetic_beat(
        bpm=bpm_fixture, duration_sec=4.0, meter="4/4", fs=fs_fixture
    )
    time_fixture = np.arange(len(audio_fixture)) / fs_fixture
    return audio_fixture, bpm_fixture, fs_fixture, time_fixture


@app.cell
def part2_compute_flux(LIBROSA_BACKEND, NAIVE_BACKEND, audio_fixture, fs_fixture):
    naive_env = NAIVE_BACKEND.compute_onset_envelope(audio_fixture, fs=fs_fixture)
    librosa_env = LIBROSA_BACKEND.compute_onset_envelope(audio_fixture, fs=fs_fixture)
    return librosa_env, naive_env


@app.cell(hide_code=True)
def render_part2_plot(
    audio_fixture,
    fig_to_image,
    librosa_env,
    mo,
    naive_env,
    np,
    plt,
    time_fixture,
):
    fig, (ax_wave, ax_flux) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # Top: Raw audio waveform
    ax_wave.plot(time_fixture, audio_fixture, color="#4a5568", alpha=0.7, lw=0.8)
    ax_wave.set_ylabel("Amplitude", fontsize=10)
    ax_wave.set_title("Raw Audio Waveform (128 BPM House Kick + Snare + Hi-Hat)", fontsize=11, fontweight="bold")
    ax_wave.grid(True, linestyle="--", alpha=0.4)

    # Bottom: Normalized Onset Strength Envelopes (Spectral Flux)
    norm_naive = naive_env.values / (np.max(naive_env.values) + 1e-9)
    norm_librosa = librosa_env.values / (np.max(librosa_env.values) + 1e-9)

    ax_flux.plot(naive_env.times, norm_naive, label="NumPy Naive (Log-Spectral Flux)", color="#3182ce", lw=1.5)
    ax_flux.plot(
        librosa_env.times,
        norm_librosa,
        label="Production (librosa.onset.onset_strength)",
        color="#e53e3e",
        linestyle="--",
        lw=1.2,
        alpha=0.85,
    )
    ax_flux.set_xlabel("Time (seconds)", fontsize=10)
    ax_flux.set_ylabel("Onset Strength", fontsize=10)
    ax_flux.set_title("Detected Attacks: Sharp Spikes on Hits, Zero on Sustains/Decays", fontsize=11, fontweight="bold")
    ax_flux.legend(loc="upper right", frameon=True, fontsize=9)
    ax_flux.grid(True, linestyle="--", alpha=0.4)

    plot_img = fig_to_image(fig)
    plot_img
    return


@app.cell(hide_code=True)
def render_part2_inspection(NAIVE_BACKEND, inspect, mo):
    code_str = inspect.getsource(NAIVE_BACKEND.compute_onset_envelope)
    acc = mo.accordion(
        {
            "💡 DJ Takeaway: Why simple volume meters fail on club tracks": mo.md(r"""
            - **The Bass Trap**: A sustained sub-bass note (808 or synth reese) has huge amplitude, but our ears only tap to the *attack* (the instantaneous hit).
            - **Spectral Flux Magic**: By taking differences in frequency bins frame-by-frame and keeping only *positive* changes ($\max(0, \Delta)$), the sustained body of the bass drops to zero while the sharp kick transient spikes violently.
            - **Log Compression**: Without $\log(1 + 100 \cdot |X|)$, kick drums would overpower snares and hats by 100:1, blinding subsequent tempo analysis to everything except the kick.
            """),
            "🔬 View Source: `compute_onset_envelope` (from `dj_dsp.timing_naive`)": mo.md(
                f"```python\n{code_str}\n```"
            ),
        }
    )
    acc
    return


@app.cell(hide_code=True)
def render_part3_theory(mo):
    mo.md(r"""
    ## Part 3: Beat & Tempo Extraction via Autocorrelation

    Once we have an onset envelope $e(n)$, the inter-beat interval $T_{\text{beat}}$ appears as strong periodic self-similarity.
    We compute the **unbiased linear autocorrelation** of the onset envelope:

    $$R_{ee}(\tau) = \sum_{n} e(n) \, e(n + \tau)$$

    Using the **Wiener–Khinchin theorem**, we calculate this in $O(N \log N)$ via the Fast Fourier Transform:

    $$R_{ee} = \mathcal{F}^{-1}\left( |\mathcal{F}(e)|^2 \right)$$

    Each lag $\tau$ in frames translates directly to candidate tempo in Beats Per Minute (BPM):

    $$\text{BPM}(\tau) = \frac{60 \times f_s}{H \times \tau}$$

    Where $f_s$ is the audio sampling rate and $H$ is the STFT hop length.

    ---

    ### Why Autocorrelation Handles Syncopation
    In funk, breakbeats, or garage, drums frequently play **syncopated hits**—accents on unexpected subdivisions (such as an 8th-note offbeat or a 16th-note anticipation) while leaving the expected onbeat empty.

    If an algorithm only inspects individual bars locally with a simple peak detector, the loudest spike might land on an offbeat, causing a phase error (misplaced downbeat) or octave confusion.

    Because autocorrelation integrates products over an extended multi-bar window ($4$ to $8$ bars):
    - Syncopated offbeats occur irregularly or at partial submultiples, so their cross-products remain diffuse.
    - The true metric grid ($T_{\text{beat}}$ and $4T_{\text{beat}}$) consistently reinforces itself at every measure through **constructive interference**, emerging as the dominant global lag peak.
    """)
    return


@app.cell
def part3_tempo_autocorr_scratch():
    def autocorr_scratch(x):
        """Compute 1D autocorrelation using FFT/IFFT Wiener-Khinchin theorem."""
        # TODO(Step 3): Implement scratch FFT-based autocorrelation
        pass

    def estimate_tempo_scratch(
        flux,
        sr: int = 44100,
        hop_length: int = 512,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
    ):
        """Locate dominant lag in autocorrelation and convert to BPM."""
        # TODO(Step 3): Implement peak picker within [min_bpm, max_bpm]
        pass

    return


@app.cell(hide_code=True)
def render_part4_theory(mo):
    mo.md(r"""
    ## Part 4: Bar & Meter Discrimination ($4/4$ vs $3/4$ / $6/8$)

    In DJ booth practice, mixing a 3/4 waltz or a 6/8 polyrhythm into standard 4/4 house music can cause clashing phrase drops.

    Once the primary beat interval $\tau_0$ is known:
    - In **4/4 meter**, musical accents fall on every 4th beat (the downbeat). Therefore, autocorrelation exhibits high energy at integer multiples: $\tau = 4\tau_0, 8\tau_0, \dots$
    - In **3/4 or 6/8 meter**, accents fall every 3rd beat: $\tau = 3\tau_0, 6\tau_0, \dots$

    We construct the **Meter Prominence Ratio**:

    $$\rho = \frac{R(4\tau_0)}{R(3\tau_0)}$$

    - If $\rho > 1 + \delta \implies$ Standard $4/4$ time signature.
    - If $\rho < 1 - \delta \implies$ Triple meter ($3/4$ or $6/8$).
    """)
    return


@app.function
def detect_meter_scratch(autocorr, beat_interval_frames: int, tolerance: float = 0.15):
    """Classify 4/4 vs 3/4 meter by sampling autocorrelation at 3x and 4x beat intervals."""
    # TODO(Step 3): Sample lags and compute ratio
    pass


@app.cell(hide_code=True)
def render_part5_phrase_theory(mo):
    mo.md(r"""
    ## Part 5: Phrase Architecture & Interactive Visualizer

    A standard dance track transitions every 16 or 32 beats (4 or 8 bars). Here we wire interactive controls to inspect any audio clip, view the synchronized waveform, onset envelope, autocorrelation harmonics ($1\times, 2\times, 3\times, 4\times, 16\times$), and tempogram stability.
    """)
    return


@app.cell
def part5_ui_controls(mo):
    track_select = mo.ui.dropdown(
        options={
            "synthetic_house_128": "Synthetic 4/4 House (128 BPM)",
            "synthetic_waltz_120": "Synthetic 3/4 Waltz (120 BPM)",
            "synthetic_breakbeat_140": "Synthetic Syncopated Breakbeat (140 BPM)",
            "synthetic_drift_drum": "Synthetic Drifting Tempo (Live Drummer)",
        },
        value="synthetic_house_128",
        label="Select Test Signal",
    )

    window_slider = mo.ui.slider(
        start=5.0,
        stop=30.0,
        step=5.0,
        value=15.0,
        label="Analysis Duration (seconds)",
    )

    tempo_range_slider = mo.ui.range_slider(
        start=50,
        stop=220,
        step=5,
        value=[70, 180],
        label="BPM Search Range",
    )
    return


@app.cell(hide_code=True)
def render_part6_rekordbox_analysis(mo):
    mo.md(r"""
    ## Part 6: Rekordbox Discrepancies, Failure Modes & DJ Peer Card

    ### Why Rekordbox Beatgrids Fail (and How to Fix Them)
    1. **Octave Confusion (Halftime / Doubletime)**: When a drum-and-bass track (174 BPM) has a prominent half-time snare on beat 3, Rekordbox locks onto the 87 BPM subharmonic.
    2. **Syncopation / Polyrhythmic Wash**: Ambient pads and syncopated melodies mask the percussive transient in the spectral flux, shifting the downbeat.
    3. **How Modern AI Fixes This**: Stem separation (BS-RoFormer / Demucs) isolates the pure drum stem before computing spectral flux, stripping out vocals and synths that cause false onsets.

    ---

    ### DJ Peer Cheat Sheet: "How to Explain Beatgrid Glitches to Your DJ Friends"
    > *"Your CDJ isn't broken—Rekordbox just picked the 2nd harmonic of the kick drum pattern instead of the fundamental quarter note. Hit 2x / 1/2x in export mode to fix the octave, or tap the downbeat marker on the ONE!"*
    """)
    return


if __name__ == "__main__":
    app.run()
