import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def init_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def render_header(mo):
    mo.md(r"""
    # Week 1 Lab: Audio Quality Verification, Codec Analysis & Transcode Detection

    *An interactive digital signal processing lab analyzing lossy vs. lossless audio compression, psychoacoustic filtering, and automated detection of upsampled/transcoded files.*
    """)
    return


@app.cell(hide_code=True)
def render_foundations(mo):
    mo.md(r"""
    ## Part 1: Audio Quality Fundamentals for DJs

    Audio fidelity matters the moment your music hits a real club sound system. Playing low-bitrate or ripped tracks through massive club speakers reveals harsh artifacts, muffled highs, and flabby kicks that you might not notice on AirPods.

    A pervasive problem in DJing is the **fake transcode** (taking a free 128 kbps MP3 or a YouTube rip and re-saving it as a "320 kbps MP3" or "WAV"). While Rekordbox and CDJ screens will show a high bitrate or lossless format, the underlying audio is permanently compromised.

    ---

    ### 1.1 Sample Rate: 44.1 kHz vs 48 kHz
    - **44.1 kHz** is the standard for music releases (CDs, Beatport, Bandcamp, Spotify). It dates back to early digital audio being stored on video tape cassettes.
    - **48 kHz** is the standard for film, video, and YouTube.
    - **DJ Takeaway**: Both sound identical to human ears. Modern CDJs and Rekordbox play both natively. However, **44.1 kHz / 16-bit** is the universal safety standard that plays reliably on *any* CDJ ever made, even legacy models with older firmware.

    ---

    ### 1.2 Bit Depth: 16-bit vs 24-bit
    - **16-bit is all you need for DJing**: 16-bit provides 96 dB of clean volume range. In a loud club where the crowd and room noise is already 75–80 dB, the background noise floor of a 16-bit track is completely inaudible.
    - **24-bit is for recording studios**: Music producers record in 24-bit because they need a large safety cushion (headroom) so loud vocals or drum hits don't accidentally clip and distort while recording.
    - **DJ Takeaway**: Buying or converting your club tracks to 24-bit does **not** improve sound quality. It just adds trailing empty zeros, making files 50% larger for zero audible gain. 16-bit / 44.1 kHz is standard club master quality.

    ---

    ### 1.3 File Formats: WAV vs AIFF vs FLAC
    - **WAV**: Uncompressed audio. Sounds pristine, but has poor metadata support. Cue points, artwork, and artist tags often get stripped or lost when moving between laptops and USB drives.
    - **AIFF**: Uncompressed audio, bit-for-bit identical in quality to WAV. Supports full ID3 metadata and album art. **Preferred by touring DJs** because tags never get lost on CDJs.
    - **FLAC**: Lossless compression (like a ZIP file for audio). Identical sound quality to WAV/AIFF at roughly half the file size. Plays on modern club gear (CDJ-2000NXS2, CDJ-3000), but legacy CDJs cannot read it.

    ---

    ### 1.4 How MP3 Compression Works (And Why Low Bitrates Sound Bad)
    Uncompressed audio takes up ~10 MB per minute. To make smaller files, MP3 uses **psychoacoustics**: discarding sounds the human ear is less likely to notice (like quiet frequencies playing at the same time as a loud kick drum).

    - **320 kbps (High Quality)**: The encoder has plenty of data budget. It only trims frequencies that are genuinely masked by your ear. Full spectrum up to 20+ kHz is preserved. Sounds virtually indistinguishable from WAV in a blind test.
    - **128 kbps (Low Quality / YouTube Rips)**: The encoder runs out of data space and makes aggressive cuts:
      - It applies a hard **brick-wall cutoff at 16 kHz**, permanently deleting all high-frequency sparkle (cymbals, snare sizzle, reverb tails).
      - It heavily distorts transients, turning tight drum hits into mush.
    - **Why 192 kbps still isn't club-ready**:
      - Even though 192 kbps reaches ~18 kHz, the encoder squashes high frequencies into **mono** (Intensity Stereo) to save space, collapsing the 3D stereo width on wide club sound systems.
      - The steep digital filter also smears the timing of high-hats and snares, making the mix sound flat, fatiguing, and harsh at high club volumes.
    """)
    return


@app.cell(hide_code=True)
def render_translation_table(mo):
    mo.md(r"""
    ### 1.5 DJ-to-DSP Translation Table

    | DJ / Club Jargon | What the DJ Hears | The Physical DSP Phenomenon |
    |---|---|---|
    | **"Muffled / Boxy / No Air"** | Cymbals, snares, and vocals lack sparkle and space. | **Brick-wall lowpass cutoff** ($f_c \le 16\text{ kHz}$) applied by 128 kbps encoders. |
    | **"Mushy Kick / Blurry Transients"** | Kick drums lose punch/snap; basslines feel flabby. | **MDCT temporal pre-echo**: Quantization noise spreads backward ~20 ms before the transient inside long MDCT blocks. |
    | **"Watery / Swishy / Phasey Cymbals"** | Hi-hats sound like they are playing through a phaser or underwater. | **Intensity Stereo + High-Band Bit Starvation**: High subbands are coarsely quantized or blended to mono, causing erratic phase variance frame-to-frame. |
    | **"Fake 320 / YouTube Rip"** | Track reports high bitrate or WAV, but sounds lifeless compared to other tracks in the mix. | **Inflated Container Header**: File was transcoded from low-bitrate lossy, leaving zero energy above 16 kHz despite the 320k/WAV header. |
    | **"Falls Apart at High Volume"** | Sounds fine on laptop/headphones, but harsh and fatiguing on the club PA. | **Psychoacoustic Masking Breakdown**: Masking models assume domestic listening (~70 dB SPL). At 100+ dB SPL club levels, hearing contours flatten and hidden quantization noise becomes glaringly audible. |
    """)
    return


@app.cell
def load_dsp_modules():
    import io
    import mutagen
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as signal
    import soundfile as sf
    from synth import (
        apply_lossy_cutoff,
        generate_chirp,
        generate_drum_loop,
        generate_white_noise,
    )

    return (plt,)


@app.cell(hide_code=True)
def render_part2_intro(mo):
    mo.md(r"""
    ## Part 2: Interactive Synthetic Ground Truth Lab (Homework Exercises)

    In this section, you will inspect synthetic test signals (white noise, log chirps, synthesized drum grooves) and simulate the frequency cutoffs imposed by lossy encoders.

    Before building an automated cutoff detector in Step 4, complete the two exercises below to:
    1. Synthesize and filter the chosen test signal based on UI controls.
    2. Compute its spectrogram and power spectrum to visually spot the lossy cutoff shelf.
    """)
    return


@app.cell
def build_controls(mo):
    signal_type = mo.ui.dropdown(
        options={
            "Drum Groove (128 BPM)": "drums",
            "Exponential Chirp (20 Hz - 22 kHz)": "chirp",
            "White Noise (Wideband)": "noise",
        },
        value="Drum Groove (128 BPM)",
        label="Test Signal",
    )

    codec_cutoff = mo.ui.dropdown(
        options={
            "Lossless / Passthrough (22.05 kHz Nyquist)": 22050.0,
            "~320 kbps MP3 (20.0 kHz shelf)": 20000.0,
            "~192 kbps MP3 (18.5 kHz cutoff)": 18500.0,
            "~128 kbps MP3 (16.0 kHz brick-wall)": 16000.0,
        },
        value="Lossless / Passthrough (22.05 kHz Nyquist)",
        label="Simulated Codec Cutoff",
    )

    mo.hstack([signal_type, codec_cutoff], justify="start", gap=2)
    return


@app.cell
def exercise_1_generate_and_filter(mo):
    # =========================================================================
    # HOMEWORK EXERCISE 1: Generate & Filter Signal
    # =========================================================================
    # Goal:
    # 1. Inspect `signal_type.value` ("drums", "chirp", or "noise") and call
    #    the matching generator from `wk1/synth.py`. Use duration_sec=3.0, fs=44100.
    # 2. Filter the resulting audio array using `apply_lossy_cutoff(...)` with
    #    `cutoff_hz=codec_cutoff.value`.
    # 3. Encode the filtered audio into a WAV in-memory buffer (io.BytesIO) via
    #    `sf.write(buf, filtered_audio, fs, format="WAV")` and create an audio player
    #    using `mo.audio(buf.getvalue())`.
    #
    # Hints:
    # - `signal_type.value` gives the selected key ("drums", "chirp", "noise").
    # - `apply_lossy_cutoff(audio, cutoff_hz=..., fs=44100)` automatically bypasses
    #   if cutoff_hz >= fs / 2 (Nyquist).
    # - Don't forget `buf.seek(0)` before passing `buf.getvalue()` to `mo.audio`.
    # =========================================================================

    fs = 44100
    duration_sec = 3.0

    # --- TODO: YOUR CODE HERE ---
    # raw_audio = ...
    # filtered_audio = ...
    # player = ...

    raw_audio = None
    filtered_audio = None
    player = mo.md("*Complete Exercise 1 to listen to the audio output.*")
    # -----------------------------

    player
    return (filtered_audio,)


@app.cell
def exercise_2_spectral_inspector(filtered_audio, mo, plt):
    # =========================================================================
    # HOMEWORK EXERCISE 2: Spectrogram & High-Frequency Power Roll-off
    # =========================================================================
    # Goal:
    # Plot two synchronized subplots:
    #   - Left / Top: STFT Spectrogram (frequency vs. time) showing energy up to Nyquist (22.05 kHz).
    #     Draw a horizontal dashed red line at `codec_cutoff.value`.
    #   - Right / Bottom: Average High-Frequency Power Spectrum P(f) (dB vs. frequency)
    #     specifically zoomed into 10 kHz - 22.05 kHz to clearly show the cliff/shelf.
    #
    # Hints:
    # - Use `signal.stft(filtered_audio, fs=fs, nperseg=2048, noverlap=1536, window='hann')`
    #   or `signal.spectrogram`.
    # - Convert magnitude to dB: `S_db = 20 * np.log10(np.abs(Zxx) + 1e-9)`.
    # - Average power spectrum: `P_db = 10 * np.log10(np.mean(np.abs(Zxx)**2, axis=1) + 1e-12)`.
    # - Highlight where the cutoff sits so you can visually correlate the audio artifact
    #   (Exercise 1) with the spectral shelf.
    # =========================================================================

    if filtered_audio is None:
        plot_output = mo.md("*Complete Exercise 1 and 2 to visualize the spectrogram.*")
    else:
        fig, (ax_spec, ax_psd) = plt.subplots(1, 2, figsize=(13, 4.5))

        # --- TODO: YOUR PLOTTING CODE HERE ---
        # ax_spec.pcolormesh(...)
        # ax_psd.plot(...)
        # -------------------------------------

        fig.tight_layout()
        plot_output = mo.ui.matplotlib(fig)

    plot_output
    return


@app.cell
def exercise_3_phrase_vs_global_spectrum(filtered_audio, mo, plt):
    # =========================================================================
    # HOMEWORK EXERCISE 3: Global FFT vs. Phrase-Windowed FFT (Tonal Balance & EQ)
    # =========================================================================
    # Goal:
    # Compare the raw 1D Fourier Transform of the ENTIRE audio track against
    # individual temporal sections / phrases (e.g. First Half vs. Second Half,
    # or Intro/Breakdown vs. Drop):
    #
    # 1. Compute the raw Discrete Fourier Transform (`np.fft.rfft`) of the
    #    entire signal `filtered_audio`. Window with `np.hanning(len(filtered_audio))`.
    # 2. Slice `filtered_audio` into two temporal phrases (e.g. Phrase A: 0.0 - 1.5s,
    #    Phrase B: 1.5s - 3.0s). Compute windowed `rfft` for each phrase.
    # 3. Plot two panels:
    #    - Panel 1 (Sub-bass & Bass: 20 Hz - 300 Hz on log scale):
    #      Observe how the long FFT reveals discrete fundamental kick/bass pitch spikes
    #      that are smeared in short STFT windows.
    #    - Panel 2 (Full Spectrum: 20 Hz - 22 kHz on log scale):
    #      Compare Phrase A vs. Phrase B vs. Full Track: see which frequencies dominate
    #      which part of the phrase (e.g., hi-hat air vs. sub weight), informing
    #      pre-drop HPF sweeps or 3-band EQ carve-outs.
    #
    # Hints:
    # - `freqs = np.fft.rfftfreq(N, d=1.0 / fs)`
    # - `mag_db = 20 * np.log10(np.abs(fft_vals) / (N / 2) + 1e-12)`
    # - Use `ax.set_xscale('log')` or `ax.semilogx(...)` for musical octaves.
    # =========================================================================

    if filtered_audio is None:
        phrase_output = mo.md("*Complete Exercise 1 to explore global vs. phrase Fourier spectra.*")
    else:
        fig_fft, (ax_sub, ax_full) = plt.subplots(1, 2, figsize=(13, 4.5))

        # --- TODO: YOUR FFT & PLOTTING CODE HERE ---
        # N_total = len(filtered_audio)
        # full_fft = np.fft.rfft(filtered_audio * np.hanning(N_total))
        # freqs_total = np.fft.rfftfreq(N_total, d=1.0 / fs)
        #
        # mid = N_total // 2
        # phrase_a = filtered_audio[:mid]
        # phrase_b = filtered_audio[mid:]
        # ...
        # ax_sub.semilogx(...)
        # ax_full.semilogx(...)
        # -------------------------------------------

        fig_fft.tight_layout()
        phrase_output = mo.ui.matplotlib(fig_fft)

    phrase_output
    return


if __name__ == "__main__":
    app.run()
