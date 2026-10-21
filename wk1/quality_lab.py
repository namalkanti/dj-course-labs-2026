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

    ---

    ### 1.5 The Gotcha of Slicing & Editing Compressed Audio (FLAC vs MP3)
    DJs and producers frequently trim, edit, or make intro edits of tracks. How the audio format stores data dictates whether simple splitting works cleanly:

    - **WAV / AIFF (Linear PCM)**: Trivial to slice at the byte level. Because samples are sequential linear amplitudes, any sub-second splice is just an offset calculation into the sample array.
    - **FLAC (Linear Predictive Coding)**: Frame-based with metadata headers. Naive stream copies without rebuilding seek tables will corrupt decoders (like `libsndfile` / Pioneer players failing with `psf_fseek()` errors). Re-encoding or using frame-aware FLAC tooling ensures valid header indices.
    - **MP3 (MDCT Coefficients & Bit Reservoir)**:
      1. *No sample-exact cutting*: MP3 is stored in 1152-sample transform frames ($\approx 26.1\text{ ms}$). You can only cut on frame boundaries.
      2. *50% Overlap (TDAC)*: MDCT frames overlap by 50% to cancel aliasing. Slicing raw frames abruptly introduces audible boundary clicks.
      3. *The Bit Reservoir*: MP3 encoders dynamically borrow bits from earlier frames (`main_data_begin` back-pointers). Slicing an MP3 without tracking the bit reservoir leaves orphan frames that fail to decode properly.
    - **DJ Takeaway**: When making track edits, quick cues, or snippets for analysis, **always decode to PCM first** (WAV, AIFF, or NumPy float array), make your sample-exact cuts, and only re-encode at the very end. Never do naive stream-copy slicing on lossy containers.
    """)
    return


@app.cell(hide_code=True)
def render_translation_table(mo):
    mo.md(r"""
    ### 1.6 Week 1 Vocabulary: Perceptual Symptoms of a "Fake 320" vs. DSP Reality

    When a DJ plays a low-bitrate stream or YouTube rip that was upscaled into a fake 320 kbps MP3 or WAV, they don't see the algorithm—they hear specific acoustic symptoms on a high-SPL club sound system.

    | DJ Perceptual Description | Domain | The Physical DSP Phenomenon |
    |---|---|---|
    | **"Muffled / Boxy / No Air"** | Frequency Domain | **Brick-wall lowpass cutoff ($f_c \le 16.0\text{ kHz}$)**: Encoders conserve bit budget by discarding the top octave ($16\text{–}22.05\text{ kHz}$). High harmonics of cymbals, snare sizzle, and vocal breath are completely eliminated. |
    | **"Mushy Kick / Lost Transient Punch"** | Time Domain | **MDCT temporal pre-echo**: Quantization noise spreads backward ~20 ms before the sharp transient inside the 1152-sample transform window, softening the initial attack of kicks and snares. |
    | **"Lacks Energy / Falls Flat in the Mix"** | Psychoacoustic Loudness | **Spectral energy depletion**: Stripping upper harmonics and aggressive psychoacoustic masking reduces perceived RMS loudness (ITU-R BS.1770 curve), making the track feel weak compared to legitimate masters. |

    *(Note: Validate these perceptual terms with your YourShot instructor during your 1-on-1 session.)*
    """)
    return


@app.cell
def load_dsp_modules():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as signal
    import soundfile as sf
    from yourshot.audio_catalog import load_audio_catalog
    from yourshot.synth import apply_lossy_cutoff

    return (
        Path,
        apply_lossy_cutoff,
        load_audio_catalog,
        plt,
        sf,
    )


@app.cell
def get_catalog_and_fs(load_audio_catalog):
    catalog, fs = load_audio_catalog(fs=44100)
    return catalog, fs


@app.cell(hide_code=True)
def render_part2_intro(mo):
    mo.md(r"""
    ## Part 2: Pristine Signal Analysis & Codec Filter Lab (Homework Exercises)

    In this lab, you'll work with an immutable catalog of clean audio signals (both mathematical synthetic signals and real reference tracks).

    Follow the natural DSP progression:
    1. **Exercise 1 (Baseline)**: Listen to the uncompressed audio, compute its raw 1D Fourier spectrum, and compute its 2D STFT spectrogram.
    2. **Exercise 2 (Phrases vs. Global)**: Compare phrase-windowed FFTs (breakdown vs. drop) against the entire signal.
    3. **Exercise 3 (Lossy Cutoff Shock)**: Apply simulated MP3 encoder filters (320k, 192k, 128k) and observe the destruction of high-frequency content.
    """)
    return

@app.cell(hide_code=True)
def render_audio_gallery(catalog, fs, mo):
    gallery_items = []
    for name, audio in catalog.items():
        gallery_items.append(
            mo.hstack(
                [
                    mo.md(f"**{name}**"),
                    mo.audio(audio, rate=fs),
                ],
                justify="space-between",
                align="center",
            )
        )
    mo.vstack([
        mo.md("### Audio Track Catalog"),
        mo.vstack(gallery_items, gap=1),
    ])
    return

@app.cell
def build_controls(catalog, mo):
    signal_selector = mo.ui.dropdown(
        options=list(catalog.keys()),
        value="Synthetic: Drum Groove (128 BPM)",
        label="Select Audio Track / Signal",
    )

    codec_cutoff = mo.ui.dropdown(
        options={
            "Lossless / Passthrough (22.05 kHz Nyquist)": 22050.0,
            "~320 kbps MP3 (20.0 kHz shelf)": 20000.0,
            "~192 kbps MP3 (18.5 kHz cutoff)": 18500.0,
            "~128 kbps MP3 (16.0 kHz brick-wall)": 16000.0,
        },
        value="Lossless / Passthrough (22.05 kHz Nyquist)",
        label="Simulated Codec Cutoff (for Ex 3)",
    )

    mo.hstack([signal_selector, codec_cutoff], justify="start", gap=2)
    return codec_cutoff, signal_selector


@app.cell
def get_selected_signal(catalog, signal_selector):
    selected_audio = catalog[signal_selector.value]
    return (selected_audio,)

@app.cell
def exercise_1_pristine_analysis(fs, mo, np, plt, selected_audio, signal):
    player = mo.audio(selected_audio, rate=fs)

    # 1. Averaged Power Spectral Density via Welch's method (N=4096, 50% overlap)
    _nperseg_welch = min(4096, len(selected_audio))
    freqs, psd = signal.welch(
        selected_audio,
        fs=fs,
        window="hann",
        nperseg=_nperseg_welch,
        noverlap=_nperseg_welch // 2,
        scaling="spectrum",
    )
    mag_db = 10.0 * np.log10(psd + 1e-12)

    # 2. 2D STFT Spectrogram (Time-Frequency)
    _nperseg_stft = 2048
    _noverlap_stft = 1536
    f_stft, t_stft, Zxx = signal.stft(
        selected_audio,
        fs=fs,
        window="hann",
        nperseg=_nperseg_stft,
        noverlap=_noverlap_stft,
    )
    S_db = 20.0 * np.log10(np.abs(Zxx) + 1e-9)
    # Clip display dynamic range to top 80 dB
    vmax = float(np.max(S_db))
    vmin = vmax - 80.0

    # 3. Dual-panel visualization
    fig, (ax_spec, ax_stft) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left: Smoothed Power Spectrum (log frequency scale for musical octaves)
    ax_spec.semilogx(freqs, mag_db, color="#00bcd4", lw=1.5)
    ax_spec.set_xlim(20.0, fs / 2.0)
    ax_spec.set_ylim(-100.0, max(0.0, float(np.max(mag_db) + 6.0)))
    ax_spec.set_title("Averaged Power Spectrum (Welch PSD)")
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Power (dBFS)")
    ax_spec.grid(True, which="both", alpha=0.25)

    # Right: 2D STFT Spectrogram
    mesh = ax_stft.pcolormesh(
        t_stft,
        f_stft,
        S_db,
        shading="gouraud",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax_stft.set_ylim(0.0, fs / 2.0)
    ax_stft.set_title("2D STFT Spectrogram (Time vs Freq)")
    ax_stft.set_xlabel("Time (s)")
    ax_stft.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax_stft, label="dB")

    fig.tight_layout()

    output_1 = mo.vstack([
        mo.md("#### Exercise 1: Pristine Audio Analysis (Averaged Spectrum vs. 2D Spectrogram)"),
        player,
        mo.as_html(fig),
    ])
    output_1
    return


@app.cell
def exercise_2_phrase_vs_global_spectrum(fs, mo, np, plt, selected_audio, signal):
    # Segment audio into Phrase A (first half) and Phrase B (second half)
    mid = len(selected_audio) // 2
    phrase_a = selected_audio[:mid]
    phrase_b = selected_audio[mid:]

    _nperseg_p = min(4096, len(phrase_a))
    _noverlap_p = _nperseg_p // 2

    freqs_total, psd_total = signal.welch(
        selected_audio, fs=fs, window="hann", nperseg=_nperseg_p, noverlap=_noverlap_p, scaling="spectrum"
    )
    db_total = 10.0 * np.log10(psd_total + 1e-12)

    freqs_a, psd_a = signal.welch(
        phrase_a, fs=fs, window="hann", nperseg=_nperseg_p, noverlap=_noverlap_p, scaling="spectrum"
    )
    db_a = 10.0 * np.log10(psd_a + 1e-12)

    freqs_b, psd_b = signal.welch(
        phrase_b, fs=fs, window="hann", nperseg=_nperseg_p, noverlap=_noverlap_p, scaling="spectrum"
    )
    db_b = 10.0 * np.log10(psd_b + 1e-12)

    fig_fft, (ax_sub, ax_full) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left: Sub-bass & Bass register (20 Hz - 300 Hz) showing kick & sub pitch spikes
    ax_sub.semilogx(freqs_total, db_total, label="Full Signal", color="#00bcd4", lw=1.5, alpha=0.9)
    ax_sub.semilogx(freqs_a, db_a, label="Phrase A (0 - 50%)", color="#ff9800", lw=1.2, alpha=0.8)
    ax_sub.semilogx(freqs_b, db_b, label="Phrase B (50 - 100%)", color="#e91e63", lw=1.2, alpha=0.8)
    ax_sub.set_xlim(20.0, 300.0)
    ax_sub.set_ylim(-80.0, max(0.0, float(np.max(db_total) + 6.0)))
    ax_sub.set_title("Sub-Bass Fundamental Peaks (20 - 300 Hz)")
    ax_sub.set_xlabel("Frequency (Hz)")
    ax_sub.set_ylabel("Power (dBFS)")
    ax_sub.grid(True, which="both", alpha=0.25)
    ax_sub.legend(loc="upper right")

    # Right: Full Frequency Spectrum (20 Hz - 22.05 kHz) showing tonal tilt & high-end air
    ax_full.semilogx(freqs_total, db_total, label="Full Signal", color="#00bcd4", lw=1.5, alpha=0.9)
    ax_full.semilogx(freqs_a, db_a, label="Phrase A", color="#ff9800", lw=1.2, alpha=0.8)
    ax_full.semilogx(freqs_b, db_b, label="Phrase B", color="#e91e63", lw=1.2, alpha=0.8)
    ax_full.set_xlim(20.0, fs / 2.0)
    ax_full.set_ylim(-90.0, max(0.0, float(np.max(db_total) + 6.0)))
    ax_full.set_title("Full Spectrum Tonal Balance (20 Hz - 22 kHz)")
    ax_full.set_xlabel("Frequency (Hz)")
    ax_full.set_ylabel("Power (dBFS)")
    ax_full.grid(True, which="both", alpha=0.25)
    ax_full.legend(loc="upper right")

    fig_fft.tight_layout()
    output_2 = mo.vstack([
        mo.md("#### Exercise 2: Global vs. Phrase-Windowed Spectral Analysis"),
        mo.as_html(fig_fft),
    ])
    output_2
    return


@app.cell
def exercise_3_codec_cutoff_simulation(
    apply_lossy_cutoff,
    codec_cutoff,
    fs,
    mo,
    np,
    plt,
    selected_audio,
    signal,
):
    filtered = apply_lossy_cutoff(selected_audio, cutoff_hz=codec_cutoff.value, fs=fs)
    player_filtered = mo.audio(filtered, rate=fs)

    # 1. Compute STFT of filtered audio
    f_filt, t_filt, Z_filt = signal.stft(
        filtered,
        fs=fs,
        window="hann",
        nperseg=2048,
        noverlap=1536,
    )
    S_filt_db = 20.0 * np.log10(np.abs(Z_filt) + 1e-9)
    vmax_filt = float(np.max(S_filt_db))
    vmin_filt = vmax_filt - 80.0

    # 2. Time-averaged power spectrum P(f) in dB
    # Mean power across time frames: 10 * log10(mean(|Zxx|^2))
    power_lin = np.mean(np.abs(Z_filt) ** 2, axis=1)
    power_db = 10.0 * np.log10(power_lin + 1e-12)
    # Normalize peak to 0 dB for easy cutoff inspection
    power_db -= np.max(power_db)

    fig_filtered, (ax_stft_filt, ax_shelf) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left: Filtered STFT Spectrogram with dashed cutoff line
    mesh_filt = ax_stft_filt.pcolormesh(
        t_filt,
        f_filt,
        S_filt_db,
        shading="gouraud",
        cmap="magma",
        vmin=vmin_filt,
        vmax=vmax_filt,
    )
    ax_stft_filt.set_ylim(0.0, fs / 2.0)
    ax_stft_filt.axhline(
        codec_cutoff.value,
        color="#00e676",
        linestyle="--",
        lw=1.5,
        label=f"Cutoff: {codec_cutoff.value / 1000.0:.1f} kHz",
    )
    ax_stft_filt.set_title("Filtered STFT Spectrogram")
    ax_stft_filt.set_xlabel("Time (s)")
    ax_stft_filt.set_ylabel("Frequency (Hz)")
    ax_stft_filt.legend(loc="upper right")
    fig_filtered.colorbar(mesh_filt, ax=ax_stft_filt, label="dB")

    # Right: High-Frequency Zoom (10 kHz - 22.05 kHz) showing steep cutoff shelf
    mask_high = f_filt >= 10000.0
    ax_shelf.plot(f_filt[mask_high] / 1000.0, power_db[mask_high], color="#ff5722", lw=1.5)
    ax_shelf.axvline(
        codec_cutoff.value / 1000.0,
        color="#00e676",
        linestyle="--",
        lw=1.5,
        label=f"Cutoff ({codec_cutoff.value / 1000.0:.1f} kHz)",
    )
    ax_shelf.set_xlim(10.0, (fs / 2.0) / 1000.0)
    ax_shelf.set_ylim(-70.0, 5.0)
    ax_shelf.set_title("High-Frequency Spectral Rolloff (10 - 22 kHz)")
    ax_shelf.set_xlabel("Frequency (kHz)")
    ax_shelf.set_ylabel("Relative Power (dB)")
    ax_shelf.grid(True, alpha=0.3)
    ax_shelf.legend(loc="lower left")

    fig_filtered.tight_layout()
    output_3 = mo.vstack([
        mo.md(f"#### Exercise 3: Codec Cutoff Filtering (Simulated Cutoff: **{codec_cutoff.value / 1000.0:.1f} kHz**)"),
        player_filtered,
        mo.as_html(fig_filtered),
    ])
    output_3
    return


if __name__ == "__main__":
    app.run()
