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
    # Audio Quality Lab: Codec Analysis & Transcode Detection

    *An interactive digital signal processing lab analyzing lossy vs. lossless audio compression, psychoacoustic filtering, and automated detection of upsampled/transcoded files.*
    """)
    return


@app.cell(hide_code=True)
def render_foundations(mo):
    mo.md(r"""
    ## Part 1: Audio Quality Fundamentals for DJs

    Audio quality matters when preparing a set. Lossy encoding can introduce audible artifacts, but whether you notice them depends on the source, encoder, playback equipment, processing, and listening conditions. A larger or louder sound system does not automatically make every difference audible.

    A pervasive problem in DJing is the **fake transcode** (taking a free 128 kbps MP3 or a YouTube rip and re-saving it as a "320 kbps MP3" or "WAV"). While Rekordbox and CDJ screens will show a high bitrate or lossless format, the underlying audio is permanently compromised.

    ---

    ### 1.1 Sample Rate: 44.1 kHz vs 48 kHz
    - **44.1 kHz** is the standard for music releases (CDs, Beatport, Bandcamp, Spotify). It dates back to early digital audio being stored on video tape cassettes.
    - **48 kHz** is the standard for film, video, and YouTube.
    - **DJ Takeaway**: Both sound identical to human ears. Modern CDJs and Rekordbox play both natively. However, **44.1 kHz / 16-bit** is the universal safety standard that plays reliably on *any* CDJ ever made, even legacy models with older firmware.

    ---

    ### 1.2 Bit Depth: 16-bit vs 24-bit
    - **16-bit is generally sufficient for DJ playback**: It offers roughly 96 dB of dynamic range. With proper preparation and gain staging, quantization noise is typically not the limiting factor in a club.
    - **24-bit is useful for production**: Its lower quantization noise allows recording at conservative levels and supports subsequent processing. It does not raise the digital clipping ceiling.
    - **DJ Takeaway**: Converting a 16-bit source to 24-bit cannot recover information. A genuine 24-bit release can contain additional low-level detail, but properly prepared 16-bit audio is generally sufficient for DJ playback.

    ---

    ### 1.3 File Formats: WAV vs AIFF vs FLAC vs MP3
    Audio formats divide strictly into two categories: **lossless** and **lossy**.
    - **Lossless** formats preserve 100% of the original PCM audio samples. When decoded, the audio waveform is bit-for-bit identical to the studio master.
    - **Lossy** formats permanently discard audio data using psychoacoustic modeling to achieve smaller file sizes. Once discarded, that information is lost forever.

    | Format | Type | Compression | Metadata / Tagging Support | Club / CDJ Compatibility | Best Use Case |
    |---|---|---|---|---|---|
    | **WAV** | Lossless | None (Raw PCM) | **Poor**: ID3 tagging is non-standard. Hot cues, album art, and artist tags often get stripped when moving between Rekordbox USBs. | Universal across all CDJ models. | Studio recording, raw stem export. |
    | **AIFF** | Lossless | None (Raw PCM) | Supports ID3 chunks for descriptive metadata; cue/export behavior depends on software. | Universal across all CDJ models. | A practical PCM option with metadata support; verify your export workflow. |
    | **FLAC** | Lossless | Compressed (~50% size, like a ZIP for audio) | **Excellent**: Vorbis comments preserve all track metadata and album art. | CDJ-2000NXS2, CDJ-3000, and Opus-Quad only. Older club gear (CDJ-900, CDJ-2000 Nexus) cannot play it. | Archival, home listening, modern festival stages. |
    | **MP3** | Lossy | Heavy (~80–90% size reduction) | **Excellent**: Full ID3v2 tag support. | Universal across all CDJ models. | Legitimate **320 kbps** is club-ready and saves USB space; lower bitrates increase artifact risk, depending on source and encoding. |

    ---

    ### 1.4 How MP3 Compression Works (And Why Low Bitrates Sound Bad)
    Uncompressed audio takes up ~10 MB per minute. To make smaller files, MP3 uses **psychoacoustics**: discarding sounds the human ear is less likely to notice (like quiet frequencies playing at the same time as a loud kick drum).

    - **320 kbps (High Quality)**: A good encoder has more budget to reduce perceptually important errors. Bandwidth depends on settings; well-encoded files are often difficult to distinguish from the source in controlled listening.
    - **128 kbps (Low Quality / YouTube Rips)**: The encoder runs out of data space and makes aggressive cuts:
      - Some encoder settings low-pass near **16 kHz**. The exact cutoff depends on the encoder and settings, not bitrate alone.
      - Quantization and temporal artifacts can affect transients, with audibility depending on the material.
    - **What about 192 kbps?**
      - It often preserves more information than 128 kbps, but bitrate alone does not specify bandwidth, stereo behavior, or perceptual quality.
      - Prefer trusted sources and avoid unnecessary lossy re-encoding. Treat listening and spectral inspection as complementary checks, not a universal bitrate pass/fail test.

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


@app.cell
def load_dsp_modules():
    import io
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal
    import soundfile as sf
    from dj_dsp.synth import apply_lossy_cutoff, encode_mp3_lame

    return apply_lossy_cutoff, encode_mp3_lame, io, np, plt, sf


@app.cell
def load_plotting():
    from dj_dsp.spectral import (
        analyze_spectral_cutoff,
        compute_welch_spectrum,
        compute_stft_spectrogram,
        compute_three_band_waveform,
    )
    from dj_dsp.plotting import (
        plot_overlaid_welch_shelf,
        plot_welch_spectrum,
        plot_stft_spectrogram,
        plot_three_band_waveform,
    )

    return (
        analyze_spectral_cutoff,
        compute_stft_spectrogram,
        compute_three_band_waveform,
        compute_welch_spectrum,
        plot_overlaid_welch_shelf,
        plot_stft_spectrogram,
        plot_three_band_waveform,
        plot_welch_spectrum,
    )


@app.cell
def get_catalog_and_fs():
    from dj_dsp.audio_catalog import load_audio_catalog
    catalog, fs = load_audio_catalog(fs=44100)
    return catalog, fs


@app.cell(hide_code=True)
def render_part2_intro(mo):
    mo.md(r"""
    ## Part 2: Pristine Signal Analysis & Codec Filter Analysis

    In this section, you'll work with an immutable catalog of clean audio signals (both mathematical synthetic signals and real reference tracks).

    Follow the natural DSP progression:
    1. **Exercise 1 (Baseline)**: Listen to uncompressed audio, inspect its Rekordbox 3-band waveform, 1D Welch spectrum, and 2D STFT spectrogram.
    2. **Exercise 2 (Lossy Cutoff Simulation)**: Apply simulated MP3 encoder filters (320k, 192k, 128k) and inspect the destruction of high-frequency content.
    3. **Exercise 3 (Try It Yourself)**: Upload your own audio files and inspect them for fake transcode cutoffs.
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


@app.cell(hide_code=True)
def build_controls(catalog, mo):
    signal_selector_ex1 = mo.ui.dropdown(
        options=list(catalog.keys()),
        value="Synthetic: Drum Groove (128 BPM)",
        label="Select Pristine Track",
    )

    signal_selector_ex2 = mo.ui.dropdown(
        options=list(catalog.keys()),
        value="Real Audio: Persistence - Dabmasterars (Lossless FLAC)",
        label="Select Track for Codec Cutoff Test",
    )

    codec_cutoff = mo.ui.dropdown(
        options={
            "Lossless Master (FLAC / WAV Passthrough)": 0,
            "Real LAME MP3: 320 kbps (Club Standard)": 320,
            "Real LAME MP3: 192 kbps (Mid-Tier Compression)": 192,
            "Real LAME MP3: 128 kbps (Typical YouTube Rip)": 128,
            "Real LAME MP3: 64 kbps (Extreme Bit-Starvation)": 64,
        },
        value="Real LAME MP3: 128 kbps (Typical YouTube Rip)",
        label="Simulated Codec Format (LAME MP3)",
    )
    return codec_cutoff, signal_selector_ex1, signal_selector_ex2


@app.cell(hide_code=True)
def get_selected_signals(catalog, signal_selector_ex1, signal_selector_ex2):
    audio_ex1 = catalog[signal_selector_ex1.value]
    audio_ex2 = catalog[signal_selector_ex2.value]
    return audio_ex1, audio_ex2


@app.cell(hide_code=True)
def make_analysis_tabs_helper(
    compute_stft_spectrogram,
    compute_three_band_waveform,
    compute_welch_spectrum,
    io,
    mo,
    plot_overlaid_welch_shelf,
    plot_stft_spectrogram,
    plot_three_band_waveform,
    plot_welch_spectrum,
    plt,
):
    """Factory cell providing the composite analytical tab view generator for Marimo exercises."""

    def fig_to_image(fig, dpi=100):
        """Render a Matplotlib figure as an optimized raster PNG for lightweight Marimo DOM payloads."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return mo.image(buf.read(), rounded=True)

    def render_analysis_tabs(audio, fs, cutoff_marker_hz=None):
        """Construct a tabbed Marimo UI component displaying the core audio diagnostic suite.

        Transforms a raw 1D audio time series into three standard analytical projections:
        1. Pioneer CDJ-3000 3-band waveform envelope (time domain, frequency-separated).
        2. Welch Power Spectral Density (frequency domain, musical log scale).
        3. 2D Short-Time Fourier Transform spectrogram (time-frequency joint domain).

        Parameters
        ----------
        audio : np.ndarray
            Monophonic 1D float32 audio signal array normalized to [-1.0, 1.0].
        fs : int
            Sampling frequency in Hz (typically 44100).
        cutoff_marker_hz : float or None, optional
            Frequency threshold in Hz to annotate with a dashed marker line across plots.

        Returns
        -------
        mo.ui.tabs
            Interactive Marimo tab widget containing individual view tabs plus a stacked overview tab.
        """
        waveform_view = compute_three_band_waveform(audio, fs=fs)
        fig_wf, _ = plot_three_band_waveform(waveform_view)
        fig_wf.tight_layout()
        img_wf = fig_to_image(fig_wf)

        spectrum_view = compute_welch_spectrum(audio, fs=fs)
        fig_spec, _ = plot_welch_spectrum(spectrum_view, log_freq=True, cutoff_marker_hz=cutoff_marker_hz)
        fig_spec.tight_layout()
        img_spec = fig_to_image(fig_spec)

        spectrogram_view = compute_stft_spectrogram(audio, fs=fs)
        fig_stft, _ = plot_stft_spectrogram(spectrogram_view, cutoff_marker_hz=cutoff_marker_hz)
        fig_stft.tight_layout()
        img_stft = fig_to_image(fig_stft)

        card_wf = mo.vstack([
            mo.md(
                "##### DJ-Style 3-Band Waveform (Illustrative Model)\n"
                "_Time-domain energy envelope split using this lab's illustrative crossovers: "
                "**Cyan** (Low, <250 Hz), **Orange** (Mid, 250–2500 Hz), **Blue** (High, >2500 Hz)._"
            ),
            img_wf,
        ])

        card_spec = mo.vstack([
            mo.md(
                "##### Averaged Power Spectrum (Tonal Balance)\n"
                "_Frequency-domain tonal balance averaged across the track (20 Hz to 22.05 kHz). "
                "Notice the sub-bass fundamentals and overall spectral tilt._"
            ),
            img_spec,
        ])

        card_stft = mo.vstack([
            mo.md(
                "##### 2D STFT Spectrogram (Time-Frequency DSP View)\n"
                "_Time-frequency representation: vertical axis is frequency (0–22 kHz), horizontal axis is time. "
                "Reveals transient bursts, cymbal sustain, and codec cutoff boundaries._"
            ),
            img_stft,
        ])

        return mo.ui.tabs({
            "CDJ 3-Band Waveform": card_wf,
            "Welch Power Spectrum": card_spec,
            "2D STFT Spectrogram": card_stft,
            "All Views (Stacked)": mo.vstack([card_wf, card_spec, card_stft], gap=1.5),
        })

    def render_comparison_tabs(
        audio_orig,
        audio_filt,
        fs,
        cutoff_hz=None,
        orig_label="Pristine Master",
        filt_label="Filtered Audio",
    ):
        """Construct side-by-side and overlaid diagnostic tabs for codec before/after comparison."""
        # 1. Overlaid High-Frequency Shelf (10–22 kHz)
        spec_orig = compute_welch_spectrum(audio_orig, fs=fs).crop(f_min=10000.0, normalize_peak=True)
        spec_filt = compute_welch_spectrum(audio_filt, fs=fs).crop(f_min=10000.0, normalize_peak=True)
        fig_shelf, _ = plot_overlaid_welch_shelf(
            spec_orig,
            spec_filt,
            cutoff_marker_hz=cutoff_hz,
            title="High-Frequency Shelf: " + orig_label + " vs. " + filt_label,
            orig_label=orig_label,
            filt_label=filt_label,
        )
        fig_shelf.tight_layout()

        # 2. Side-by-Side 2D Spectrograms (Filtered on Left, Pristine on Right)
        stft_filt = compute_stft_spectrogram(audio_filt, fs=fs)
        fig_stft_filt, _ = plot_stft_spectrogram(
            stft_filt,
            title=f"{filt_label} Spectrogram",
            cutoff_marker_hz=cutoff_hz,
        )
        fig_stft_filt.tight_layout()

        stft_orig = compute_stft_spectrogram(audio_orig, fs=fs)
        fig_stft_orig, _ = plot_stft_spectrogram(
            stft_orig,
            title=f"{orig_label} Spectrogram",
        )
        fig_stft_orig.tight_layout()

        img_shelf = fig_to_image(fig_shelf)
        img_stft_side = mo.hstack(
            [fig_to_image(fig_stft_filt), fig_to_image(fig_stft_orig)],
            justify="space-between",
            gap=1.5,
        )

        return mo.ui.tabs({
            "High-Freq Shelf (Overlaid Inspector)": mo.vstack([
                mo.md(
                    "_Both curves plotted together on a linear scale (10–22.05 kHz). "
                    "The shaded purple band marks the upper human hearing limit (16–20 kHz). "
                    "Notice the sharp drop into the noise floor on the filtered audio._"
                ),
                img_shelf,
            ]),
            "2D Spectrograms (Side-by-Side)": mo.vstack([
                mo.md(
                    "_Compare time-frequency heatmaps: **Filtered (Left)** vs. **Pristine (Right)**. "
                    "Notice the horizontal cutoff void above the dashed marker line._"
                ),
                img_stft_side,
            ]),
        })

    return render_analysis_tabs, render_comparison_tabs


@app.cell
def exercise_1_analysis(
    audio_ex1,
    fs,
    mo,
    render_analysis_tabs,
    signal_selector_ex1,
):
    player = mo.audio(audio_ex1, rate=fs)
    analysis_tabs = render_analysis_tabs(audio_ex1, fs=fs, cutoff_marker_hz=None)

    output_1 = mo.vstack([
        mo.md("#### Exercise 1: Audio Analysis"),
        signal_selector_ex1,
        player,
        analysis_tabs,
    ], gap=1.2)

    output_1
    return


@app.cell(hide_code=True)
def exercise_2_codec_cutoff_simulation(
    audio_ex2,
    codec_cutoff,
    encode_mp3_lame,
    fs,
    mo,
    render_comparison_tabs,
    signal_selector_ex2,
):
    _bitrate = int(codec_cutoff.value)
    _encoded = encode_mp3_lame(audio_ex2, fs=fs, bitrate_kbps=_bitrate) if _bitrate > 0 else audio_ex2.copy()
    _cutoff_hz = {320: 20500.0, 192: 18500.0, 128: 16000.0}.get(_bitrate, None)
    _filt_label = f"MP3 ({_bitrate} kbps)" if _bitrate > 0 else "Lossless Passthrough"

    # 1. Audio Players Side-by-Side
    _player_orig = mo.audio(audio_ex2, rate=fs)
    _player_filt = mo.audio(_encoded, rate=fs)
    _players_row = mo.hstack([
        mo.vstack([mo.md("**Pristine (Lossless Master)**"), _player_orig], gap=0.4),
        mo.vstack([mo.md(f"**{_filt_label}**"), _player_filt], gap=0.4),
    ], justify="space-around", gap=2.0)

    # 2. Honest Educational Callout
    _callout_note = mo.md(
        "> #### Listening and Spectral Inspection Answer Different Questions\n"
        ">\n"
        "> **1. Listening on Headphones:**\n"
        "> Switch between **Lossless Master** and **128 kbps** above and listen carefully. "
        "In our comparisons on tracks like *Persistence*, differences were difficult to identify by ear. Your results may differ. "
        "Difficulty hearing a difference is consistent with effective perceptual encoding; it does not establish that the signals are identical. "
        "High-frequency energy can be much weaker than bass and midrange; hearing, playback response, and listening conditions also affect what you can detect.\n"
        ">\n"
        "> **2. What the Microscope Sees:**\n"
        "> Look at the **High-Freq Shelf** and **2D Spectrogram** tabs below. Even when a difference is not apparent by ear, "
        "these plots can reveal high-frequency attenuation that is difficult to hear. Its location and depth depend on the encoder settings and source.\n"
        ">\n"
        "> **3. Practical Preparation:**\n"
        "> Different playback systems, adjacent tracks, and pitch/time processing may expose artifacts that were not obvious in an initial audition; they do not guarantee that a difference becomes audible. Increasing volume is not a reliable or safe quality test.\n"
        ">\n"
        "> **The DJ Rule of Thumb**: *Do not rely on listening alone to establish provenance. Prefer trusted sources; use listening to assess audible problems and spectral inspection to investigate bandwidth. Neither a cutoff nor its absence certifies encoding history.*"
    )

    # 3. Side-by-Side and Overlaid Analytical Tabs (via helper)
    _tabs = render_comparison_tabs(
        audio_orig=audio_ex2,
        audio_filt=_encoded,
        fs=fs,
        cutoff_hz=_cutoff_hz,
        orig_label="Lossless Master",
        filt_label=_filt_label,
    )

    _controls_row = mo.hstack([signal_selector_ex2, codec_cutoff], justify="start", gap=2)

    output_2 = mo.vstack([
        mo.md("#### Exercise 2: Real MP3 Codec Encoding (LAME Bitrate Comparison)"),
        _controls_row,
        _players_row,
        _callout_note,
        _tabs,
    ], gap=1.4)

    output_2
    return


@app.cell(hide_code=True)
def build_uploader(mo):
    audio_uploader = mo.ui.file(
        filetypes=[".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"],
        label="Upload an audio track (.wav, .aiff, .flac, .mp3) to inspect",
    )

    upload_cutoff = mo.ui.dropdown(
        options={
            "None / Passthrough": 22050.0,
            "~320 kbps MP3 (20.0 kHz shelf)": 20000.0,
            "~192 kbps MP3 (18.5 kHz cutoff)": 18500.0,
            "~128 kbps MP3 (16.0 kHz brick-wall)": 16000.0,
        },
        value="~128 kbps MP3 (16.0 kHz brick-wall)",
        label="Simulate Lossy Codec on Uploaded Track",
    )
    return audio_uploader, upload_cutoff


@app.cell(hide_code=True)
def parse_uploaded_audio(audio_uploader, io, np, sf):
    """Decode uploaded audio, normalize, and extract duration for downstream interactive windowing."""
    if not audio_uploader.value:
        mono_audio = None
        upload_fs = 44100
        total_duration_sec = 0.0
    else:
        file_bytes = audio_uploader.contents()
        with io.BytesIO(file_bytes) as buf:
            raw_data, upload_fs = sf.read(buf, dtype="float32")

        # Downmix stereo to mono if necessary
        if raw_data.ndim > 1:
            mono_audio = np.mean(raw_data, axis=1)
        else:
            mono_audio = raw_data

        # Peak normalization to [-1.0, 1.0] if non-silent
        peak = float(np.max(np.abs(mono_audio)))
        if peak > 1e-6:
            mono_audio = mono_audio / peak

        total_duration_sec = float(len(mono_audio) / upload_fs)
    return mono_audio, total_duration_sec, upload_fs


@app.cell(hide_code=True)
def build_upload_window_slider(mo, np, total_duration_sec):
    """Build a dynamic 20-second window start slider if the uploaded track exceeds 20 seconds."""
    if total_duration_sec <= 20.0:
        window_slider = None
    else:
        max_start = int(np.floor(total_duration_sec - 20.0))
        default_val = min(15, max_start)
        window_slider = mo.ui.slider(
            start=0,
            stop=max_start,
            step=1,
            value=default_val,
            label=f"20s Analysis Window Start (Total: {total_duration_sec:.1f}s)",
        )
    return (window_slider,)


@app.cell(hide_code=True)
def exercise_3_upload_inspector(
    analyze_spectral_cutoff,
    apply_lossy_cutoff,
    audio_uploader,
    mo,
    mono_audio,
    render_comparison_tabs,
    total_duration_sec,
    upload_cutoff,
    upload_fs,
    window_slider,
):
    if mono_audio is None:
        output_3 = mo.vstack([
            mo.md("#### Exercise 3: Inspect Your Own Track (Transcode Forensic Audit)"),
            audio_uploader,
            mo.md(
                "_Upload any audio file from your personal collection above. "
                "The lab will render its high-frequency shelf, 2D spectrogram, "
                "and run a heuristic cutoff inspection. This cannot verify authenticity or encoding history._"
            ),
        ], gap=1.2)
    else:
        # Determine 20-second time slice from the interactive slider
        if window_slider is not None:
            start_sec = float(window_slider.value)
            duration_sec = 20.0
        else:
            start_sec = 0.0
            duration_sec = total_duration_sec

        start_sample = int(start_sec * upload_fs)
        num_samples = int(duration_sec * upload_fs)
        analyzed_orig = mono_audio[start_sample : start_sample + num_samples]

        # Apply cutoff filter if enabled
        _is_filtered = upload_cutoff.value < 22000.0
        _cutoff_hz = upload_cutoff.value if _is_filtered else None
        analyzed_filt = (
            apply_lossy_cutoff(analyzed_orig, cutoff_hz=upload_cutoff.value, fs=upload_fs)
            if _is_filtered
            else analyzed_orig
        )

        # Automated Spectral Cutoff Detection Analysis on the uploaded audio
        analysis = analyze_spectral_cutoff(analyzed_orig, fs=upload_fs)

        if analysis.is_suspicious_transcode:
            badge_color = "#f44336"  # Red alert
            icon = "⚠️"
        else:
            badge_color = "#607d8b"  # Neutral: absence of a flag is not verification
            icon = "ℹ️"

        verdict_card = mo.md(
            f"""
            <div style="border: 2px solid {badge_color}; border-radius: 8px; padding: 12px 18px; margin: 8px 0; background-color: rgba(0,0,0,0.03);">
                <div style="font-size: 1.15em; font-weight: bold; color: {badge_color};">
                    {icon} Spectral Inspection: {analysis.verdict}
                </div>
                <div style="margin-top: 6px; font-size: 0.95em;">
                    • <b>Detected High-Frequency Cutoff:</b> <code>{analysis.detected_cutoff_hz:,.0f} Hz</code><br>
                    • <b>Stopband Attenuation:</b> <code>{analysis.stopband_attenuation_db:.1f} dB</code>
                </div>
                <div style="margin-top: 6px; font-size: 0.85em; color: #666;">
                    {"<b>What this suggests:</b> The high-frequency cutoff resembles patterns seen in some lower-bitrate MP3 encodings. The bitrate in the label is a rough comparison, not a measurement of the original file's bitrate. Other encoder settings or filtering during production can produce similar results; this is a clue, not proof." if analysis.is_suspicious_transcode else "<b>What this means:</b> We did not find the cutoff pattern used to flag possible lower-bitrate sources, or this passage had insufficient energy for analysis. This does not certify lossless audio or a high-bitrate source."}
                    <br>Try another passage and consider the file's source alongside listening and spectral inspection.
                </div>
            </div>
            """
        )

        uploaded_name = audio_uploader.name(0) or "Uploaded Audio"

        _player_orig = mo.audio(analyzed_orig, rate=upload_fs)
        _player_filt = mo.audio(analyzed_filt, rate=upload_fs)

        _players_row = mo.hstack([
            mo.vstack([mo.md(f"**As Uploaded (`{uploaded_name}`)**"), _player_orig], gap=0.4),
            mo.vstack([mo.md(f"**With Simulated Cutoff ({upload_cutoff.value / 1000.0:.1f} kHz)**" if _is_filtered else "**Cutoff (None)**"), _player_filt], gap=0.4),
        ], justify="space-around", gap=2.0)

        _tabs = render_comparison_tabs(
            audio_orig=analyzed_orig,
            audio_filt=analyzed_filt,
            fs=upload_fs,
            cutoff_hz=_cutoff_hz,
            orig_label="As Uploaded",
            filt_label=f"Cutoff ({upload_cutoff.value / 1000.0:.1f} kHz)" if _is_filtered else "Filtered",
        )

        file_info = mo.md(
            f"**File:** `{uploaded_name}` &nbsp;|&nbsp; "
            f"**Sample Rate:** `{upload_fs} Hz` &nbsp;|&nbsp; "
            f"**Total Track Duration:** `{total_duration_sec:.1f} s` &nbsp;|&nbsp; "
            f"**Active Slice:** `{start_sec:.1f}s – {start_sec + duration_sec:.1f}s`"
        )

        controls = [audio_uploader, upload_cutoff]
        if window_slider is not None:
            controls.append(window_slider)

        output_3 = mo.vstack([
            mo.md("#### Exercise 3: Inspect Your Own Track (Transcode Forensic Audit)"),
            mo.vstack(controls, gap=1.0),
            file_info,
            verdict_card,
            _players_row,
            _tabs,
        ], gap=1.4)

    output_3
    return


@app.cell(hide_code=True)
def render_translation_table(mo):
    mo.md(r"""
    ## Part 3: DJ vs. DSP Vocabulary & Visual Translation Guide

    In this lab, we computed three distinct mathematical projections of audio signals. A DJ looking at a CDJ screen or prep software sees familiar musical tools, while a DSP engineer sees filterbanks and Fourier transforms.

    Use this table to connect the physical math to DJ booth practice, and discuss the open terminology with your instructor during a feedback session.

    | DJ Concept / Terminology | Domain | Physical DSP Implementation | Question to Validate with Instructor |
    |---|---|---|---|
    | **"3-Band RGB Waveform"** *(The Lows, Mids, & Highs)* | Time Domain (Colored Envelope) | **Illustrative 3-Band Butterworth IIR Filterbank + RMS Envelope Pooling** (not a verified reconstruction of Pioneer internals): Audio is split into Low (<250 Hz, Cyan), Mid (250–2500 Hz, Amber), and High (>=2500 Hz, Blue), followed by 256-sample RMS envelope pooling. *(Note: Because the blue band spans 2.5–22.05 kHz, a 16 kHz cutoff barely changes the blue height, making this broad-band view a poor tool for investigating high-frequency cutoffs).* | *"Does this 3-color envelope separation match what you look for when reading drops and breakdowns on a CDJ-3000 screen?"* |
    | *[DJ Term: Tonal Balance / Master EQ Curve]* | Frequency Domain (1D Curve) | **Welch Averaged Power Spectral Density (PSD)**: Multi-window Fourier analysis ($N=2048$, Hann window, 75% overlap) averaged over time and plotted on a musical log-frequency scale (20 Hz to 22.05 kHz). | *"When DJs or mastering engineers evaluate the overall tonal balance (sub-bass vs. treble tilt) of a track, what do you usually call this full-frequency curve?"* |
    | *[DJ Term: Spek View / Transcode Heatmap]* | Joint Time-Frequency Domain (2D Heatmap) | **2D Short-Time Fourier Transform (STFT)**: Sliding window FFT mapping time (x-axis) vs. frequency bins up to 22.05 kHz (y-axis), with color intensity indicating decibel power (dBFS). Clearly reveals horizontal cutoff voids. | *"Have you or fellow DJs used software like Spek to check for fake 320 kbps files? What term do you and other selectors use for this frequency-time heatmap?"* |
    """)
    return


if __name__ == "__main__":
    app.run()
