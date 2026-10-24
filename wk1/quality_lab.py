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

    ### 1.3 File Formats: WAV vs AIFF vs FLAC vs MP3
    Audio formats divide strictly into two categories: **lossless** and **lossy**.
    - **Lossless** formats preserve 100% of the original PCM audio samples. When decoded, the audio waveform is bit-for-bit identical to the studio master.
    - **Lossy** formats permanently discard audio data using psychoacoustic modeling to achieve smaller file sizes. Once discarded, that information is lost forever.

    | Format | Type | Compression | Metadata / Tagging Support | Club / CDJ Compatibility | Best Use Case |
    |---|---|---|---|---|---|
    | **WAV** | Lossless | None (Raw PCM) | **Poor**: ID3 tagging is non-standard. Hot cues, album art, and artist tags often get stripped when moving between Rekordbox USBs. | Universal across all CDJ models. | Studio recording, raw stem export. |
    | **AIFF** | Lossless | None (Raw PCM) | **Excellent**: Native ID3 tag chunk support. Artwork, cues, and metadata never get dropped. | Universal across all CDJ models. | **The touring DJ gold standard**. Bit-for-bit identical to WAV with bulletproof metadata. |
    | **FLAC** | Lossless | Compressed (~50% size, like a ZIP for audio) | **Excellent**: Vorbis comments preserve all track metadata and album art. | CDJ-2000NXS2, CDJ-3000, and Opus-Quad only. Older club gear (CDJ-900, CDJ-2000 Nexus) cannot play it. | Archival, home listening, modern festival stages. |
    | **MP3** | Lossy | Heavy (~80–90% size reduction) | **Excellent**: Full ID3v2 tag support. | Universal across all CDJ models. | Legitimate **320 kbps** is club-ready and saves USB space; **< 256 kbps** or rips degrade club sound. |

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


@app.cell
def load_dsp_modules():
    import io
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal
    import soundfile as sf
    from yourshot.synth import apply_lossy_cutoff

    return apply_lossy_cutoff, io, np, plt, sf


@app.cell
def load_plotting():
    from yourshot.spectral import (
        compute_welch_spectrum,
        compute_stft_spectrogram,
        compute_three_band_waveform,
    )
    from yourshot.plotting import (
        plot_welch_spectrum,
        plot_stft_spectrogram,
        plot_three_band_waveform,
    )

    return (
        compute_stft_spectrogram,
        compute_three_band_waveform,
        compute_welch_spectrum,
        plot_stft_spectrogram,
        plot_three_band_waveform,
        plot_welch_spectrum,
    )


@app.cell
def get_catalog_and_fs():
    from yourshot.audio_catalog import load_audio_catalog
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
            "Lossless / Passthrough (22.05 kHz Nyquist)": 22050.0,
            "~320 kbps MP3 (20.0 kHz shelf)": 20000.0,
            "~192 kbps MP3 (18.5 kHz cutoff)": 18500.0,
            "~128 kbps MP3 (16.0 kHz brick-wall)": 16000.0,
        },
        value="~128 kbps MP3 (16.0 kHz brick-wall)",
        label="Simulated Codec Cutoff",
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
                "##### Rekordbox 3-Band Waveform (DJ View)\n"
                "_Time-domain energy envelope split into Rekordbox 3-band crossovers: "
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

    return fig_to_image, render_analysis_tabs


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


@app.cell
def exercise_2_codec_cutoff_simulation(
    apply_lossy_cutoff,
    audio_ex2,
    codec_cutoff,
    compute_stft_spectrogram,
    compute_three_band_waveform,
    compute_welch_spectrum,
    fig_to_image,
    fs,
    mo,
    plot_stft_spectrogram,
    plot_three_band_waveform,
    plot_welch_spectrum,
    signal_selector_ex2,
):
    _cutoff_hz = codec_cutoff.value if codec_cutoff.value < 22000.0 else None
    _filtered = apply_lossy_cutoff(audio_ex2, cutoff_hz=codec_cutoff.value, fs=fs)

    # 1. Audio Players Side-by-Side
    _player_orig = mo.audio(audio_ex2, rate=fs)
    _player_filt = mo.audio(_filtered, rate=fs)

    _orig_player_card = mo.vstack([
        mo.md("**Pristine (Lossless Master)**"),
        _player_orig,
    ], gap=0.5)

    _filt_player_card = mo.vstack([
        mo.md(f"**Codec Filtered ({codec_cutoff.value / 1000.0:.1f} kHz)**"),
        _player_filt,
    ], gap=0.5)

    _players_row = mo.hstack([_orig_player_card, _filt_player_card], justify="space-around", gap=2)

    # 2. Side-by-Side Forensic Shelf Comparison (10 kHz to Nyquist)
    _spec_orig = compute_welch_spectrum(audio_ex2, fs=fs)
    _shelf_orig = _spec_orig.crop(f_min=10000.0, normalize_peak=True)
    _fig_shelf_orig, _ = plot_welch_spectrum(
        _shelf_orig,
        log_freq=False,
        title="Pristine High-Frequency Shelf (10–22 kHz)",
        cutoff_marker_hz=None,
    )
    _fig_shelf_orig.tight_layout()

    _spec_filt = compute_welch_spectrum(_filtered, fs=fs)
    _shelf_filt = _spec_filt.crop(f_min=10000.0, normalize_peak=True)
    _fig_shelf_filt, _ = plot_welch_spectrum(
        _shelf_filt,
        log_freq=False,
        title=f"Filtered Shelf ({codec_cutoff.value / 1000.0:.1f} kHz Cutoff)",
        cutoff_marker_hz=_cutoff_hz,
    )
    _fig_shelf_filt.tight_layout()

    _shelf_side_by_side = mo.hstack(
        [fig_to_image(_fig_shelf_orig), fig_to_image(_fig_shelf_filt)],
        justify="space-between",
        gap=1.5,
    )

    # 3. Side-by-Side Waveform Comparison
    _wf_orig = compute_three_band_waveform(audio_ex2, fs=fs)
    _fig_wf_orig, _ = plot_three_band_waveform(_wf_orig, title="Pristine 3-Band Waveform")
    _fig_wf_orig.tight_layout()

    _wf_filt = compute_three_band_waveform(_filtered, fs=fs)
    _fig_wf_filt, _ = plot_three_band_waveform(_wf_filt, title=f"Filtered 3-Band Waveform ({codec_cutoff.value / 1000.0:.1f} kHz)")
    _fig_wf_filt.tight_layout()

    _wf_side_by_side = mo.hstack(
        [fig_to_image(_fig_wf_orig), fig_to_image(_fig_wf_filt)],
        justify="space-between",
        gap=1.5,
    )

    # 4. Side-by-Side Spectrogram Comparison
    _stft_orig = compute_stft_spectrogram(audio_ex2, fs=fs)
    _fig_stft_orig, _ = plot_stft_spectrogram(_stft_orig, title="Pristine 2D Spectrogram")
    _fig_stft_orig.tight_layout()

    _stft_filt = compute_stft_spectrogram(_filtered, fs=fs)
    _fig_stft_filt, _ = plot_stft_spectrogram(_stft_filt, title=f"Filtered 2D Spectrogram ({codec_cutoff.value / 1000.0:.1f} kHz)", cutoff_marker_hz=_cutoff_hz)
    _fig_stft_filt.tight_layout()

    _stft_side_by_side = mo.hstack(
        [fig_to_image(_fig_stft_orig), fig_to_image(_fig_stft_filt)],
        justify="space-between",
        gap=1.5,
    )

    _comparison_tabs = mo.ui.tabs({
        "High-Freq Shelf (Cliff Inspector)": mo.vstack([
            mo.md("_Compare the top octave (10–22 kHz). Observe how lossy filtering creates an unnatural steep cliff._"),
            _shelf_side_by_side,
        ]),
        "CDJ 3-Band Waveforms": mo.vstack([
            mo.md("_Compare amplitude envelopes. Notice how the blue high-frequency layer shrinks after the cutoff._"),
            _wf_side_by_side,
        ]),
        "2D Spectrograms": mo.vstack([
            mo.md("_Compare time-frequency heatmaps. Notice the horizontal cutoff void above the green cutoff line._"),
            _stft_side_by_side,
        ]),
    })

    _controls_row = mo.hstack([signal_selector_ex2, codec_cutoff], justify="start", gap=2)

    output_2 = mo.vstack([
        mo.md("#### Exercise 2: Codec Cutoff Filtering (Before vs. After Comparison)"),
        _controls_row,
        _players_row,
        _comparison_tabs,
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
    apply_lossy_cutoff,
    audio_uploader,
    compute_stft_spectrogram,
    compute_three_band_waveform,
    compute_welch_spectrum,
    fig_to_image,
    io,
    mo,
    mono_audio,
    plot_stft_spectrogram,
    plot_three_band_waveform,
    plot_welch_spectrum,
    sf,
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
                "The lab will render its CDJ 3-band waveform, 20 Hz–22 kHz power spectrum, "
                "STFT spectrogram, and forensic high-frequency shelf, with optional side-by-side codec filtering._"
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

        # 1. Convert audio slices to 16-bit PCM for snappy players
        def _to_pcm_audio(sig):
            buf = io.BytesIO()
            sf.write(buf, sig, upload_fs, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return mo.audio(buf.read())

        _player_orig = _to_pcm_audio(analyzed_orig)
        _player_filt = _to_pcm_audio(analyzed_filt)

        _players_row = mo.hstack([
            mo.vstack([mo.md(f"**As Uploaded (`{audio_uploader.name}`)**"), _player_orig], gap=0.5),
            mo.vstack([mo.md(f"**With Simulated Cutoff ({upload_cutoff.value / 1000.0:.1f} kHz)**"), _player_filt], gap=0.5),
        ], justify="space-around", gap=2)

        # 2. Side-by-Side Forensic Shelf Comparison (10 kHz to Nyquist)
        _spec_orig = compute_welch_spectrum(analyzed_orig, fs=upload_fs)
        _shelf_orig = _spec_orig.crop(f_min=10000.0, normalize_peak=True)
        _fig_shelf_orig, _ = plot_welch_spectrum(
            _shelf_orig,
            log_freq=False,
            title="As Uploaded Shelf (10–Nyquist)",
            cutoff_marker_hz=None,
        )
        _fig_shelf_orig.tight_layout()

        _spec_filt = compute_welch_spectrum(analyzed_filt, fs=upload_fs)
        _shelf_filt = _spec_filt.crop(f_min=10000.0, normalize_peak=True)
        _fig_shelf_filt, _ = plot_welch_spectrum(
            _shelf_filt,
            log_freq=False,
            title=f"Filtered Shelf ({upload_cutoff.value / 1000.0:.1f} kHz)",
            cutoff_marker_hz=_cutoff_hz,
        )
        _fig_shelf_filt.tight_layout()

        _shelf_side_by_side = mo.hstack(
            [fig_to_image(_fig_shelf_orig), fig_to_image(_fig_shelf_filt)],
            justify="space-between",
            gap=1.5,
        )

        # 3. Side-by-Side 3-Band Waveforms
        _wf_orig = compute_three_band_waveform(analyzed_orig, fs=upload_fs)
        _fig_wf_orig, _ = plot_three_band_waveform(_wf_orig, title="As Uploaded Waveform")
        _fig_wf_orig.tight_layout()

        _wf_filt = compute_three_band_waveform(analyzed_filt, fs=upload_fs)
        _fig_wf_filt, _ = plot_three_band_waveform(_wf_filt, title=f"Filtered Waveform ({upload_cutoff.value / 1000.0:.1f} kHz)")
        _fig_wf_filt.tight_layout()

        _wf_side_by_side = mo.hstack(
            [fig_to_image(_fig_wf_orig), fig_to_image(_fig_wf_filt)],
            justify="space-between",
            gap=1.5,
        )

        # 4. Side-by-Side 2D Spectrograms
        _stft_orig = compute_stft_spectrogram(analyzed_orig, fs=upload_fs)
        _fig_stft_orig, _ = plot_stft_spectrogram(_stft_orig, title="As Uploaded Spectrogram")
        _fig_stft_orig.tight_layout()

        _stft_filt = compute_stft_spectrogram(analyzed_filt, fs=upload_fs)
        _fig_stft_filt, _ = plot_stft_spectrogram(_stft_filt, title=f"Filtered Spectrogram ({upload_cutoff.value / 1000.0:.1f} kHz)", cutoff_marker_hz=_cutoff_hz)
        _fig_stft_filt.tight_layout()

        _stft_side_by_side = mo.hstack(
            [fig_to_image(_fig_stft_orig), fig_to_image(_fig_stft_filt)],
            justify="space-between",
            gap=1.5,
        )

        _comparison_tabs = mo.ui.tabs({
            "High-Freq Shelf (Cliff Inspector)": mo.vstack([
                mo.md(
                    "_Inspect the region between 15 kHz and 22 kHz. "
                    "If your uploaded file already exhibits a steep drop at **16 kHz** or **18.5 kHz**, "
                    "it is an upscaled lossy transcode regardless of its file format!_"
                ),
                _shelf_side_by_side,
            ]),
            "CDJ 3-Band Waveforms": mo.vstack([
                mo.md("_Compare amplitude envelopes. Notice how the blue high-frequency layer contracts after cutoff._"),
                _wf_side_by_side,
            ]),
            "2D Spectrograms": mo.vstack([
                mo.md("_Compare time-frequency heatmaps to inspect transient preservation and cutoff voids._"),
                _stft_side_by_side,
            ]),
        })

        file_info = mo.md(
            f"**File:** `{audio_uploader.name}` &nbsp;|&nbsp; "
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
            _players_row,
            _comparison_tabs,
        ], gap=1.4)

    output_3
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


if __name__ == "__main__":
    app.run()
