import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def init_marimo():
    import marimo as mo
    return (mo,)


@app.cell
def render_header(mo):
    mo.md(
        r"""
        # Week 1 Lab: Audio Quality Verification, Codec Analysis & Transcode Detection

        *An interactive digital signal processing lab analyzing lossy vs. lossless audio compression, psychoacoustic filtering, and automated detection of upsampled/transcoded files.*
        """
    )
    return


@app.cell
def render_foundations(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def render_translation_table(mo):
    mo.md(
        r"""
        ### 1.5 DJ-to-DSP Translation Table

        | DJ / Club Jargon | What the DJ Hears | The Physical DSP Phenomenon |
        |---|---|---|
        | **"Muffled / Boxy / No Air"** | Cymbals, snares, and vocals lack sparkle and space. | **Brick-wall lowpass cutoff** ($f_c \le 16\text{ kHz}$) applied by 128 kbps encoders. |
        | **"Mushy Kick / Blurry Transients"** | Kick drums lose punch/snap; basslines feel flabby. | **MDCT temporal pre-echo**: Quantization noise spreads backward ~20 ms before the transient inside long MDCT blocks. |
        | **"Watery / Swishy / Phasey Cymbals"** | Hi-hats sound like they are playing through a phaser or underwater. | **Intensity Stereo + High-Band Bit Starvation**: High subbands are coarsely quantized or blended to mono, causing erratic phase variance frame-to-frame. |
        | **"Fake 320 / YouTube Rip"** | Track reports high bitrate or WAV, but sounds lifeless compared to other tracks in the mix. | **Inflated Container Header**: File was transcoded from low-bitrate lossy, leaving zero energy above 16 kHz despite the 320k/WAV header. |
        | **"Falls Apart at High Volume"** | Sounds fine on laptop/headphones, but harsh and fatiguing on the club PA. | **Psychoacoustic Masking Breakdown**: Masking models assume domestic listening (~70 dB SPL). At 100+ dB SPL club levels, hearing contours flatten and hidden quantization noise becomes glaringly audible. |
        """
    )
    return


@app.cell
def load_dsp_modules():
    import numpy as np
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    import soundfile as sf
    import mutagen
    return mutagen, np, plt, sf, signal


if __name__ == "__main__":
    app.run()
