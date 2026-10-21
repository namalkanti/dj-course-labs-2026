"""Unit tests verifying physical DSP properties of synthetic audio fixtures.

These tests serve as an educational reference for understanding signal properties:
- White noise: flat spectral density across low vs high bands.
- Chirp: monotonic instantaneous frequency progression across octaves.
- Drum loop: rhythmic transient placement at beat intervals & frequency separation.
- Lossy cutoff: brick-wall lowpass attenuation in stopband vs passband preservation.
"""

import unittest
import numpy as np
import scipy.signal as signal

from yourshot.synth import (
    generate_white_noise,
    generate_chirp,
    generate_drum_loop,
    apply_lossy_cutoff,
)


class TestAudioSynthesisFixtures(unittest.TestCase):

    def test_white_noise_spectral_flatness_and_level(self):
        """White noise should have flat power across frequency bands and bounded peak."""
        fs = 44100
        duration_sec = 2.0
        audio = generate_white_noise(duration_sec=duration_sec, fs=fs)

        # 1. Output length matches duration * fs
        self.assertEqual(len(audio), int(duration_sec * fs))

        # 2. Peak normalization bounded to 0.5 (-6 dBFS)
        peak = np.max(np.abs(audio))
        self.assertAlmostEqual(peak, 0.5, places=5)

        # 3. Spectral flatness: compare average power in low band (1-5 kHz)
        # to high band (15-19 kHz). True white noise has equal power in all linear bands.
        freqs, psd = signal.welch(audio, fs=fs, nperseg=2048)
        low_band_power = np.mean(psd[(freqs >= 1000) & (freqs <= 5000)])
        high_band_power = np.mean(psd[(freqs >= 15000) & (freqs <= 19000)])

        power_ratio_db = 10.0 * np.log10(high_band_power / low_band_power)
        # Power across linear bands should match within +/- 2 dB
        self.assertAlmostEqual(power_ratio_db, 0.0, delta=2.0)

    def test_chirp_frequency_progression(self):
        """Chirp should sweep monotonically from low frequency to high frequency."""
        fs = 44100
        duration_sec = 1.0
        audio = generate_chirp(duration_sec=duration_sec, fs=fs, f_start=50.0, f_end=20000.0)

        self.assertEqual(len(audio), int(duration_sec * fs))
        self.assertLessEqual(np.max(np.abs(audio)), 0.5)

        # Estimate local instantaneous frequency by counting zero crossings in early vs late windows
        early_window = audio[: int(0.1 * fs)]  # First 100 ms (starts near 50 Hz)
        late_window = audio[int(0.9 * fs) :]   # Last 100 ms (ends near 20 kHz)

        zero_crossings_early = np.sum(np.diff(np.signbit(early_window)))
        zero_crossings_late = np.sum(np.diff(np.signbit(late_window)))

        # High-frequency end must have dramatically more zero-crossings than the start
        self.assertGreater(zero_crossings_late, zero_crossings_early * 10)

    def test_drum_loop_rhythm_and_frequency_split(self):
        """Drum loop must have kick transients on beats and cymbal sizzle on offbeats."""
        fs = 44100
        bpm = 120.0  # 1 beat = 0.5 s, offbeat = 0.25 s
        duration_sec = 2.0  # Exactly 4 beats (1 full bar)
        audio = generate_drum_loop(duration_sec=duration_sec, bpm=bpm, fs=fs)

        self.assertEqual(len(audio), int(duration_sec * fs))
        self.assertAlmostEqual(np.max(np.abs(audio)), 0.7, places=5)

        # Inspect the first downbeat (kick @ t=0s) vs first offbeat (hi-hat @ t=0.25s)
        kick_segment = audio[: int(0.1 * fs)]
        hat_segment = audio[int(0.24 * fs) : int(0.30 * fs)]

        # Kick segment should be dominated by low-frequency energy (< 200 Hz)
        freqs_k, psd_k = signal.welch(kick_segment, fs=fs, nperseg=1024)
        sub_power = np.sum(psd_k[freqs_k < 200.0])
        total_k_power = np.sum(psd_k)
        self.assertGreater(sub_power / total_k_power, 0.7)  # Over 70% sub energy

        # Hi-hat segment should have dominant high-frequency energy (> 5 kHz)
        freqs_h, psd_h = signal.welch(hat_segment, fs=fs, nperseg=1024)
        high_power = np.sum(psd_h[freqs_h > 5000.0])
        total_h_power = np.sum(psd_h)
        self.assertGreater(high_power / total_h_power, 0.5)

    def test_apply_lossy_cutoff_brickwall_attenuation(self):
        """Lossy cutoff filter must cleanly attenuate the stopband by at least 40 dB."""
        fs = 44100
        noise = generate_white_noise(duration_sec=2.0, fs=fs)

        # 1. Nyquist bypass: cutoff >= Nyquist returns unmodified audio
        bypass = apply_lossy_cutoff(noise, cutoff_hz=22050.0, fs=fs)
        np.testing.assert_array_equal(bypass, noise)

        # 2. 16 kHz cutoff (simulating 128 kbps MP3 rolloff)
        cutoff_hz = 16000.0
        filtered = apply_lossy_cutoff(noise, cutoff_hz=cutoff_hz, fs=fs)

        freqs, psd = signal.welch(filtered, fs=fs, nperseg=2048)

        # Passband power (5-14 kHz) vs stopband power (18-21 kHz)
        passband_power = np.mean(psd[(freqs >= 5000) & (freqs <= 14000)])
        stopband_power = np.mean(psd[(freqs >= 18000) & (freqs <= 21000)])

        attenuation_db = 10.0 * np.log10(passband_power / stopband_power)
        # Order-10 Butterworth should yield >= 40 dB of separation
        self.assertGreater(attenuation_db, 40.0)


if __name__ == "__main__":
    unittest.main()
