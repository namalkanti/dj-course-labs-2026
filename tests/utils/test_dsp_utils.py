"""Unit tests for pure DSP numerical utilities."""

import numpy as np
import pytest
from dj_dsp.utils.dsp import (
    bpm_range_to_lag_bounds,
    bpm_to_lag_frames,
    calculate_normalized_ratio_confidence,
    find_peak_in_lag_window,
    frames_to_times,
    lag_frames_to_bpm,
    log_compression,
    normalize_peak,
    smooth_and_normalize_autocorr,
)


def test_bpm_to_lag_frames_exact():
    """Verify BPM to lag conversion matches mathematical expectation."""
    # 120 BPM, fs=44100, hop=512 -> frame_rate=86.1328 frames/sec
    # 1 beat = 60 / 120 = 0.5 sec -> 0.5 * 86.1328 = 43.066 -> round to 43 frames
    lag = bpm_to_lag_frames(120.0, fs=44100, hop_length=512)
    assert lag == 43


def test_bpm_to_lag_frames_invalid():
    """Verify invalid BPM raises ValueError."""
    with pytest.raises(ValueError):
        bpm_to_lag_frames(0.0, fs=44100, hop_length=512)
    with pytest.raises(ValueError):
        bpm_to_lag_frames(-10.0, fs=44100, hop_length=512)


def test_lag_frames_to_bpm_roundtrip():
    """Verify roundtrip conversion between lag and BPM."""
    frame_rate = 44100.0 / 512.0
    lag = 43
    bpm = lag_frames_to_bpm(lag, frame_rate)
    recovered_lag = bpm_to_lag_frames(bpm, fs=44100, hop_length=512)
    assert recovered_lag == lag


def test_bpm_range_to_lag_bounds():
    """Verify inverse ordering: higher BPM results in smaller lag."""
    frame_rate = 86.1328
    min_lag, max_lag = bpm_range_to_lag_bounds(60.0, 180.0, frame_rate)
    assert min_lag < max_lag
    assert min_lag == int(round((60.0 * frame_rate) / 180.0))
    assert max_lag == int(round((60.0 * frame_rate) / 60.0))


def test_frames_to_times():
    """Verify time calculation from frame indices."""
    times = frames_to_times(num_frames=5, hop_length=512, fs=44100)
    assert len(times) == 5
    assert times[0] == 0.0
    assert np.isclose(times[1], 512 / 44100)


def test_normalize_peak():
    """Verify peak normalization to 0 dBFS and negative headroom."""
    signal = np.array([-0.5, 0.2, 0.8, -0.4])
    normalized = normalize_peak(signal, headroom_db=0.0)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)

    # -6 dB headroom is approximately 0.501 (-6 dBFS)
    headroom_sig = normalize_peak(signal, headroom_db=-6.0)
    assert np.isclose(np.max(np.abs(headroom_sig)), 10.0 ** (-6.0 / 20.0))


def test_normalize_peak_silent():
    """Verify silent signal remains all zeros without dividing by zero."""
    silent = np.zeros(10)
    assert np.all(normalize_peak(silent) == 0.0)


def test_log_compression():
    """Verify log compression strictly increases and maps 0 to 0."""
    mag = np.array([0.0, 1.0, 10.0])
    compressed = log_compression(mag, factor=100.0)
    assert compressed[0] == 0.0
    assert compressed[1] < compressed[2]
    assert np.isclose(compressed[1], np.log1p(100.0))


def test_find_peak_in_lag_window():
    """Verify peak window lookup finds max in window and respects bounds."""
    autocorr = np.array([0.1, 0.2, 0.9, 0.3, 0.1])
    # Target index 2 with tolerance 1 covers indices [1, 2, 3] -> peak 0.9
    peak = find_peak_in_lag_window(autocorr, target_lag=2, tolerance_frames=1)
    assert peak == 0.9


def test_calculate_normalized_ratio_confidence():
    """Verify confidence mapping: ratio 1.0 gives 0.0, extreme ratios approach 1.0."""
    assert calculate_normalized_ratio_confidence(1.0) == 0.0
    assert calculate_normalized_ratio_confidence(0.0) == 0.0
    assert np.isclose(calculate_normalized_ratio_confidence(3.0), (3 - 1) / (3 + 1))
    assert calculate_normalized_ratio_confidence(100.0) > 0.98
