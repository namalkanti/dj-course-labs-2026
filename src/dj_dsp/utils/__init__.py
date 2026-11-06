"""Shared utilities for digital signal processing and Marimo labs."""

from dj_dsp.utils.dsp import (
    bpm_range_to_lag_bounds,
    bpm_to_lag_frames,
    calculate_normalized_ratio_confidence,
    classify_meter_from_autocorr,
    find_peak_in_lag_window,
    frames_to_times,
    lag_frames_to_bpm,
    log_compression,
    normalize_peak,
    smooth_and_normalize_autocorr,
)
from dj_dsp.utils.lab import fig_to_image, parse_audio_upload

__all__ = [
    "bpm_range_to_lag_bounds",
    "bpm_to_lag_frames",
    "calculate_normalized_ratio_confidence",
    "classify_meter_from_autocorr",
    "fig_to_image",
    "find_peak_in_lag_window",
    "frames_to_times",
    "lag_frames_to_bpm",
    "log_compression",
    "normalize_peak",
    "parse_audio_upload",
    "smooth_and_normalize_autocorr",
]
