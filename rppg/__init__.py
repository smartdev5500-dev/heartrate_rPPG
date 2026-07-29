"""rPPG webcam heart-rate monitor.

The signal-processing core (``rppg.signal_processing``) depends only on
numpy and scipy and is independently unit-testable. Camera/UI modules pull
in OpenCV and (optionally) MediaPipe.
"""

from .signal_processing import (
    HeartRateEstimate,
    bandpass_filter,
    compute_pulse,
    detrend_signal,
    estimate_heart_rate,
    green_pulse,
    pos_pulse,
    process_buffer,
    resample_uniform,
)

__all__ = [
    "HeartRateEstimate",
    "bandpass_filter",
    "compute_pulse",
    "detrend_signal",
    "estimate_heart_rate",
    "green_pulse",
    "pos_pulse",
    "process_buffer",
    "resample_uniform",
]

__version__ = "1.0.0"
