"""Pure signal-processing core for rPPG heart-rate estimation.

This module intentionally depends only on numpy and scipy so that the
signal-processing math can be unit-tested without a camera, MediaPipe, or
OpenCV. Everything camera- or UI-related lives in other modules.

The pipeline implemented here:

    irregular (t, R, G, B)  --resample-->  uniform grid
                            --POS or green-->  pulse waveform
                            --detrend + bandpass-->  clean waveform
                            --Hann + zero-padded FFT-->  spectrum
                            --peak + parabolic interp-->  BPM
                            --peak/harmonic power ratio-->  SNR

References
----------
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
"Algorithmic Principles of Remote PPG." IEEE Transactions on Biomedical
Engineering, 64(7), 1479-1491.  (The POS algorithm.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

# Heart-rate band: 0.7-4.0 Hz == 42-240 bpm.
HR_BAND_LOW_HZ = 0.7
HR_BAND_HIGH_HZ = 4.0


def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an irregularly-sampled signal onto a uniform time grid.

    Real webcams do not deliver frames at a constant rate, so the raw
    (timestamp, value) samples are unevenly spaced in time. Feeding those
    straight into an FFT biases the frequency estimate. We linearly
    interpolate onto a uniform grid at ``fs`` Hz first.

    Parameters
    ----------
    timestamps : (N,) array of monotonically increasing seconds.
    values : (N,) or (N, C) array of samples aligned with ``timestamps``.
    fs : target sampling rate in Hz.

    Returns
    -------
    (uniform_t, uniform_values) where ``uniform_t`` is 1-D and
    ``uniform_values`` matches the channel layout of ``values``.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if timestamps.ndim != 1:
        raise ValueError("timestamps must be 1-D")
    if timestamps.shape[0] < 2:
        raise ValueError("need at least 2 samples to resample")

    t0, t1 = timestamps[0], timestamps[-1]
    duration = t1 - t0
    n = int(np.floor(duration * fs)) + 1
    if n < 2:
        raise ValueError("buffer too short for the requested sampling rate")
    uniform_t = t0 + np.arange(n) / fs

    if values.ndim == 1:
        uniform_values = np.interp(uniform_t, timestamps, values)
    else:
        uniform_values = np.column_stack(
            [np.interp(uniform_t, timestamps, values[:, c]) for c in range(values.shape[1])]
        )
    return uniform_t, uniform_values


def pos_pulse(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Extract a pulse waveform using POS (Plane-Orthogonal-to-Skin).

    Implements the overlap-add form from Wang et al. 2017. For every
    sliding window of length ~1.6 s the RGB samples are temporally
    normalized, projected onto a plane orthogonal to the skin-tone
    direction, the two projections are combined with a tuning factor, and
    the result is overlap-added into the output.

    Parameters
    ----------
    rgb : (N, 3) array of per-frame mean R, G, B on a uniform time grid.
    fs : sampling rate in Hz of that grid.

    Returns
    -------
    (N,) pulse waveform.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape (N, 3)")

    n = rgb.shape[0]
    h = np.zeros(n, dtype=np.float64)
    win = max(int(np.round(1.6 * fs)), 2)

    # Projection matrix onto the plane orthogonal to the skin-tone axis.
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

    for m in range(0, n - win + 1):
        block = rgb[m : m + win, :]  # (win, 3)
        mean_c = np.mean(block, axis=0)
        # Guard against a fully black / zero ROI.
        mean_c = np.where(mean_c == 0.0, 1.0, mean_c)
        cn = block / mean_c  # temporal normalization

        s = cn @ proj.T  # (win, 2)
        s1, s2 = s[:, 0], s[:, 1]
        std2 = np.std(s2)
        alpha = (np.std(s1) / std2) if std2 > 1e-12 else 0.0
        hwin = s1 + alpha * s2
        h[m : m + win] += hwin - np.mean(hwin)

    return h


def green_pulse(rgb: np.ndarray) -> np.ndarray:
    """Simple baseline: the (negated) mean-green channel.

    Green light is absorbed most strongly by hemoglobin, so the green
    channel carries the strongest raw pulse signal. We negate so that a
    rise in blood volume (which *darkens* green) reads as a positive peak,
    matching POS's polarity convention. The sign does not affect the FFT
    magnitude, but keeps the plotted waveforms consistent.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape (N, 3)")
    g = rgb[:, 1]
    return -(g - np.mean(g))


def detrend_signal(x: np.ndarray) -> np.ndarray:
    """Remove slow drift with a linear detrend."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return x - np.mean(x) if x.size else x
    return scipy_signal.detrend(x, type="linear")


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float = HR_BAND_LOW_HZ,
    high: float = HR_BAND_HIGH_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass over the heart-rate band."""
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    if not (0 < low_n < high_n < 1):
        # Sampling rate too low for this band; return the detrended signal.
        return x
    b, a = scipy_signal.butter(order, [low_n, high_n], btype="band")
    # filtfilt needs a minimum length relative to the filter order.
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.shape[0] <= padlen:
        return x
    return scipy_signal.filtfilt(b, a, x)


@dataclass
class HeartRateEstimate:
    """Result of a heart-rate estimation over one buffer."""

    bpm: float
    snr_db: float
    confidence: float  # 0..1, a squashed SNR suitable for a UI bar
    freqs_hz: np.ndarray  # spectrum frequency axis (for plotting)
    power: np.ndarray  # spectrum power aligned with freqs_hz
    peak_hz: float


def _parabolic_interp(y: np.ndarray, k: int) -> tuple[float, float]:
    """Parabolic (quadratic) peak interpolation around integer index ``k``.

    Returns (offset, value) where ``offset`` is the sub-bin shift in [-0.5,
    0.5] added to ``k`` for the true peak location.
    """
    if k <= 0 or k >= len(y) - 1:
        return 0.0, float(y[k])
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-15:
        return 0.0, float(y1)
    offset = 0.5 * (y0 - y2) / denom
    offset = float(np.clip(offset, -0.5, 0.5))
    value = y1 - 0.25 * (y0 - y2) * offset
    return offset, float(value)


def estimate_heart_rate(
    pulse: np.ndarray,
    fs: float,
    low: float = HR_BAND_LOW_HZ,
    high: float = HR_BAND_HIGH_HZ,
    zero_pad_factor: int = 4,
) -> HeartRateEstimate:
    """Estimate BPM and SNR from a (already band-limited) pulse waveform.

    Steps: Hann window, zero-padded FFT for spectral resolution, find the
    peak within the HR band, refine with parabolic interpolation, then
    measure an SNR as the power near the peak and its first harmonic
    divided by the remaining in-band power.
    """
    pulse = np.asarray(pulse, dtype=np.float64)
    n = pulse.shape[0]
    if n < 4:
        return HeartRateEstimate(0.0, -np.inf, 0.0, np.array([0.0]), np.array([0.0]), 0.0)

    pulse = pulse - np.mean(pulse)
    window = np.hanning(n)
    windowed = pulse * window

    nfft = int(2 ** np.ceil(np.log2(n * max(zero_pad_factor, 1))))
    spectrum = np.fft.rfft(windowed, n=nfft)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return HeartRateEstimate(0.0, -np.inf, 0.0, freqs, power, 0.0)

    band_idx = np.where(band)[0]
    local_peak = band_idx[np.argmax(power[band_idx])]
    offset, _ = _parabolic_interp(power, local_peak)
    df = freqs[1] - freqs[0]
    peak_hz = freqs[local_peak] + offset * df
    peak_hz = float(np.clip(peak_hz, low, high))
    bpm = peak_hz * 60.0

    snr_db, confidence = _compute_snr(freqs, power, peak_hz, low, high)
    return HeartRateEstimate(
        bpm=bpm,
        snr_db=snr_db,
        confidence=confidence,
        freqs_hz=freqs,
        power=power,
        peak_hz=peak_hz,
    )


def _compute_snr(
    freqs: np.ndarray,
    power: np.ndarray,
    peak_hz: float,
    low: float,
    high: float,
    half_width_hz: float = 0.2,
) -> tuple[float, float]:
    """SNR = (power near peak + power near first harmonic) / rest of band.

    A clean pulse concentrates its energy at the fundamental and its
    harmonic; noise spreads across the band. High ratio => trustworthy BPM.
    """
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return -np.inf, 0.0

    def near(center: float) -> np.ndarray:
        return band & (np.abs(freqs - center) <= half_width_hz)

    signal_mask = near(peak_hz)
    harmonic = 2.0 * peak_hz
    if harmonic <= high:
        signal_mask = signal_mask | near(harmonic)

    signal_power = np.sum(power[signal_mask])
    noise_power = np.sum(power[band & ~signal_mask])
    if noise_power <= 1e-15:
        snr_linear = np.inf
    else:
        snr_linear = signal_power / noise_power

    if np.isinf(snr_linear):
        snr_db = 40.0
    elif snr_linear <= 1e-15:
        snr_db = -np.inf
    else:
        snr_db = 10.0 * np.log10(snr_linear)

    # Squash to 0..1 for a UI confidence bar. ~0 dB -> low, ~12 dB -> high.
    confidence = float(np.clip((snr_db) / 12.0, 0.0, 1.0))
    return float(snr_db), confidence


def compute_pulse(rgb: np.ndarray, fs: float, method: str = "pos") -> np.ndarray:
    """Dispatch to the selected pulse-extraction method."""
    if method == "pos":
        return pos_pulse(rgb, fs)
    if method == "green":
        return green_pulse(rgb)
    raise ValueError(f"unknown method: {method!r}")


def process_buffer(
    timestamps: np.ndarray,
    rgb: np.ndarray,
    fs: float = 30.0,
    method: str = "pos",
    low: float = HR_BAND_LOW_HZ,
    high: float = HR_BAND_HIGH_HZ,
) -> tuple[HeartRateEstimate, np.ndarray]:
    """Full pipeline from raw irregular (t, RGB) to a heart-rate estimate.

    Returns (estimate, filtered_pulse) where ``filtered_pulse`` is the
    clean band-limited waveform suitable for plotting.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)

    _, uniform_rgb = resample_uniform(timestamps, rgb, fs)
    pulse = compute_pulse(uniform_rgb, fs, method=method)
    pulse = detrend_signal(pulse)
    filtered = bandpass_filter(pulse, fs, low=low, high=high)
    estimate = estimate_heart_rate(filtered, fs, low=low, high=high)
    return estimate, filtered
