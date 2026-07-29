"""Unit tests for the rPPG signal-processing core.

These prove the signal-processing math is correct without a camera, by
feeding synthetic RGB signals with a known pulse frequency through the
estimator and checking the recovered BPM.

Run:  python -m pytest
"""

from __future__ import annotations

import numpy as np
import pytest

from rppg.signal_processing import (
    bandpass_filter,
    detrend_signal,
    estimate_heart_rate,
    green_pulse,
    pos_pulse,
    process_buffer,
    resample_uniform,
)


def _make_synthetic_rgb(
    bpm: float,
    duration_s: float = 12.0,
    fs: float = 30.0,
    noise: float = 0.4,
    seed: int = 0,
):
    """Synthetic (timestamps, rgb) with a pulse at ``bpm``.

    The pulse appears in all three channels (as with real skin) but is
    strongest in green. Each channel has a distinct DC offset (skin tone)
    plus additive Gaussian noise. Timestamps are jittered to mimic
    irregular webcam frame timing.
    """
    rng = np.random.default_rng(seed)
    freq = bpm / 60.0
    n = int(duration_s * fs)

    # Irregular timestamps: nominal grid + jitter, still monotonic.
    base = np.arange(n) / fs
    jitter = rng.normal(0.0, 0.15 / fs, size=n)
    timestamps = np.cumsum(np.abs(np.diff(base, prepend=base[0]) + jitter))
    timestamps += 0.001  # avoid a zero-length first step

    pulse = np.sin(2 * np.pi * freq * timestamps)

    # Channel gains: green carries the strongest pulse.
    r = 130 + 1.0 * pulse + rng.normal(0, noise, n)
    g = 110 + 3.0 * pulse + rng.normal(0, noise, n)
    b = 90 + 0.6 * pulse + rng.normal(0, noise, n)

    rgb = np.column_stack([r, g, b])
    return timestamps, rgb


@pytest.mark.parametrize("bpm", [60.0, 72.0, 90.0, 120.0])
def test_green_method_recovers_bpm(bpm):
    """Green-channel baseline recovers a known BPM within +-2 bpm."""
    timestamps, rgb = _make_synthetic_rgb(bpm, seed=1)
    estimate, _ = process_buffer(timestamps, rgb, fs=30.0, method="green")
    assert abs(estimate.bpm - bpm) <= 2.0, f"got {estimate.bpm:.2f}, want {bpm}"


@pytest.mark.parametrize("bpm", [60.0, 72.0, 90.0, 120.0])
def test_pos_method_recovers_bpm(bpm):
    """POS recovers a known BPM within +-2 bpm on the same synthetic data."""
    timestamps, rgb = _make_synthetic_rgb(bpm, seed=2)
    estimate, _ = process_buffer(timestamps, rgb, fs=30.0, method="pos")
    assert abs(estimate.bpm - bpm) <= 2.0, f"got {estimate.bpm:.2f}, want {bpm}"


def test_estimator_on_pure_green_oscillation():
    """The spec's core check: green oscillates at 1.2 Hz (72 bpm) + noise."""
    timestamps, rgb = _make_synthetic_rgb(72.0, noise=0.5, seed=3)
    estimate, wave = process_buffer(timestamps, rgb, fs=30.0, method="green")
    assert abs(estimate.bpm - 72.0) <= 2.0
    assert np.isfinite(estimate.snr_db)
    assert len(wave) > 0


def test_high_snr_for_clean_signal():
    """A clean pulse should be much more confident than pure noise."""
    timestamps, clean = _make_synthetic_rgb(72.0, noise=0.2, seed=4)
    clean_est, _ = process_buffer(timestamps, clean, fs=30.0, method="green")

    rng = np.random.default_rng(5)
    n = len(timestamps)
    noise_rgb = np.column_stack(
        [110 + rng.normal(0, 2, n) for _ in range(3)]
    )
    noise_est, _ = process_buffer(timestamps, noise_rgb, fs=30.0, method="green")

    assert clean_est.snr_db > noise_est.snr_db
    assert clean_est.confidence > noise_est.confidence


def test_resample_uniform_grid_spacing():
    """Resampling produces a uniform grid at the requested rate."""
    t = np.array([0.0, 0.1, 0.25, 0.31, 0.5])
    v = np.sin(2 * np.pi * 1.0 * t)
    ut, uv = resample_uniform(t, v, fs=30.0)
    diffs = np.diff(ut)
    assert np.allclose(diffs, diffs[0])
    assert np.isclose(diffs[0], 1.0 / 30.0)
    assert len(ut) == len(uv)


def test_resample_multichannel():
    t = np.linspace(0, 1, 20)
    rgb = np.column_stack([t, 2 * t, 3 * t])
    ut, urgb = resample_uniform(t, rgb, fs=15.0)
    assert urgb.shape[1] == 3
    # Linear ramps should stay linear after interpolation.
    assert np.allclose(urgb[:, 1], 2 * urgb[:, 0], atol=1e-6)
    assert np.allclose(urgb[:, 2], 3 * urgb[:, 0], atol=1e-6)


def test_bandpass_rejects_out_of_band():
    """A 0.2 Hz drift (12 bpm, below band) should be strongly attenuated."""
    fs = 30.0
    t = np.arange(0, 12, 1 / fs)
    low_freq = np.sin(2 * np.pi * 0.2 * t)  # out of band
    in_band = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm, in band
    filtered = bandpass_filter(low_freq + in_band, fs)
    # In-band component should dominate the filtered output.
    assert np.std(filtered) < np.std(low_freq + in_band)
    est = estimate_heart_rate(filtered, fs)
    assert abs(est.bpm - 72.0) <= 2.0


def test_detrend_removes_linear_trend():
    x = np.linspace(0, 10, 100) + np.sin(np.linspace(0, 20, 100))
    d = detrend_signal(x)
    # After detrending, the mean slope should be ~0.
    slope = np.polyfit(np.arange(len(d)), d, 1)[0]
    assert abs(slope) < 1e-6


def test_pos_and_green_shapes():
    _, rgb = _make_synthetic_rgb(72.0)
    _, uniform = resample_uniform(*_make_synthetic_rgb(72.0), fs=30.0)
    p = pos_pulse(uniform, 30.0)
    g = green_pulse(uniform)
    assert p.shape[0] == uniform.shape[0]
    assert g.shape[0] == uniform.shape[0]


def test_short_buffer_does_not_crash():
    """Estimator returns a benign result on a too-short buffer."""
    est = estimate_heart_rate(np.array([0.0, 1.0, 0.0]), fs=30.0)
    assert est.bpm == 0.0
