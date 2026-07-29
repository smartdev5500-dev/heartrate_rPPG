"""Stretch goal: simplified real-time Eulerian video magnification.

This is deliberately isolated from the core heart-rate pipeline so a bug
here can never break the main app. It temporally bandpasses each pixel's
colour inside the face ROI over a short history and amplifies the result,
making the subtle per-beat flush visible.

This is a lightweight, causal approximation of:
    Wu et al. (2012), "Eulerian Video Magnification for Revealing Subtle
    Changes in the World", ACM Transactions on Graphics.

We skip the spatial Laplacian pyramid and instead apply a first-order IIR
temporal bandpass per pixel on a small fixed-size ROI, which is cheap
enough to run every frame.
"""

from __future__ import annotations

import cv2
import numpy as np


class EulerianMagnifier:
    """Per-pixel temporal bandpass + amplification on a fixed-size ROI.

    The temporal bandpass is the difference of two first-order IIR
    low-pass filters (a cheap causal approximation of an ideal bandpass in
    the heart-rate band), applied independently per pixel and channel.
    """

    def __init__(
        self,
        size: tuple[int, int] = (160, 160),
        low_hz: float = 0.7,
        high_hz: float = 3.0,
        amplification: float = 50.0,
        fs: float = 15.0,
    ) -> None:
        self.size = size
        self.amplification = amplification
        self._lowpass_slow: np.ndarray | None = None
        self._lowpass_fast: np.ndarray | None = None
        # Convert cutoff frequencies to per-frame IIR coefficients.
        self._alpha_slow = self._alpha(low_hz, fs)
        self._alpha_fast = self._alpha(high_hz, fs)

    @staticmethod
    def _alpha(cutoff_hz: float, fs: float) -> float:
        # Standard one-pole low-pass smoothing factor.
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        return dt / (rc + dt)

    def reset(self) -> None:
        self._lowpass_slow = None
        self._lowpass_fast = None

    def process(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return an amplified version of ``roi_bgr`` (same size)."""
        resized = cv2.resize(roi_bgr, self.size).astype(np.float32)

        if self._lowpass_slow is None:
            self._lowpass_slow = resized.copy()
            self._lowpass_fast = resized.copy()
            return roi_bgr

        # Two one-pole low-pass filters at different cutoffs.
        self._lowpass_slow += self._alpha_slow * (resized - self._lowpass_slow)
        self._lowpass_fast += self._alpha_fast * (resized - self._lowpass_fast)

        # Bandpass = fast lowpass - slow lowpass.
        bandpassed = self._lowpass_fast - self._lowpass_slow

        amplified = resized + self.amplification * bandpassed
        amplified = np.clip(amplified, 0, 255).astype(np.uint8)
        return cv2.resize(amplified, (roi_bgr.shape[1], roi_bgr.shape[0]))
