"""Webcam capture and the rolling signal buffer.

Frame timing from a real webcam is irregular, so every frame is stamped
with ``time.perf_counter()`` at grab time rather than assuming a fixed fps.
The rolling buffer keeps the last ``window_seconds`` of (t, R, G, B).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class SignalBuffer:
    """Rolling buffer of (timestamp, R, G, B) samples, trimmed by time."""

    def __init__(self, window_seconds: float = 10.0) -> None:
        self.window_seconds = window_seconds
        self._t: deque[float] = deque()
        self._rgb: deque[tuple[float, float, float]] = deque()

    def append(self, t: float, rgb: tuple[float, float, float]) -> None:
        self._t.append(t)
        self._rgb.append(rgb)
        cutoff = t - self.window_seconds
        while self._t and self._t[0] < cutoff:
            self._t.popleft()
            self._rgb.popleft()

    def __len__(self) -> int:
        return len(self._t)

    def duration(self) -> float:
        if len(self._t) < 2:
            return 0.0
        return self._t[-1] - self._t[0]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps (N,), rgb (N, 3)) as numpy arrays."""
        return (
            np.asarray(self._t, dtype=np.float64),
            np.asarray(self._rgb, dtype=np.float64),
        )

    def clear(self) -> None:
        self._t.clear()
        self._rgb.clear()


class Webcam:
    """Thin wrapper over ``cv2.VideoCapture`` with perf_counter stamps."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        import cv2

        self._cv2 = cv2
        # On macOS the first-ever run triggers the OS camera-permission
        # dialog, and the very first open attempt fails while it is shown.
        # Retry for a while so clicking "Allow" lets this run succeed
        # instead of forcing a restart.
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            print(
                "[info] webcam not available yet - if the OS is asking for "
                "camera permission, click Allow (retrying for 15 s)..."
            )
            deadline = time.monotonic() + 15.0
            while not self._cap.isOpened() and time.monotonic() < deadline:
                time.sleep(1.0)
                self._cap.release()
                self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open webcam at index {index}. If this is macOS, "
                "grant camera access in System Settings > Privacy & Security "
                "> Camera (enable your terminal app), then run again. "
                "A different camera can be selected with --camera N."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> tuple[float, "np.ndarray"] | None:
        """Grab one frame. Returns (timestamp_seconds, frame_bgr) or None."""
        ok, frame = self._cap.read()
        t = time.perf_counter()
        if not ok or frame is None:
            return None
        return t, frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
