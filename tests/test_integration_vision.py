"""Integration tests for the vision stack (ROI tracking + full pipeline).

These run the *complete* chain — face detection on real image data, a
synthetic pulsing "video", the rolling buffer, resampling, POS/green, FFT —
without any camera. They are skipped automatically when OpenCV, MediaPipe,
the landmark model, or the test image are unavailable, so the pure
signal-processing tests always run everywhere.

The test face image is MediaPipe's standard portrait test asset; it is
downloaded once into the model cache directory, not committed to the repo.
"""

from __future__ import annotations

import urllib.request

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rppg import process_buffer  # noqa: E402
from rppg.assets import cache_dir  # noqa: E402
from rppg.capture import SignalBuffer  # noqa: E402
from rppg.eulerian import EulerianMagnifier  # noqa: E402
from rppg.roi import HaarForeheadROI, create_roi_tracker  # noqa: E402
from rppg.ui import compose, draw_roi  # noqa: E402

PORTRAIT_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"
TRUE_BPM = 72.0


@pytest.fixture(scope="module")
def face_image() -> np.ndarray:
    """A real face image (cached download); skip if unavailable."""
    path = cache_dir() / "portrait.jpg"
    if not path.exists():
        try:
            urllib.request.urlretrieve(PORTRAIT_URL, path)  # noqa: S310
        except Exception as exc:  # no network
            pytest.skip(f"test image unavailable: {exc}")
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip("test image failed to decode")
    return cv2.resize(img, (512, 640))


def _synthetic_video_frames(base: np.ndarray, seed: int = 11):
    """Yield (t, frame) with the green channel pulsing at TRUE_BPM."""
    rng = np.random.default_rng(seed)
    freq = TRUE_BPM / 60.0
    basef = base.astype(np.float64)
    t = 0.0
    while t < 12.0:
        pulse = np.sin(2 * np.pi * freq * t)
        f = basef.copy()
        f[:, :, 1] -= 1.5 * pulse  # green (BGR index 1)
        f[:, :, 2] -= 0.5 * pulse
        f += rng.normal(0, 0.8, f.shape)
        yield t, np.clip(f, 0, 255).astype(np.uint8)
        t += rng.uniform(1 / 34, 1 / 22)  # irregular webcam timing


def _run_tracker_pipeline(tracker, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    buf = SignalBuffer(window_seconds=10.0)
    for t, frame in _synthetic_video_frames(base):
        r = tracker.process(frame, timestamp_ms=int(t * 1000))
        if r.found and r.rgb is not None:
            buf.append(t, r.rgb)
    assert len(buf) > 100, "tracker lost the face on most frames"
    return buf.arrays()


def test_haar_tracker_end_to_end(face_image):
    """Haar fallback: full pipeline recovers the synthetic pulse +-2 bpm."""
    tracker = HaarForeheadROI()
    try:
        ts, rgb = _run_tracker_pipeline(tracker, face_image)
    finally:
        tracker.close()
    for method in ("pos", "green"):
        est, _ = process_buffer(ts, rgb, fs=30.0, method=method)
        assert abs(est.bpm - TRUE_BPM) <= 2.0, f"{method}: got {est.bpm:.2f}"


def test_best_tracker_end_to_end(face_image):
    """Preferred tracker (MediaPipe if available): recovers the pulse."""
    try:
        tracker, backend = create_roi_tracker()
    except Exception as exc:
        pytest.skip(f"no tracker available: {exc}")
    try:
        ts, rgb = _run_tracker_pipeline(tracker, face_image)
    finally:
        tracker.close()
    est, _ = process_buffer(ts, rgb, fs=30.0, method="pos")
    assert abs(est.bpm - TRUE_BPM) <= 2.0, f"{backend}: got {est.bpm:.2f}"
    assert est.confidence > 0.5


def test_roi_detects_skin_tone(face_image):
    """Detected ROI colour should look like skin (R > B, mid-range)."""
    tracker = HaarForeheadROI()
    try:
        r = tracker.process(face_image)
    finally:
        tracker.close()
    assert r.found and r.rgb is not None
    red, green, blue = r.rgb
    assert red > blue, "skin should reflect more red than blue"
    assert 20 < green < 250


def test_ui_compose_renders(face_image):
    """The composed UI canvas has the expected geometry and content."""
    wave = np.sin(np.linspace(0, 12 * np.pi, 300))
    frame = face_image.copy()
    poly = np.array([[10, 10], [60, 10], [60, 40], [10, 40]], dtype=np.int32)
    draw_roi(frame, [poly], active=True)
    canvas = compose(
        frame, wave, 72.0, 15.0, 0.9, "pos", "test", acquiring=False, fps=30.0
    )
    assert canvas.shape[0] >= frame.shape[0]
    assert canvas.shape[1] > frame.shape[1]  # side panel present
    assert canvas.dtype == np.uint8
    # Acquiring mode must also render without error.
    canvas2 = compose(frame, None, 0.0, -np.inf, 0.0, "green", "test", acquiring=True)
    assert canvas2.shape == canvas.shape


def test_eulerian_amplifies_pulse(face_image):
    """The magnifier should amplify a small periodic colour oscillation."""
    mag = EulerianMagnifier(amplification=40.0, fs=30.0)
    basef = face_image.astype(np.float64)
    raw_means, mag_means = [], []
    for t in np.arange(0, 8, 1 / 30):
        f = basef.copy()
        f[:, :, 1] -= 1.5 * np.sin(2 * np.pi * 1.2 * t)
        frame = np.clip(f, 0, 255).astype(np.uint8)
        out = mag.process(frame)
        raw_means.append(frame[:, :, 1].mean())
        mag_means.append(out[:, :, 1].mean())
    # Ignore the filter's settling period.
    raw_amp = np.ptp(raw_means[60:])
    mag_amp = np.ptp(mag_means[60:])
    assert mag_amp > 5 * raw_amp, f"raw {raw_amp:.2f} -> magnified {mag_amp:.2f}"
