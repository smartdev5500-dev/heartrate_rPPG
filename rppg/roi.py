"""Face tracking and region-of-interest (ROI) extraction.

Three backends, tried in order of preference:

1. ``FaceLandmarkerROI`` — MediaPipe >= 1.0 "Tasks" API (the current one).
   Needs the ``face_landmarker.task`` model, auto-downloaded and cached by
   :mod:`rppg.assets` on first run.
2. ``FaceMeshROI`` — the legacy ``mp.solutions.face_mesh`` API for older
   MediaPipe installs (< 1.0), which bundled the model in the wheel.
3. ``HaarForeheadROI`` — OpenCV Haar cascade face detector with a fixed
   forehead box. No MediaPipe needed; the cascade XML is vendored in
   ``rppg/data/`` because OpenCV 5 wheels no longer ship it.

Both MediaPipe backends use the same 468/478-point face-mesh landmark
indexing, so the ROI polygon definitions are shared.

Each frame we average the pixel colour inside the ROI to get a single
(R, G, B) triple. That per-frame triple is the raw rPPG signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .assets import get_face_landmarker_model, get_haar_cascade_path

try:  # MediaPipe is preferred but optional.
    import mediapipe as mp

    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover - depends on the environment
    mp = None
    _HAS_MEDIAPIPE = False

# Does this MediaPipe expose the legacy solutions API (< 1.0)?
_HAS_MP_SOLUTIONS = _HAS_MEDIAPIPE and hasattr(mp, "solutions")
# Does it expose the Tasks API (0.10+ and all 1.x)?
_HAS_MP_TASKS = _HAS_MEDIAPIPE and hasattr(mp, "tasks")


# Face-mesh landmark indices for stable, well-perfused, usually-exposed
# skin. Identical for the legacy and Tasks APIs.
FOREHEAD = [67, 69, 109, 10, 338, 299, 297, 336, 9, 107]
# Left cheek (subject's left = image right).
LEFT_CHEEK = [330, 347, 346, 352, 411, 425, 266, 329]
# Right cheek.
RIGHT_CHEEK = [101, 118, 117, 123, 187, 205, 36, 100]


@dataclass
class ROIResult:
    """Per-frame ROI extraction output."""

    found: bool
    rgb: tuple[float, float, float] | None  # mean (R, G, B) inside the ROI
    # List of polygons (each an (K, 2) int array) to draw on the preview.
    polygons: list = field(default_factory=list)
    # Optional face bounding box (x, y, w, h) for the Eulerian stretch view.
    face_box: tuple[int, int, int, int] | None = None


def _poly_from_landmarks(landmarks, idxs, w: int, h: int) -> np.ndarray:
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs],
        dtype=np.float32,
    )
    return pts.astype(np.int32)


def _result_from_polys(frame_bgr: np.ndarray, polys: list) -> ROIResult:
    """Mask the polygons, average the colour, build the ROIResult."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        cv2.fillConvexPoly(mask, poly, 255)

    if cv2.countNonZero(mask) == 0:
        return ROIResult(found=False, rgb=None, polygons=polys)

    mean_bgr = cv2.mean(frame_bgr, mask=mask)  # (B, G, R, _)
    rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])

    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    x0 = int(np.clip(xs.min(), 0, w - 1))
    y0 = int(np.clip(ys.min(), 0, h - 1))
    x1 = int(np.clip(xs.max(), 0, w))
    y1 = int(np.clip(ys.max(), 0, h))
    face_box = (x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))
    return ROIResult(found=True, rgb=rgb, polygons=polys, face_box=face_box)


class FaceLandmarkerROI:
    """MediaPipe Tasks API (mediapipe >= 1.0): forehead + both cheeks."""

    def __init__(self) -> None:
        if not _HAS_MP_TASKS:
            raise RuntimeError("mediapipe Tasks API is not available")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = get_face_landmarker_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ROIResult:
        h, w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode needs strictly increasing timestamps in ms.
        if timestamp_ms is None or timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 33
        self._last_ts_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return ROIResult(found=False, rgb=None)

        lm = result.face_landmarks[0]
        polys = [
            _poly_from_landmarks(lm, FOREHEAD, w, h),
            _poly_from_landmarks(lm, LEFT_CHEEK, w, h),
            _poly_from_landmarks(lm, RIGHT_CHEEK, w, h),
        ]
        return _result_from_polys(frame_bgr, polys)


class FaceMeshROI:
    """Legacy MediaPipe solutions API (mediapipe < 1.0)."""

    def __init__(self) -> None:
        if not _HAS_MP_SOLUTIONS:
            raise RuntimeError("mediapipe legacy solutions API is not available")
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._mesh.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ROIResult:
        h, w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return ROIResult(found=False, rgb=None)

        lm = result.multi_face_landmarks[0].landmark
        polys = [
            _poly_from_landmarks(lm, FOREHEAD, w, h),
            _poly_from_landmarks(lm, LEFT_CHEEK, w, h),
            _poly_from_landmarks(lm, RIGHT_CHEEK, w, h),
        ]
        return _result_from_polys(frame_bgr, polys)


class HaarForeheadROI:
    """Fallback ROI: OpenCV Haar face detector + a fixed forehead box."""

    def __init__(self) -> None:
        cascade_path = str(get_haar_cascade_path())
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade from {cascade_path}")

    def close(self) -> None:  # symmetry with the MediaPipe trackers
        pass

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ROIResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return ROIResult(found=False, rgb=None)

        # Largest face.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Forehead box: horizontally centred, upper portion of the face.
        fx = int(x + 0.25 * w)
        fw = int(0.5 * w)
        fy = int(y + 0.10 * h)
        fh = int(0.20 * h)
        fx = max(fx, 0)
        fy = max(fy, 0)
        fw = min(fw, frame_bgr.shape[1] - fx)
        fh = min(fh, frame_bgr.shape[0] - fy)
        if fw <= 0 or fh <= 0:
            return ROIResult(found=False, rgb=None)

        patch = frame_bgr[fy : fy + fh, fx : fx + fw]
        mean_bgr = cv2.mean(patch)
        rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])

        poly = np.array(
            [[fx, fy], [fx + fw, fy], [fx + fw, fy + fh], [fx, fy + fh]],
            dtype=np.int32,
        )
        return ROIResult(
            found=True, rgb=rgb, polygons=[poly], face_box=(int(x), int(y), int(w), int(h))
        )


def create_roi_tracker(prefer_mediapipe: bool = True):
    """Create the best available ROI tracker.

    Returns a tuple (tracker, backend_name).
    """
    if prefer_mediapipe and _HAS_MEDIAPIPE:
        if _HAS_MP_SOLUTIONS:  # older mediapipe: model ships in the wheel
            try:
                return FaceMeshROI(), "mediapipe-facemesh"
            except Exception as exc:
                print(f"[warn] legacy FaceMesh unavailable ({exc}); trying Tasks API")
        if _HAS_MP_TASKS:
            try:
                return FaceLandmarkerROI(), "mediapipe-tasks"
            except Exception as exc:
                print(f"[warn] MediaPipe Tasks unavailable ({exc}); falling back to Haar")
    return HaarForeheadROI(), "opencv-haar"
