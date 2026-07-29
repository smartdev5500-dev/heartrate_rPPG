"""Face tracking and region-of-interest (ROI) extraction.

Prefers MediaPipe Face Mesh for accurate, stable forehead + cheek regions.
If MediaPipe is unavailable (install trouble on some platforms), falls back
to OpenCV's Haar cascade face detector with a fixed forehead box.

Each frame we average the pixel colour inside the ROI to get a single
(R, G, B) triple. That per-frame triple is the raw rPPG signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

try:  # MediaPipe is preferred but optional.
    import mediapipe as mp

    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover - depends on the environment
    _HAS_MEDIAPIPE = False


@dataclass
class ROIResult:
    """Per-frame ROI extraction output."""

    found: bool
    rgb: tuple[float, float, float] | None  # mean (R, G, B) inside the ROI
    # List of polygons (each an (K, 2) int array) to draw on the preview.
    polygons: list = field(default_factory=list)
    # Optional face bounding box (x, y, w, h) for the Eulerian stretch view.
    face_box: tuple[int, int, int, int] | None = None


class FaceMeshROI:
    """MediaPipe Face Mesh based ROI: forehead + both cheeks."""

    # Landmark indices for stable, well-perfused, usually-exposed skin.
    # Forehead centre patch.
    FOREHEAD = [67, 69, 109, 10, 338, 299, 297, 336, 9, 107]
    # Left cheek (subject's left = image right).
    LEFT_CHEEK = [330, 347, 346, 352, 411, 425, 266, 329]
    # Right cheek.
    RIGHT_CHEEK = [101, 118, 117, 123, 187, 205, 36, 100]

    def __init__(self) -> None:
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe is not installed")
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._mesh.close()

    @staticmethod
    def _poly(landmarks, idxs, w: int, h: int) -> np.ndarray:
        pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs],
            dtype=np.float32,
        )
        return pts.astype(np.int32)

    def process(self, frame_bgr: np.ndarray) -> ROIResult:
        h, w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return ROIResult(found=False, rgb=None)

        lm = result.multi_face_landmarks[0].landmark
        polys = [
            self._poly(lm, self.FOREHEAD, w, h),
            self._poly(lm, self.LEFT_CHEEK, w, h),
            self._poly(lm, self.RIGHT_CHEEK, w, h),
        ]

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            cv2.fillConvexPoly(mask, poly, 255)

        if cv2.countNonZero(mask) == 0:
            return ROIResult(found=False, rgb=None, polygons=polys)

        mean_bgr = cv2.mean(frame_bgr, mask=mask)  # (B, G, R, _)
        rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])

        xs = np.concatenate([p[:, 0] for p in polys])
        ys = np.concatenate([p[:, 1] for p in polys])
        face_box = (
            int(max(xs.min(), 0)),
            int(max(ys.min(), 0)),
            int(min(xs.max() - xs.min(), w)),
            int(min(ys.max() - ys.min(), h)),
        )
        return ROIResult(found=True, rgb=rgb, polygons=polys, face_box=face_box)


class HaarForeheadROI:
    """Fallback ROI: OpenCV Haar face detector + a fixed forehead box."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("failed to load Haar cascade for face detection")

    def close(self) -> None:  # symmetry with FaceMeshROI
        pass

    def process(self, frame_bgr: np.ndarray) -> ROIResult:
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
            found=True, rgb=rgb, polygons=[poly], face_box=(x, y, w, h)
        )


def create_roi_tracker(prefer_mediapipe: bool = True):
    """Create the best available ROI tracker.

    Returns a tuple (tracker, backend_name).
    """
    if prefer_mediapipe and _HAS_MEDIAPIPE:
        try:
            return FaceMeshROI(), "mediapipe"
        except Exception:
            pass
    return HaarForeheadROI(), "opencv-haar"
