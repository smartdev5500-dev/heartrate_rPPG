"""Model / asset management for the face trackers.

Two assets can be needed at runtime:

- ``face_landmarker.task`` — the MediaPipe Tasks face-landmark model,
  required by MediaPipe >= 1.0 (the legacy ``mp.solutions`` API bundled its
  model inside the wheel; the Tasks API does not). It is ~3.7 MB and is
  downloaded once to a local cache directory, then reused.

- ``haarcascade_frontalface_default.xml`` — the OpenCV Haar face detector
  used by the fallback tracker. OpenCV 5 wheels stopped bundling the
  cascade XMLs, so a copy is vendored in ``rppg/data/`` (Apache-2.0, from
  the OpenCV project).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_PACKAGE_DIR = Path(__file__).resolve().parent


def cache_dir() -> Path:
    """Local cache directory for downloaded models (override: RPPG_CACHE_DIR)."""
    override = os.environ.get("RPPG_CACHE_DIR")
    if override:
        path = Path(override)
    else:
        base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        path = Path(base) / "rppg-heartrate"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_face_landmarker_model() -> Path:
    """Return the path to face_landmarker.task, downloading it if needed."""
    path = cache_dir() / "face_landmarker.task"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    print(f"[info] downloading face landmark model (~3.7 MB) to {path} ...")
    tmp = path.with_suffix(".task.part")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)  # noqa: S310 - fixed https URL
    tmp.replace(path)
    print("[info] model downloaded")
    return path


def get_haar_cascade_path() -> Path:
    """Return a usable Haar frontal-face cascade path.

    Prefers the copy bundled with OpenCV (present in 4.x wheels), falls
    back to the copy vendored in this package (needed for OpenCV 5 wheels,
    which dropped the XMLs).
    """
    try:
        import cv2

        bundled = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if bundled.exists():
            return bundled
    except Exception:
        pass
    vendored = _PACKAGE_DIR / "data" / "haarcascade_frontalface_default.xml"
    if vendored.exists():
        return vendored
    raise FileNotFoundError(
        "haarcascade_frontalface_default.xml not found (neither bundled with "
        "OpenCV nor vendored in rppg/data)"
    )
