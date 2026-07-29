"""Real-time webcam heart-rate monitor (rPPG).

Reads your pulse from a webcam using remote photoplethysmography: each
heartbeat slightly changes skin colour as blood flows through it, and the
camera picks up that tiny shift. No contact sensors.

Run:
    python main.py

Keys:
    m   toggle pulse method (POS <-> green-channel baseline)
    e   toggle the Eulerian magnification view (stretch feature)
    q / Esc   quit
"""

from __future__ import annotations

import argparse
import sys
import time

if sys.version_info < (3, 9):
    raise SystemExit(
        f"This app needs Python 3.9 or newer; you are running "
        f"{sys.version_info.major}.{sys.version_info.minor}. "
        "Install a current Python from https://python.org and re-create the "
        "virtual environment."
    )

import cv2
import numpy as np

from rppg import process_buffer
from rppg.capture import SignalBuffer, Webcam
from rppg.eulerian import EulerianMagnifier
from rppg.roi import create_roi_tracker
from rppg.ui import compose, draw_roi

WINDOW_NAME = "rPPG Heart-Rate Monitor"

# Analysis parameters.
BUFFER_SECONDS = 10.0
RESAMPLE_FS = 30.0
MIN_SECONDS_TO_ESTIMATE = 5.0
# Below this SNR the BPM is greyed out and shown as "acquiring...".
SNR_ACQUIRE_THRESHOLD_DB = 3.0
# How often to recompute the estimate (it changes slowly; every frame is
# wasted CPU that lowers the capture frame rate).
ESTIMATE_INTERVAL_S = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam rPPG heart-rate monitor")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--method",
        choices=["pos", "green"],
        default="pos",
        help="initial pulse-extraction method",
    )
    p.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="force the OpenCV Haar fallback instead of MediaPipe",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tracker, backend = create_roi_tracker(prefer_mediapipe=not args.no_mediapipe)
    print(f"[info] ROI backend: {backend}")

    buffer = SignalBuffer(window_seconds=BUFFER_SECONDS)
    method = args.method
    magnifier = EulerianMagnifier()
    show_magnify = False

    # Smoothed estimate for a steady readout.
    smoothed_bpm: float | None = None
    last_estimate = None
    last_wave = None

    fps_ema = 0.0
    last_frame_t = None
    last_estimate_t = -float("inf")

    try:
        cam = Webcam(args.camera, args.width, args.height)
    except RuntimeError as exc:
        tracker.close()
        print(f"[error] {exc}")
        raise SystemExit(1) from None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with cam:
        while True:
            grabbed = cam.read()
            if grabbed is None:
                print("[warn] dropped frame")
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            t, frame = grabbed
            frame = cv2.flip(frame, 1)  # mirror for a natural preview

            # FPS estimate (for display only; analysis uses real timestamps).
            if last_frame_t is not None:
                dt = t - last_frame_t
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            last_frame_t = t

            roi = tracker.process(frame, timestamp_ms=int(t * 1000.0))
            if roi.found and roi.rgb is not None:
                buffer.append(t, roi.rgb)

            if roi.polygons:
                draw_roi(frame, roi.polygons, active=roi.found)

            # Optional Eulerian magnification overlaid on the face box.
            # Deliberately applied AFTER the ROI outlines are drawn: the
            # magnifier amplifying the drawn overlay produces a vivid
            # psychedelic effect that makes the mode visually distinct and
            # fun to toggle. (Signal quality is unaffected — the ROI colour
            # is sampled from the clean frame before any drawing.)
            if show_magnify and roi.face_box is not None:
                x, y, w, h = roi.face_box
                x, y = max(x, 0), max(y, 0)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                if w > 10 and h > 10:
                    patch = frame[y : y + h, x : x + w]
                    frame[y : y + h, x : x + w] = magnifier.process(patch)

            # Estimate heart rate once we have enough signal. The estimate
            # only meaningfully changes as new samples span the buffer, so
            # recompute a few times a second instead of every frame — the
            # saved CPU goes to capture/tracking frame rate.
            acquiring = True
            bpm = 0.0
            snr_db = -np.inf
            confidence = 0.0
            if (
                buffer.duration() >= MIN_SECONDS_TO_ESTIMATE
                and len(buffer) >= 32
                and t - last_estimate_t >= ESTIMATE_INTERVAL_S
            ):
                last_estimate_t = t
                ts, rgb = buffer.arrays()
                try:
                    estimate, wave = process_buffer(
                        ts, rgb, fs=RESAMPLE_FS, method=method
                    )
                    last_estimate = estimate
                    last_wave = wave
                except Exception as exc:  # keep the UI alive on any math hiccup
                    print(f"[warn] estimation error: {exc}")

            if last_estimate is not None:
                snr_db = last_estimate.snr_db
                confidence = last_estimate.confidence
                if np.isfinite(snr_db) and snr_db >= SNR_ACQUIRE_THRESHOLD_DB:
                    acquiring = False
                    if smoothed_bpm is None:
                        smoothed_bpm = last_estimate.bpm
                    else:
                        smoothed_bpm = 0.8 * smoothed_bpm + 0.2 * last_estimate.bpm
                    bpm = smoothed_bpm

            canvas = compose(
                frame,
                last_wave,
                bpm,
                snr_db,
                confidence,
                method,
                backend,
                acquiring,
                fps=fps_ema,
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                method = "green" if method == "pos" else "pos"
                smoothed_bpm = None
                print(f"[info] method -> {method}")
            elif key == ord("e"):
                show_magnify = not show_magnify
                magnifier.reset()
                print(f"[info] magnify -> {show_magnify}")

    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
