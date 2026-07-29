"""Self-drawn OpenCV UI: video panel + scrolling waveform + BPM readout.

The waveform is drawn with cv2 line segments rather than matplotlib to
avoid event-loop lag in a real-time capture loop.
"""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colours in BGR.
_BG = (18, 18, 18)
_PANEL = (30, 30, 30)
_ACCENT = (120, 220, 120)
_ACCENT_DIM = (90, 120, 90)
_TEXT = (235, 235, 235)
_TEXT_DIM = (140, 140, 140)
_WARN = (60, 170, 240)


def draw_roi(frame_bgr: np.ndarray, polygons, active: bool = True) -> None:
    """Overlay the tracked ROI polygons on the video frame in-place."""
    colour = _ACCENT if active else _ACCENT_DIM
    for poly in polygons:
        cv2.polylines(frame_bgr, [poly.reshape(-1, 1, 2)], True, colour, 2, cv2.LINE_AA)


def _draw_waveform(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, wave: np.ndarray) -> None:
    """Draw a normalized waveform into the box (x0, y0, w, h)."""
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), _PANEL, -1)
    # Midline.
    mid = y0 + h // 2
    cv2.line(canvas, (x0, mid), (x0 + w, mid), (55, 55, 55), 1, cv2.LINE_AA)

    if wave is None or len(wave) < 2:
        return

    wave = np.asarray(wave, dtype=np.float64)
    amp = np.max(np.abs(wave))
    if amp < 1e-9:
        return
    norm = wave / amp  # -1..1

    n = len(norm)
    xs = x0 + (np.arange(n) / (n - 1) * (w - 1)).astype(np.int32)
    ys = (mid - norm * (h * 0.45)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, _ACCENT, 2, cv2.LINE_AA)


def _draw_confidence_bar(
    canvas: np.ndarray, x0: int, y0: int, w: int, h: int, confidence: float, snr_db: float
) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), _PANEL, -1)
    fill_w = int(np.clip(confidence, 0.0, 1.0) * w)
    bar_colour = _ACCENT if confidence >= 0.5 else _WARN
    if fill_w > 0:
        cv2.rectangle(canvas, (x0, y0), (x0 + fill_w, y0 + h), bar_colour, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    label = f"SNR {snr_db:4.1f} dB" if np.isfinite(snr_db) else "SNR  -- dB"
    cv2.putText(canvas, label, (x0, y0 - 8), _FONT, 0.5, _TEXT_DIM, 1, cv2.LINE_AA)


def compose(
    frame_bgr: np.ndarray,
    wave: np.ndarray,
    bpm: float,
    snr_db: float,
    confidence: float,
    method: str,
    backend: str,
    acquiring: bool,
    fps: float = 0.0,
) -> np.ndarray:
    """Compose the full window: video left, plot + readouts right."""
    vh, vw = frame_bgr.shape[:2]
    panel_w = 460
    out_h = max(vh, 480)
    out_w = vw + panel_w
    canvas = np.full((out_h, out_w, 3), _BG, dtype=np.uint8)

    # Video on the left.
    canvas[0:vh, 0:vw] = frame_bgr

    px = vw + 20
    pw = panel_w - 40

    # BPM readout.
    if acquiring:
        bpm_text = "-- "
        bpm_colour = _TEXT_DIM
        status = "acquiring..."
    else:
        bpm_text = f"{bpm:0.0f}"
        bpm_colour = _TEXT
        status = "beats / min"
    cv2.putText(canvas, bpm_text, (px, 90), _FONT, 3.0, bpm_colour, 4, cv2.LINE_AA)
    cv2.putText(canvas, status, (px, 125), _FONT, 0.6, _TEXT_DIM, 1, cv2.LINE_AA)

    # Confidence bar.
    _draw_confidence_bar(canvas, px, 150, pw, 18, confidence, snr_db)

    # Waveform.
    _draw_waveform(canvas, px, 210, pw, 180, wave)
    cv2.putText(canvas, "filtered pulse", (px, 205), _FONT, 0.5, _TEXT_DIM, 1, cv2.LINE_AA)

    # Footer: method / backend / fps / keys.
    y = out_h - 70
    cv2.putText(
        canvas, f"method: {method.upper()}", (px, y), _FONT, 0.55, _ACCENT, 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, f"tracker: {backend}", (px, y + 22), _FONT, 0.5, _TEXT_DIM, 1, cv2.LINE_AA
    )
    if fps > 0:
        cv2.putText(
            canvas, f"{fps:0.1f} fps", (px + 250, y), _FONT, 0.5, _TEXT_DIM, 1, cv2.LINE_AA
        )
    cv2.putText(
        canvas,
        "[m] method  [e] magnify  [q] quit",
        (px, out_h - 20),
        _FONT,
        0.5,
        _TEXT_DIM,
        1,
        cv2.LINE_AA,
    )
    return canvas
