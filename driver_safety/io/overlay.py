from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from driver_safety.core.models import DriverState, ProcessedFrame

Array = NDArray[Any]

STATE_COLORS = {
    DriverState.ATTENTIVE: (76, 190, 118),
    DriverState.EYES_CLOSED: (36, 174, 222),
    DriverState.DROWSY: (38, 64, 230),
    DriverState.YAWNING: (36, 174, 222),
    DriverState.DISTRACTED: (42, 42, 238),
    DriverState.PHONE_USE: (42, 42, 238),
}

LOGO_PATHS = (
    Path("docs/logo-overlay.png"),
    Path(__file__).resolve().parents[2] / "docs" / "logo-overlay.png",
)


class AnnotatedVideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {self.path}")

    def write(self, frame: Array) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def draw_overlay(processed: ProcessedFrame) -> Array:
    frame = processed.packet.frame.copy()
    h, w = frame.shape[:2]
    state_color = STATE_COLORS.get(processed.state, (255, 255, 255))
    _draw_panel(frame, 12, 12, 318, 136, alpha=0.72)
    _draw_brand(frame, 26, 25)
    cv2.putText(
        frame,
        processed.state.value.replace("_", " ").upper(),
        (28, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        state_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Risk {processed.risk_score:.2f}  {processed.latency_ms:.1f} ms",
        (28, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (212, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if processed.face_bbox:
        x, y, bw, bh = processed.face_bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), state_color, 2)
    for point in processed.landmarks[:: max(1, len(processed.landmarks) // 36 or 1)]:
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (110, 224, 255), -1)
    for obj in processed.objects:
        x, y, bw, bh = obj["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (55, 65, 235), 2)
        cv2.putText(
            frame,
            str(obj["label"]),
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (55, 65, 235),
            2,
            cv2.LINE_AA,
        )
    _draw_signal_bars(frame, processed, w, h)
    for idx, event in enumerate(processed.events[:3]):
        y = h - 90 + idx * 24
        cv2.putText(
            frame,
            event.message,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return cast(Array, frame)


def _draw_panel(frame: Array, x: int, y: int, w: int, h: int, *, alpha: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 22, 23), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_brand(frame: Array, x: int, y: int) -> None:
    logo = _load_logo()
    if logo is None:
        cv2.putText(
            frame,
            "AI DRIVER SAFETY",
            (28, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (246, 248, 248),
            2,
            cv2.LINE_AA,
        )
        return
    logo_h, logo_w = logo.shape[:2]
    scale = min(216 / logo_w, 44 / logo_h, 1.0)
    if scale < 1.0:
        logo = cv2.resize(
            logo,
            (int(logo_w * scale), int(logo_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    _blend_rgba(frame, logo, x, y)


@lru_cache(maxsize=1)
def _load_logo() -> Array | None:
    for path in LOGO_PATHS:
        if path.exists():
            logo = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if logo is not None and logo.shape[2] == 4:
                return cast(Array, logo)
    return None


def _blend_rgba(frame: Array, overlay: Array, x: int, y: int) -> None:
    h, w = overlay.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    if x >= frame_w or y >= frame_h:
        return
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    if w <= 0 or h <= 0:
        return
    logo_rgb = overlay[:h, :w, :3].astype(np.float32)
    alpha = overlay[:h, :w, 3:4].astype(np.float32) / 255.0
    region = frame[y : y + h, x : x + w].astype(np.float32)
    frame[y : y + h, x : x + w] = (alpha * logo_rgb + (1.0 - alpha) * region).astype(np.uint8)


def _draw_signal_bars(frame: Array, processed: ProcessedFrame, width: int, height: int) -> None:
    labels = ["eyes_closed", "drowsy", "yawning", "distracted", "phone_use"]
    start_x = max(20, width - 238)
    start_y = 24
    _draw_panel(frame, start_x - 12, start_y - 12, 226, 168, alpha=0.68)
    for idx, label in enumerate(labels):
        value = min(1.0, max(0.0, processed.signals.get(label, 0.0)))
        y = start_y + idx * 28
        cv2.putText(
            frame,
            label.replace("_", " "),
            (start_x, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (216, 222, 222),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (start_x + 106, y), (start_x + 196, y + 12), (60, 66, 68), -1)
        cv2.rectangle(
            frame,
            (start_x + 106, y),
            (start_x + 106 + int(90 * value), y + 12),
            _signal_color(label, value),
            -1,
        )


def _signal_color(label: str, value: float) -> tuple[int, int, int]:
    if value < 0.45:
        return (62, 197, 124)
    if label == "distracted":
        return (42, 42, 238)
    return (46, 167, 235)
