from __future__ import annotations

from pathlib import Path
from time import monotonic

import cv2

from driver_safety.core.models import FramePacket


class VideoFrameSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0) or 24.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def __iter__(self) -> VideoFrameSource:
        return self

    def __next__(self) -> FramePacket:
        ok, frame = self.capture.read()
        if not ok:
            self.close()
            raise StopIteration
        frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = frame_index / self.fps
        return FramePacket(
            frame=frame,
            timestamp=timestamp,
            frame_index=frame_index,
            source_id=str(self.path),
            fps=self.fps,
        )

    def close(self) -> None:
        self.capture.release()


class WebcamFrameSource:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index {index}")
        self.started = monotonic()
        self.frame_index = -1
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0) or 24.0

    def __iter__(self) -> WebcamFrameSource:
        return self

    def __next__(self) -> FramePacket:
        ok, frame = self.capture.read()
        if not ok:
            self.close()
            raise StopIteration
        self.frame_index += 1
        return FramePacket(
            frame=frame,
            timestamp=monotonic() - self.started,
            frame_index=self.frame_index,
            source_id=f"webcam:{self.index}",
            fps=self.fps,
        )

    def close(self) -> None:
        self.capture.release()
