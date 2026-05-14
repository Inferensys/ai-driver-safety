from pathlib import Path

import cv2
import numpy as np

from driver_safety.config import load_config
from driver_safety.core.models import DriverState, FramePacket
from driver_safety.runtime.runner import analyze_video
from driver_safety.vision.landmarks import FaceObservation
from driver_safety.vision.object_detector import ObjectObservation
from driver_safety.vision.pipeline import DriverSafetyPipeline


def test_analyze_video_writes_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    _write_video(video)
    config = load_config(Path("configs/default.yaml"))
    config.runtime.max_frames = 40
    out = tmp_path / "run"
    artifacts = analyze_video(video, out, config)
    assert artifacts["events"].exists()
    assert artifacts["summary"].exists()
    assert artifacts["html"].exists()
    assert artifacts["csv"].exists()
    assert artifacts["annotated_video"].exists()


def test_missing_driver_view_is_labeled_distracted() -> None:
    config = load_config(Path("configs/default.yaml"))
    config.thresholds.missing_face_frames = 1
    pipeline = DriverSafetyPipeline(config, face_detector=_NoFaceDetector())
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    result = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))

    assert result.state == DriverState.DISTRACTED
    assert result.events[0].signal == "distracted"
    assert "attention not trackable" in result.events[0].message


def test_phone_use_is_temporally_gated() -> None:
    config = load_config(Path("configs/default.yaml"))
    config.thresholds.missing_face_frames = 999
    config.thresholds.phone_confidence = 0.5
    config.thresholds.phone_use_frames = 2
    pipeline = DriverSafetyPipeline(
        config,
        face_detector=_NoFaceDetector(),
        object_detector=_PhoneObjectDetector(),
    )
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    first = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))
    second = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.1, frame_index=1))

    assert not [event for event in first.events if event.signal == "phone_use"]
    assert [event for event in second.events if event.signal == "phone_use"]


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
    assert writer.isOpened()
    frame = np.full((180, 320, 3), 28, dtype=np.uint8)
    for _ in range(48):
        writer.write(frame)
    writer.release()


class _NoFaceDetector:
    provider = "test"

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        return []


class _PhoneObjectDetector:
    provider = "test"

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        return [
            ObjectObservation(
                label="cell phone",
                confidence=0.8,
                bbox=(120, 90, 32, 42),
                provider=self.provider,
            )
        ]
