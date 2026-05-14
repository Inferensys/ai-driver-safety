from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.models import FramePacket


@dataclass(slots=True)
class ObjectObservation:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    provider: str = "unknown"

    def to_dict(self) -> dict[str, float | int | str | tuple[int, int, int, int]]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "provider": self.provider,
        }


class ObjectDetector(Protocol):
    provider: str

    def detect(self, packet: FramePacket) -> list[ObjectObservation]: ...


class NoopObjectDetector:
    provider = "none"

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        return []


class FixtureObjectDetector:
    provider = "fixture"

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        phase = packet.timestamp % 54
        if 38 <= phase < 45:
            h, w = packet.frame.shape[:2]
            return [
                ObjectObservation(
                    label="phone",
                    confidence=0.88,
                    bbox=(int(w * 0.63), int(h * 0.44), int(w * 0.09), int(h * 0.17)),
                    provider=self.provider,
                )
            ]
        return []


class OnnxObjectDetector:
    provider = "onnx"

    def __init__(self, model_path: Path, labels_path: Path, input_size: int = 640) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ONNX Runtime is not installed. Install with `pip install ai-driver-safety[onnx]`."
            ) from exc
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.input_size = input_size
        self.labels = _load_labels(labels_path)
        self.session = ort.InferenceSession(
            str(model_path), providers=ort.get_available_providers()
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        frame = packet.frame
        image = _letterbox(frame, self.input_size)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input_name: image})
        return _parse_yolo_like(
            outputs[0], frame.shape[1], frame.shape[0], self.labels, self.provider
        )


def create_object_detector(config: DriverSafetyConfig) -> ObjectDetector:
    obj_config = config.object_detector
    if not obj_config.enabled or obj_config.provider == "none":
        return NoopObjectDetector()
    if obj_config.provider == "fixture":
        return FixtureObjectDetector()
    if obj_config.provider == "onnx":
        return OnnxObjectDetector(Path(obj_config.model_path), Path(obj_config.labels_path))
    raise ValueError(f"Unsupported object detector provider: {obj_config.provider}")


def _load_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(__import__("cv2").resize(frame, (size, size)))


def _parse_yolo_like(
    output: np.ndarray,
    frame_width: int,
    frame_height: int,
    labels: list[str],
    provider: str,
    confidence_threshold: float = 0.35,
) -> list[ObjectObservation]:
    detections = np.squeeze(output)
    if detections.ndim == 1:
        detections = detections[None, :]
    observations: list[ObjectObservation] = []
    for row in detections:
        if row.shape[0] < 6:
            continue
        x_center, y_center, width, height = row[:4]
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        x = int((x_center - width / 2) * frame_width)
        y = int((y_center - height / 2) * frame_height)
        observations.append(
            ObjectObservation(
                label=label,
                confidence=confidence,
                bbox=(x, y, int(width * frame_width), int(height * frame_height)),
                provider=provider,
            )
        )
    return observations
