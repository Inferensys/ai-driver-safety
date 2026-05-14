from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.models import FramePacket

Array = NDArray[Any]


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


class OnnxObjectDetector:
    provider = "onnx"

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        *,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ONNX Runtime is not installed. Install with `pip install ai-driver-safety[onnx]`."
            ) from exc
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.labels = _load_labels(labels_path)
        self.session = ort.InferenceSession(
            str(model_path), providers=ort.get_available_providers()
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_size = _input_size(model_input.shape, fallback=input_size)

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        frame = packet.frame
        image = _letterbox(frame, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input_name: image})
        return _parse_yolo_like(
            outputs[0],
            frame.shape[1],
            frame.shape[0],
            self.labels,
            self.provider,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            input_size=self.input_size,
        )


def create_object_detector(config: DriverSafetyConfig) -> ObjectDetector:
    obj_config = config.object_detector
    if not obj_config.enabled or obj_config.provider == "none":
        return NoopObjectDetector()
    if obj_config.provider == "onnx":
        return OnnxObjectDetector(
            Path(obj_config.model_path),
            Path(obj_config.labels_path),
            confidence_threshold=obj_config.confidence_threshold,
            iou_threshold=obj_config.iou_threshold,
        )
    raise ValueError(f"Unsupported object detector provider: {obj_config.provider}")


def _load_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _letterbox(frame: Array, size: int) -> Array:
    return np.asarray(cv2.resize(frame, (size, size)))


def _input_size(shape: list[Any], *, fallback: int) -> int:
    for value in reversed(shape):
        if isinstance(value, int) and value > 0:
            return value
    return fallback


def _parse_yolo_like(
    output: Array,
    frame_width: int,
    frame_height: int,
    labels: list[str],
    provider: str,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    input_size: int = 640,
) -> list[ObjectObservation]:
    detections = _detection_rows(output)
    candidates: list[tuple[ObjectObservation, float, tuple[float, float, float, float]]] = []
    for row in detections:
        if row.shape[0] < 6:
            continue
        x_center, y_center, width, height = [float(value) for value in row[:4]]
        objectness, class_scores = _split_scores(row, labels)
        class_id = int(np.argmax(class_scores))
        confidence = float(objectness * class_scores[class_id])
        if confidence < confidence_threshold:
            continue
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        x1, y1, x2, y2 = _scale_box(
            x_center,
            y_center,
            width,
            height,
            frame_width=frame_width,
            frame_height=frame_height,
            input_size=input_size,
        )
        bbox = (
            int(round(x1)),
            int(round(y1)),
            int(round(max(0.0, x2 - x1))),
            int(round(max(0.0, y2 - y1))),
        )
        candidates.append(
            (
                ObjectObservation(label=label, confidence=confidence, bbox=bbox, provider=provider),
                confidence,
                (x1, y1, x2, y2),
            )
        )
    return [candidate[0] for candidate in _nms(candidates, iou_threshold=iou_threshold)]


def _detection_rows(output: Array) -> Array:
    detections = np.squeeze(output)
    if detections.ndim == 1:
        detections = detections[None, :]
    if detections.ndim != 2:
        detections = detections.reshape(-1, detections.shape[-1])
    if detections.shape[0] < detections.shape[1] and detections.shape[0] <= 256:
        detections = detections.T
    return detections


def _split_scores(row: Array, labels: list[str]) -> tuple[float, Array]:
    label_count = len(labels)
    if label_count and row.shape[0] >= 5 + label_count:
        return float(row[4]), row[5 : 5 + label_count]
    if label_count and row.shape[0] >= 4 + label_count:
        return 1.0, row[4 : 4 + label_count]
    return 1.0, row[4:]


def _scale_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    *,
    frame_width: int,
    frame_height: int,
    input_size: int,
) -> tuple[float, float, float, float]:
    if max(abs(x_center), abs(y_center), abs(width), abs(height)) <= 2.0:
        scale_x = float(frame_width)
        scale_y = float(frame_height)
    else:
        scale_x = frame_width / float(input_size)
        scale_y = frame_height / float(input_size)
    x1 = max(0.0, (x_center - width / 2.0) * scale_x)
    y1 = max(0.0, (y_center - height / 2.0) * scale_y)
    x2 = min(float(frame_width), (x_center + width / 2.0) * scale_x)
    y2 = min(float(frame_height), (y_center + height / 2.0) * scale_y)
    return x1, y1, x2, y2


def _nms(
    candidates: list[tuple[ObjectObservation, float, tuple[float, float, float, float]]],
    *,
    iou_threshold: float,
    max_detections: int = 50,
) -> list[tuple[ObjectObservation, float, tuple[float, float, float, float]]]:
    selected: list[tuple[ObjectObservation, float, tuple[float, float, float, float]]] = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        observation, _, box = candidate
        if any(
            observation.label == kept[0].label and _iou(box, kept[2]) > iou_threshold
            for kept in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_detections:
            break
    return selected


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-9)
