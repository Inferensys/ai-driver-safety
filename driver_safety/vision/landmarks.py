from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.models import FramePacket

Point = tuple[float, float]


@dataclass(slots=True)
class FaceObservation:
    bbox: tuple[int, int, int, int]
    landmarks: dict[str, list[Point]]
    confidence: float = 1.0
    provider: str = "unknown"
    metadata: dict[str, float | str] = field(default_factory=dict)

    @property
    def all_points(self) -> list[Point]:
        points: list[Point] = []
        for group in self.landmarks.values():
            points.extend(group)
        return points


class FaceLandmarkDetector(Protocol):
    provider: str

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        ...


class ModelNotAvailableError(RuntimeError):
    pass


class SyntheticFaceLandmarkDetector:
    provider = "synthetic"

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        frame = packet.frame
        h, w = frame.shape[:2]
        scenario = _scenario_for_timestamp(packet.timestamp)
        if scenario == "face_missing":
            return []
        face_w = int(w * 0.26)
        face_h = int(h * 0.42)
        x_offset = int(w * 0.22 if scenario == "distracted" else w * 0.0)
        cx = w // 2 + x_offset
        cy = int(h * 0.42)
        bbox = (cx - face_w // 2, cy - face_h // 2, face_w, face_h)
        eye_open = scenario != "eyes_closed"
        mouth_open = scenario == "yawning"
        landmarks = _synthetic_landmarks(cx, cy, eye_open=eye_open, mouth_open=mouth_open)
        return [FaceObservation(bbox=bbox, landmarks=landmarks, provider=self.provider)]


class HaarFaceDetector:
    provider = "haar"

    def __init__(self, cascade_path: Path | None = None) -> None:
        if cascade_path is None:
            cascade_path = Path("legacy/haar-cascade-files/haarcascade_frontalface_default.xml")
        self.cascade_path = cascade_path
        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise ModelNotAvailableError(f"Cannot load Haar cascade: {cascade_path}")

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        frame = packet.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        observations: list[FaceObservation] = []
        for x, y, w, h in faces[:1]:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            observations.append(
                FaceObservation(
                    bbox=(int(x), int(y), int(w), int(h)),
                    landmarks=_synthetic_landmarks(cx, cy, eye_open=True, mouth_open=False, scale=w / 150),
                    provider=self.provider,
                    confidence=0.65,
                )
            )
        return observations


class MediaPipeFaceLandmarker:
    provider = "mediapipe"

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise ModelNotAvailableError(
                f"MediaPipe face landmarker model is missing: {model_path}. "
                "Run `python scripts/download_models.py --mediapipe-face` or set vision.provider=auto."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ModelNotAvailableError(
                "MediaPipe is not installed. Install with `pip install ai-driver-safety[vision]`."
            ) from exc

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(image, int(packet.timestamp * 1000))
        if not result.face_landmarks:
            return []
        h, w = packet.frame.shape[:2]
        landmarks = result.face_landmarks[0]
        points = [(landmark.x * w, landmark.y * h) for landmark in landmarks]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        bbox = (
            int(max(0, min(xs))),
            int(max(0, min(ys))),
            int(min(w, max(xs)) - max(0, min(xs))),
            int(min(h, max(ys)) - max(0, min(ys))),
        )
        groups = {
            "left_eye": _pick(points, [33, 160, 158, 133, 153, 144]),
            "right_eye": _pick(points, [362, 385, 387, 263, 373, 380]),
            "mouth": _pick(points, [61, 81, 13, 291, 14, 178]),
            "pose": _pick(points, [1, 33, 263, 61, 291]),
        }
        return [FaceObservation(bbox=bbox, landmarks=groups, provider=self.provider)]


def create_face_detector(config: DriverSafetyConfig) -> FaceLandmarkDetector:
    provider = config.vision.provider.lower()
    if provider == "synthetic":
        return SyntheticFaceLandmarkDetector()
    if provider in {"mediapipe", "auto"}:
        try:
            return MediaPipeFaceLandmarker(Path(config.vision.face_landmarker_model))
        except ModelNotAvailableError:
            if provider == "mediapipe" and not config.vision.fallback_to_haar:
                raise
    if provider in {"haar", "opencv", "auto", "mediapipe"} and config.vision.fallback_to_haar:
        return HaarFaceDetector()
    raise ModelNotAvailableError(f"Unsupported face detector provider: {config.vision.provider}")


def _pick(points: list[Point], indexes: list[int]) -> list[Point]:
    return [points[index] for index in indexes if index < len(points)]


def _synthetic_landmarks(
    cx: int,
    cy: int,
    *,
    eye_open: bool,
    mouth_open: bool,
    scale: float = 1.0,
) -> dict[str, list[Point]]:
    eye_half_w = 22 * scale
    eye_v = (8 if eye_open else 1.2) * scale
    mouth_half_w = 34 * scale
    mouth_v = (24 if mouth_open else 6) * scale
    left_eye_cx = cx - int(38 * scale)
    right_eye_cx = cx + int(38 * scale)
    eye_y = cy - int(32 * scale)
    mouth_y = cy + int(48 * scale)
    return {
        "left_eye": _eye_points(left_eye_cx, eye_y, eye_half_w, eye_v),
        "right_eye": _eye_points(right_eye_cx, eye_y, eye_half_w, eye_v),
        "mouth": [
            (cx - mouth_half_w, mouth_y),
            (cx - mouth_half_w / 2, mouth_y - mouth_v),
            (cx + mouth_half_w / 2, mouth_y - mouth_v),
            (cx + mouth_half_w, mouth_y),
            (cx + mouth_half_w / 2, mouth_y + mouth_v),
            (cx - mouth_half_w / 2, mouth_y + mouth_v),
        ],
        "pose": [(cx, cy), (left_eye_cx, eye_y), (right_eye_cx, eye_y)],
    }


def _eye_points(cx: float, cy: float, half_w: float, vertical: float) -> list[Point]:
    return [
        (cx - half_w, cy),
        (cx - half_w / 2, cy - vertical),
        (cx + half_w / 2, cy - vertical),
        (cx + half_w, cy),
        (cx + half_w / 2, cy + vertical),
        (cx - half_w / 2, cy + vertical),
    ]


def _scenario_for_timestamp(timestamp: float) -> str:
    phase = timestamp % 54
    if 8 <= phase < 13:
        return "eyes_closed"
    if 18 <= phase < 23:
        return "yawning"
    if 29 <= phase < 35:
        return "distracted"
    if 47 <= phase < 51:
        return "face_missing"
    return "attentive"


def draw_synthetic_driver_frame(width: int, height: int, timestamp: float) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 31, 32)
    cv2.rectangle(frame, (0, int(height * 0.58)), (width, height), (22, 24, 25), -1)
    cv2.rectangle(frame, (int(width * 0.08), int(height * 0.05)), (int(width * 0.92), int(height * 0.52)), (62, 68, 72), -1)
    cv2.line(frame, (int(width * 0.5), int(height * 0.08)), (int(width * 0.48), int(height * 0.52)), (235, 190, 82), 3)
    cv2.line(frame, (int(width * 0.5), int(height * 0.08)), (int(width * 0.54), int(height * 0.52)), (235, 190, 82), 3)
    cv2.circle(frame, (int(width * 0.5), int(height * 0.73)), int(width * 0.12), (12, 12, 13), 16)
    cv2.putText(
        frame,
        "AI Driver Safety demo cabin",
        (24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (210, 218, 218),
        2,
        cv2.LINE_AA,
    )
    packet = FramePacket(frame=frame, timestamp=timestamp, frame_index=int(timestamp * 24))
    detector = SyntheticFaceLandmarkDetector()
    observations = detector.detect(packet)
    if not observations:
        return frame
    obs = observations[0]
    x, y, w, h = obs.bbox
    cv2.ellipse(frame, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (92, 126, 150), -1)
    for eye in ("left_eye", "right_eye"):
        pts = np.array(obs.landmarks[eye], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (245, 245, 235), 2)
    mouth = np.array(obs.landmarks["mouth"], dtype=np.int32)
    cv2.polylines(frame, [mouth], True, (80, 90, 210), 2)
    return frame

