from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class DriverState(str, Enum):
    ATTENTIVE = "attentive"
    EYES_CLOSED = "eyes_closed"
    DROWSY = "drowsy"
    YAWNING = "yawning"
    DISTRACTED = "distracted"
    PHONE_USE = "phone_use"
    FACE_MISSING = "face_missing"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(slots=True)
class FramePacket:
    frame: Any
    timestamp: float
    frame_index: int
    source_id: str = "video"
    fps: float | None = None


@dataclass(slots=True)
class DetectionEvent:
    timestamp: float
    frame_index: int
    signal: str
    state: DriverState
    score: float
    severity: Severity
    message: str
    bbox: tuple[int, int, int, int] | None = None
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["severity"] = self.severity.value
        return data


@dataclass(slots=True)
class ProcessedFrame:
    packet: FramePacket
    state: DriverState
    risk_score: float
    signals: dict[str, float]
    events: list[DetectionEvent]
    latency_ms: float
    face_bbox: tuple[int, int, int, int] | None = None
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SessionSummary:
    session_id: str
    source: str
    duration_seconds: float
    processed_frames: int
    event_counts: dict[str, int]
    risk_timeline: list[dict[str, float]]
    longest_unsafe_interval_seconds: float
    confidence_distribution: dict[str, float]
    metrics: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
