from __future__ import annotations

from collections import Counter
from statistics import mean

from driver_safety.core.models import DetectionEvent, DriverState, SessionSummary

DEFAULT_SIGNAL_WEIGHTS = {
    "eyes_closed": 0.34,
    "drowsy": 0.54,
    "yawning": 0.22,
    "distracted": 0.34,
    "phone_use": 0.64,
    "face_missing": 0.24,
}


class RiskScorer:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or DEFAULT_SIGNAL_WEIGHTS

    def score(self, signals: dict[str, float]) -> float:
        weighted = sum(signals.get(name, 0.0) * weight for name, weight in self.weights.items())
        return round(min(1.0, max(0.0, weighted)), 4)

    def state_from_events(self, events: list[DetectionEvent], risk_score: float) -> DriverState:
        priority = [
            DriverState.PHONE_USE,
            DriverState.DROWSY,
            DriverState.EYES_CLOSED,
            DriverState.YAWNING,
            DriverState.DISTRACTED,
            DriverState.FACE_MISSING,
        ]
        active = {event.state for event in events}
        for state in priority:
            if state in active:
                return state
        if risk_score >= 0.55:
            return DriverState.DISTRACTED
        return DriverState.ATTENTIVE

    def summarize(
        self,
        *,
        session_id: str,
        source: str,
        duration_seconds: float,
        processed_frames: int,
        events: list[DetectionEvent],
        frame_scores: list[tuple[float, float]],
        metrics: dict[str, float | int | str],
    ) -> SessionSummary:
        event_counts = Counter(event.signal for event in events)
        risk_timeline = [
            {"timestamp": round(timestamp, 3), "risk_score": round(score, 4)}
            for timestamp, score in frame_scores
        ]
        unsafe_timestamps = [timestamp for timestamp, score in frame_scores if score >= 0.45]
        longest_unsafe = _longest_contiguous_interval(unsafe_timestamps)
        signal_scores: dict[str, list[float]] = {}
        for event in events:
            signal_scores.setdefault(event.signal, []).append(event.score)
        confidence_distribution = {
            signal: round(mean(scores), 4) for signal, scores in sorted(signal_scores.items())
        }
        return SessionSummary(
            session_id=session_id,
            source=source,
            duration_seconds=round(duration_seconds, 3),
            processed_frames=processed_frames,
            event_counts=dict(sorted(event_counts.items())),
            risk_timeline=risk_timeline,
            longest_unsafe_interval_seconds=round(longest_unsafe, 3),
            confidence_distribution=confidence_distribution,
            metrics=metrics,
        )


def _longest_contiguous_interval(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    longest = 0.0
    start = previous = timestamps[0]
    for timestamp in timestamps[1:]:
        if timestamp - previous > 1.25:
            longest = max(longest, previous - start)
            start = timestamp
        previous = timestamp
    return max(longest, previous - start)

