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
    "sensor_drowsiness": 0.58,
    "lane_drift": 0.32,
    "short_time_to_collision": 0.48,
    "hard_maneuver": 0.26,
    "speeding": 0.2,
}

FUSION_MODEL_NAME = "driver-risk-fusion-v1"


class RiskScorer:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or DEFAULT_SIGNAL_WEIGHTS

    def score(self, signals: dict[str, float]) -> float:
        evidence = [
            _clamp(signals.get(name, 0.0)) * weight
            for name, weight in self.weights.items()
            if signals.get(name, 0.0) > 0
        ]
        fused = _noisy_or(evidence)
        fused += self._cross_signal_boost(signals)
        return round(_clamp(fused), 4)

    def fusion_channels(self, signals: dict[str, float]) -> dict[str, float]:
        return {
            "vision_fatigue": max(
                _clamp(signals.get("drowsy", 0.0)),
                _clamp(signals.get("eyes_closed", 0.0)) * 0.85,
                _clamp(signals.get("yawning", 0.0)) * 0.55,
            ),
            "visual_distraction": max(
                _clamp(signals.get("distracted", 0.0)),
                _clamp(signals.get("phone_use", 0.0)),
                _clamp(signals.get("face_missing", 0.0)) * 0.6,
            ),
            "physiology_fatigue": _clamp(signals.get("sensor_drowsiness", 0.0)),
            "vehicle_risk": max(
                _clamp(signals.get("short_time_to_collision", 0.0)),
                _clamp(signals.get("lane_drift", 0.0)) * 0.8,
                _clamp(signals.get("hard_maneuver", 0.0)) * 0.7,
                _clamp(signals.get("speeding", 0.0)) * 0.55,
            ),
        }

    def _cross_signal_boost(self, signals: dict[str, float]) -> float:
        channels = self.fusion_channels(signals)
        boost = 0.0
        if (
            _clamp(signals.get("drowsy", 0.0)) >= 0.75
            and _clamp(signals.get("eyes_closed", 0.0)) >= 0.75
        ):
            boost += 0.08
        if _clamp(signals.get("drowsy", 0.0)) >= 0.6 and _clamp(signals.get("yawning", 0.0)) >= 0.5:
            boost += 0.08
        if channels["vision_fatigue"] >= 0.6 and channels["physiology_fatigue"] >= 0.6:
            boost += 0.14
        if channels["vision_fatigue"] >= 0.6 and channels["vehicle_risk"] >= 0.5:
            boost += 0.1
        if channels["visual_distraction"] >= 0.6 and channels["vehicle_risk"] >= 0.5:
            boost += 0.14
        return boost

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
        summary_metrics = dict(metrics)
        summary_metrics.setdefault("fusion_model", FUSION_MODEL_NAME)
        return SessionSummary(
            session_id=session_id,
            source=source,
            duration_seconds=round(duration_seconds, 3),
            processed_frames=processed_frames,
            event_counts=dict(sorted(event_counts.items())),
            risk_timeline=risk_timeline,
            longest_unsafe_interval_seconds=round(longest_unsafe, 3),
            confidence_distribution=confidence_distribution,
            metrics=summary_metrics,
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


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _noisy_or(evidence: list[float]) -> float:
    probability = 1.0
    for value in evidence:
        probability *= 1.0 - _clamp(value)
    return 1.0 - probability
