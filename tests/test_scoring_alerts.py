from driver_safety.core.alerts import AlertPolicy
from driver_safety.core.models import DetectionEvent, DriverState, Severity
from driver_safety.core.scoring import RiskScorer


def test_risk_scorer_weights_signals() -> None:
    scorer = RiskScorer()
    assert scorer.score({"phone_use": 1.0}) > scorer.score({"yawning": 1.0})
    assert scorer.score({"phone_use": 1.0, "drowsy": 1.0}) == 1.0


def test_alert_policy_applies_cooldown() -> None:
    policy = AlertPolicy(cooldown_seconds=2)
    first = _event(0.0)
    second = _event(1.0)
    third = _event(3.0)
    assert len(policy.evaluate([first])) == 1
    assert policy.evaluate([second]) == []
    assert len(policy.evaluate([third])) == 1


def _event(timestamp: float) -> DetectionEvent:
    return DetectionEvent(
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        signal="eyes_closed",
        state=DriverState.EYES_CLOSED,
        score=1.0,
        severity=Severity.WARNING,
        message="test",
    )

