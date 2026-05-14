from driver_safety.vision.metrics import (
    eye_aspect_ratio,
    horizontal_head_offset,
    mouth_aspect_ratio,
)


def test_eye_aspect_ratio_detects_open_vs_closed_eye() -> None:
    open_eye = [(0, 0), (5, -4), (15, -4), (20, 0), (15, 4), (5, 4)]
    closed_eye = [(0, 0), (5, -0.5), (15, -0.5), (20, 0), (15, 0.5), (5, 0.5)]
    assert eye_aspect_ratio(open_eye) > 0.3
    assert eye_aspect_ratio(closed_eye) < 0.08


def test_mouth_aspect_ratio_detects_yawn() -> None:
    normal = [(0, 0), (10, -2), (30, -2), (40, 0), (30, 2), (10, 2)]
    yawn = [(0, 0), (10, -16), (30, -16), (40, 0), (30, 16), (10, 16)]
    assert mouth_aspect_ratio(normal) < 0.2
    assert mouth_aspect_ratio(yawn) > 0.7


def test_horizontal_head_offset() -> None:
    assert horizontal_head_offset((40, 20, 20, 20), 100) == 0
    assert horizontal_head_offset((75, 20, 20, 20), 100) > 0.6
