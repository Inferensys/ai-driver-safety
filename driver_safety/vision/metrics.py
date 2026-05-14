from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: Sequence[Point]) -> float:
    if len(eye) < 6:
        raise ValueError("eye_aspect_ratio expects six eye landmarks")
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(mouth: Sequence[Point]) -> float:
    if len(mouth) < 6:
        raise ValueError("mouth_aspect_ratio expects at least six mouth landmarks")
    width = distance(mouth[0], mouth[3])
    vertical_a = distance(mouth[1], mouth[5])
    vertical_b = distance(mouth[2], mouth[4])
    if width == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * width)


def horizontal_head_offset(face_bbox: tuple[int, int, int, int], frame_width: int) -> float:
    x, _, w, _ = face_bbox
    center_x = x + w / 2.0
    frame_center = frame_width / 2.0
    if frame_center == 0:
        return 0.0
    return abs(center_x - frame_center) / frame_center
