from pathlib import Path

import cv2

from driver_safety.config import load_config
from driver_safety.runtime.runner import analyze_video
from driver_safety.vision.landmarks import draw_synthetic_driver_frame


def test_analyze_video_writes_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    _write_video(video)
    config = load_config(Path("configs/synthetic-demo.yaml"))
    config.runtime.max_frames = 160
    out = tmp_path / "run"
    artifacts = analyze_video(video, out, config)
    assert artifacts["events"].exists()
    assert artifacts["summary"].exists()
    assert artifacts["html"].exists()
    assert artifacts["csv"].exists()
    assert artifacts["annotated_video"].exists()


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
    assert writer.isOpened()
    for index in range(180):
        writer.write(draw_synthetic_driver_frame(320, 180, index / 24))
    writer.release()

