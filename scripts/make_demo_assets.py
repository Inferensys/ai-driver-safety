#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import cv2

from driver_safety.config import load_config
from driver_safety.runtime.runner import analyze_video, copy_demo_artifacts
from driver_safety.vision.landmarks import draw_synthetic_driver_frame


def main() -> None:
    samples = Path("samples")
    samples.mkdir(exist_ok=True)
    video_path = samples / "demo-driving.mp4"
    _write_synthetic_video(video_path)
    run_dir = Path("runs/demo")
    analyze_video(video_path, run_dir, load_config("configs/synthetic-demo.yaml"))
    copy_demo_artifacts(run_dir, "docs")
    _write_screenshot(run_dir / "annotated.mp4", Path("docs/screenshots/live-monitor.png"), frame_index=180)
    _write_screenshot(run_dir / "annotated.mp4", Path("docs/screenshots/event-timeline.png"), frame_index=510)
    _write_gif(run_dir / "annotated.mp4", Path("docs/demo/demo.gif"))


def _write_synthetic_video(path: Path, *, seconds: int = 54, fps: int = 24) -> None:
    width, height = 960, 540
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create demo video: {path}")
    try:
        for index in range(seconds * fps):
            timestamp = index / fps
            frame = draw_synthetic_driver_frame(width, height, timestamp)
            writer.write(frame)
    finally:
        writer.release()


def _write_screenshot(video_path: Path, output_path: Path, *, frame_index: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    cv2.imwrite(str(output_path), frame)


def _write_gif(video_path: Path, output_path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % 12 == 0 and 140 <= index <= 520:
            frames.append(cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if frames:
        imageio.mimsave(output_path, frames, duration=0.12)


if __name__ == "__main__":
    main()

