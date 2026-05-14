#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from driver_safety.config import load_config
from driver_safety.runtime.runner import analyze_video


def main() -> None:
    args = _parse_args()
    video_path = args.video
    run_dir = args.out_run
    artifacts = analyze_video(video_path, run_dir, load_config(args.config))
    print(f"analysis: {run_dir}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")

    if args.publish_docs:
        if not args.license_note:
            raise SystemExit("--license-note is required when --publish-docs is used")
        docs = args.docs_dir
        published = publish_docs_artifacts(
            run_dir=run_dir,
            source_video=video_path,
            docs_dir=docs,
            source_name=args.source_name,
            source_url=args.source_url,
            license_note=args.license_note,
        )
        for name, path in published.items():
            print(f"{name}: {path}")


def publish_docs_artifacts(
    *,
    run_dir: Path,
    source_video: Path,
    docs_dir: Path,
    source_name: str,
    source_url: str | None,
    license_note: str,
) -> dict[str, Path]:
    demo_dir = docs_dir / "demo"
    screenshot_dir = docs_dir / "screenshots"
    output_dir = docs_dir / "sample-output"
    demo_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = run_dir / "annotated.mp4"
    events_path = run_dir / "events.json"
    summary_path = run_dir / "summary.json"
    if not annotated.exists():
        raise FileNotFoundError(f"Annotated video was not generated: {annotated}")
    if not events_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Run is missing JSON outputs: {run_dir}")

    events = json.loads(events_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    key_frame = _best_event_frame(events, summary)

    published = {
        "readme_video": demo_dir / "real-human-demo.mp4",
        "readme_gif": demo_dir / "real-human-demo.gif",
        "readme_frame": screenshot_dir / "real-human-live-monitor.png",
        "readme_timeline": screenshot_dir / "real-human-event-timeline.png",
        "readme_events": output_dir / "real-human-events.json",
        "readme_summary": output_dir / "real-human-summary.json",
        "source_metadata": output_dir / "real-human-demo-source.json",
    }

    shutil.copy2(annotated, published["readme_video"])
    shutil.copy2(events_path, published["readme_events"])
    shutil.copy2(summary_path, published["readme_summary"])
    _write_screenshot(annotated, published["readme_frame"], frame_index=key_frame)
    _write_gif(annotated, published["readme_gif"], center_frame=key_frame)
    _write_timeline(events, summary, published["readme_timeline"], source_name=source_name)
    _write_source_metadata(
        published["source_metadata"],
        source_video=source_video,
        source_name=source_name,
        source_url=source_url,
        license_note=license_note,
        artifacts=published,
    )
    return published


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an approved real human driver/yawning clip and publish README assets."
    )
    parser.add_argument("--video", type=Path, required=True, help="Approved source video path.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out-run", type=Path, default=Path("runs/real-human-demo"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--publish-docs", action="store_true")
    parser.add_argument("--source-name", default="Approved real human driver clip")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--license-note", default="")
    args = parser.parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video does not exist: {args.video}")
    if not args.config.exists():
        raise SystemExit(f"Config does not exist: {args.config}")
    return args


def _best_event_frame(events: list[dict[str, Any]], summary: dict[str, Any]) -> int:
    preferred = ("yawning", "eyes_closed", "drowsy", "distracted", "phone_use")
    for signal in preferred:
        candidates = [event for event in events if event.get("signal") == signal]
        if candidates:
            strongest = max(candidates, key=lambda event: float(event.get("score", 0)))
            return max(0, int(strongest.get("frame_index", 0)))
    processed = int(summary.get("processed_frames") or 0)
    return max(0, processed // 2)


def _write_screenshot(video_path: Path, output_path: Path, *, frame_index: int) -> None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")
    cv2.imwrite(str(output_path), frame)


def _write_gif(video_path: Path, output_path: Path, *, center_frame: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    capture = cv2.VideoCapture(str(video_path))
    start = max(0, center_frame - 90)
    end = center_frame + 90
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if start <= index <= end and (index - start) % 6 == 0:
            resized = cv2.resize(frame, (640, 360))
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if frames:
        imageio.mimsave(output_path, frames, duration=0.12)


def _write_timeline(
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
    *,
    source_name: str,
) -> None:
    import numpy as np

    width, height = 1280, 720
    image = np.full((height, width, 3), (248, 249, 247), dtype=np.uint8)
    duration = max(float(summary.get("duration_seconds") or 1.0), 1.0)
    title = "AI Driver Safety - Real Human Demo Timeline"
    _put_text(image, title, (48, 72), 1.0, (24, 32, 30), 2)
    _put_text(image, source_name[:95], (48, 112), 0.55, (78, 86, 82), 1)

    lanes = [
        ("eyes_closed", (64, 118, 184)),
        ("drowsy", (210, 83, 64)),
        ("yawning", (230, 158, 48)),
        ("distracted", (91, 134, 112)),
        ("phone_use", (130, 95, 171)),
        ("face_missing", (112, 112, 112)),
    ]
    timeline_left = 190
    timeline_right = width - 72
    y0 = 180
    lane_gap = 62
    cv2.rectangle(image, (40, 145), (width - 40, 610), (255, 255, 255), -1)
    cv2.rectangle(image, (40, 145), (width - 40, 610), (220, 226, 223), 2)

    for tick in range(0, 6):
        x = int(timeline_left + (timeline_right - timeline_left) * tick / 5)
        cv2.line(image, (x, y0 - 28), (x, y0 + lane_gap * len(lanes) - 8), (232, 235, 233), 1)
        seconds = duration * tick / 5
        _put_text(
            image,
            f"{seconds:0.0f}s",
            (x - 18, y0 + lane_gap * len(lanes) + 22),
            0.42,
            (88, 96, 92),
            1,
        )

    for lane_index, (signal, color) in enumerate(lanes):
        y = y0 + lane_index * lane_gap
        _put_text(image, signal.replace("_", " "), (58, y + 7), 0.48, color, 2)
        cv2.line(image, (timeline_left, y), (timeline_right, y), (209, 216, 212), 2)
        for event in events:
            if event.get("signal") != signal:
                continue
            timestamp = max(0.0, min(duration, float(event.get("timestamp", 0.0))))
            score = max(0.15, min(1.0, float(event.get("score", 0.0))))
            x = int(timeline_left + (timeline_right - timeline_left) * timestamp / duration)
            radius = int(5 + score * 9)
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius, (255, 255, 255), 2)

    counts = summary.get("event_counts") or {}
    count_text = "  ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    _wrap_text(image, count_text or "No events emitted", (58, 662), 1120, 0.5, (52, 62, 58))
    cv2.imwrite(str(output_path), image)


def _write_source_metadata(
    output_path: Path,
    *,
    source_video: Path,
    source_name: str,
    source_url: str | None,
    license_note: str,
    artifacts: dict[str, Path],
) -> None:
    payload = {
        "source_name": source_name,
        "source_url": source_url,
        "license_note": license_note,
        "source_video": str(source_video),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "policy": (
            "These README assets must come from an approved human recording or a dataset whose "
            "terms allow public derived demo media."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _put_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wrap_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    words = text.split()
    line = ""
    y = origin[1]
    for word in words:
        candidate = f"{line} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width > max_width and line:
            _put_text(image, line, (origin[0], y), scale, color, 1)
            line = word
            y += 28
        else:
            line = candidate
    if line:
        _put_text(image, line, (origin[0], y), scale, color, 1)


if __name__ == "__main__":
    main()
