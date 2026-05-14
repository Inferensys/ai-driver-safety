from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, quantiles

import cv2

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.scoring import RiskScorer
from driver_safety.io.overlay import AnnotatedVideoWriter, draw_overlay
from driver_safety.io.sources import VideoFrameSource, WebcamFrameSource
from driver_safety.reporting.exports import export_run_artifacts
from driver_safety.reporting.recorder import SessionRecorder
from driver_safety.vision.pipeline import DriverSafetyPipeline


def analyze_video(
    video_path: str | Path,
    output_dir: str | Path,
    config: DriverSafetyConfig,
) -> dict[str, Path]:
    source = VideoFrameSource(video_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = DriverSafetyPipeline(config)
    recorder = SessionRecorder(output)
    writer: AnnotatedVideoWriter | None = None
    if config.runtime.write_video:
        output_fps = config.runtime.output_fps or source.fps
        writer = AnnotatedVideoWriter(
            output / "annotated.mp4", output_fps, (source.width, source.height)
        )
    processed_frames = 0
    last_timestamp = 0.0
    try:
        for packet in source:
            if config.runtime.max_frames and processed_frames >= config.runtime.max_frames:
                break
            if packet.frame_index % config.vision.process_every_n_frames != 0:
                continue
            processed = pipeline.process_frame(packet)
            last_timestamp = packet.timestamp
            processed_frames += 1
            recorder.write_frame_score(packet.timestamp, processed.risk_score, processed.latency_ms)
            for event in processed.events:
                recorder.write_event(event)
            if writer:
                writer.write(draw_overlay(processed))
    finally:
        source.close()
        if writer:
            writer.close()

    summary = RiskScorer(config.signal_weights or None).summarize(
        session_id=_session_id(video_path),
        source=str(video_path),
        duration_seconds=last_timestamp,
        processed_frames=processed_frames,
        events=recorder.events,
        frame_scores=recorder.frame_scores,
        metrics=_metrics(source.fps, recorder.latencies_ms, pipeline),
    )
    artifacts = export_run_artifacts(output, events=recorder.events, summary=summary)
    if writer:
        artifacts["annotated_video"] = output / "annotated.mp4"
    return artifacts


def run_webcam(config: DriverSafetyConfig, index: int = 0) -> None:
    source = WebcamFrameSource(index)
    pipeline = DriverSafetyPipeline(config)
    try:
        for packet in source:
            processed = pipeline.process_frame(packet)
            frame = draw_overlay(processed)
            cv2.imshow("AI Driver Safety", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


def _session_id(video_path: str | Path) -> str:
    stem = Path(video_path).stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}-{timestamp}"


def _metrics(
    source_fps: float,
    latencies: list[float],
    pipeline: DriverSafetyPipeline,
) -> dict[str, float | int | str]:
    if not latencies:
        return {
            "source_fps": source_fps,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "face_provider": pipeline.face_detector.provider,
            "object_provider": pipeline.object_detector.provider,
        }
    p95 = quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    avg_latency = mean(latencies)
    return {
        "source_fps": round(source_fps, 2),
        "avg_latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95, 3),
        "estimated_runtime_fps": round(1000.0 / avg_latency, 2) if avg_latency else 0.0,
        "face_provider": pipeline.face_detector.provider,
        "object_provider": pipeline.object_detector.provider,
    }
