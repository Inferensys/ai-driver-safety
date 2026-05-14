from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DatasetIntelligence:
    key: str
    name: str
    role: str
    real_world_value: str
    public_data_shape: dict[str, str | int | float]
    project_signals: list[str]
    analysis: list[str]
    gaps: list[str]
    demo_policy: str
    next_benchmark: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetIntelligenceReport:
    title: str
    thesis: str
    datasets: list[DatasetIntelligence]
    fused_solution: list[str] = field(default_factory=list)
    readme_results: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "thesis": self.thesis,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "fused_solution": self.fused_solution,
            "readme_results": self.readme_results,
        }


def build_dataset_intelligence_report(
    *,
    dd_file_index: str | Path | None = None,
    yawdd_manifest: str | Path | None = None,
    nitymed_manifest: str | Path | None = None,
    uah_manifest: str | Path | None = None,
) -> DatasetIntelligenceReport:
    dd_stats = _dd_stats(dd_file_index)
    yawdd_stats = _yawdd_stats(yawdd_manifest)
    nitymed_stats = _nitymed_stats(nitymed_manifest)
    uah_stats = _uah_stats(uah_manifest)
    datasets = [
        DatasetIntelligence(
            key="nitymed",
            name="NITYMED: Night-Time Yawning-Microsleep-Eyeblink-driver Distraction",
            role="Primary README demo source for real in-car yawning and microsleep video.",
            real_world_value=(
                "Exercises the visible cabin-video path on real drivers in real cars at night, "
                "including short yawning clips and longer microsleep clips."
            ),
            public_data_shape=nitymed_stats,
            project_signals=[
                "face_present",
                "mouth_aspect_ratio",
                "yawning",
                "eyes_closed",
                "microsleep_review_window",
            ],
            analysis=[
                "Use NITYMED as the public-facing real human demo source once dataset access is approved.",
                "The yawning subset is short enough for README clips and should replace generated cabin visuals.",
                "The microsleep subset is better for event-review timelines because clips are longer and include talking, looking around, and microsleep windows.",
                "Because the dataset page allows distinct frames in research publications, README images should cite the dataset and avoid committing the raw full dataset.",
            ],
            gaps=[
                "Access requires submitting name, affiliation, and business or academic email.",
                "Nighttime only, so it does not replace day/night benchmark coverage from NTHU.",
                "Frame-level event labels still need local review windows for polished demos.",
            ],
            demo_policy=(
                "Preferred source for `docs/demo/real-human-demo.*` after access approval; "
                "commit only short derived assets with attribution."
            ),
            next_benchmark=(
                "Analyze one 15-25 second yawning clip and one microsleep clip, then publish "
                "annotated GIF, MP4, timeline PNG, and summary JSON."
            ),
            source="https://datasets.esdalab.ece.uop.gr/",
        ),
        DatasetIntelligence(
            key="yawdd",
            name="YawDD: Yawning Detection Dataset",
            role="Cabin camera benchmark for real human yawning and mouth-state analysis.",
            real_world_value=(
                "Exercises the computer-vision path on real in-car faces, varied illumination, "
                "glasses/sunglasses, talking/singing, silent driving, and yawning."
            ),
            public_data_shape=yawdd_stats,
            project_signals=[
                "face_present",
                "mouth_aspect_ratio",
                "yawning",
                "talking_or_singing_context",
                "camera_view_robustness",
            ],
            analysis=[
                "Use YawDD as the first real-human visual proof because it aligns directly with yawn and mouth-state detection.",
                "The dash-camera set is stronger for product demos because each subject video contains silent, talking, and yawning behavior in one clip.",
                "The mirror-camera set is stronger for detector stress testing because the behavior classes are separated across many short clips.",
                "Frame-level labels are not guaranteed in the base release, so project scoring should report event timelines and manual-review windows rather than overclaiming supervised accuracy.",
            ],
            gaps=[
                "Not a full distraction or object-recognition dataset.",
                "Public screenshot/video display is participant-gated.",
                "Yawn start/end timestamps may need local annotation or YawDD+ style frame labels.",
            ],
            demo_policy=(
                "Run local video analysis and publish only screenshots explicitly allowed by the Participants Information file."
            ),
            next_benchmark=(
                "Analyze one dash-camera clip per public-shareable participant and export event timelines, annotated screenshots, and yawn-window JSON."
            ),
            source="https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset",
        ),
        DatasetIntelligence(
            key="dd-database",
            name="Drivers Drowsiness Database (DD-Database)",
            role="Physiological drowsiness benchmark for EEG, EOG, ECG, and annotation events.",
            real_world_value=(
                "Gives the project a sensor-backed drowsiness path instead of only relying on face landmarks."
            ),
            public_data_shape=dd_stats,
            project_signals=[
                "sensor_drowsiness",
                "eye_movement_proxy_eog",
                "heart_rate_proxy_ecg",
                "fatigue_annotation_event",
                "multimodal_risk_score",
            ],
            analysis=[
                "Use DD-Database as the physiological validation track for the original steering-wheel/heartbeat idea.",
                "EOG and ECG channels make it useful for comparing visual eye-closure/yawn events against sensor evidence.",
                "The annotation EDF files can be converted into `sensor_drowsiness` events and fused into the same risk scorer as camera events.",
                "The data comes from a driving simulator, so it is strong for drowsiness physiology but should not be treated as road-driving telemetry.",
            ],
            gaps=[
                "No cabin video/object context.",
                "Sensor hardware differs from Apple Watch; the project should describe this as physiological sensor validation, not Apple Watch-specific validation.",
                "EDF parsing requires the optional `datasets` extra.",
            ],
            demo_policy="Commit metadata-only analysis and generated event summaries; keep raw EDF under `data/`.",
            next_benchmark=(
                "Parse all annotation EDF files, compute event density per subject/trial, and compare against vision drowsiness windows when synchronized video is available."
            ),
            source="https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c",
        ),
        DatasetIntelligence(
            key="uah-driveset",
            name="UAH-DriveSet",
            role="Real car-sensor benchmark for driving style, road context, and vehicle-risk scoring.",
            real_world_value=(
                "Connects the driver-monitoring UI to actual vehicle behavior: accelerations, braking, turns, lane drift, overspeeding, and car following."
            ),
            public_data_shape=uah_stats,
            project_signals=[
                "lane_drift",
                "short_time_to_collision",
                "hard_maneuver",
                "speeding",
                "driving_style_fuzzy_score",
            ],
            analysis=[
                "Use UAH-DriveSet as the main proof for the fuzzy driving-style classifier.",
                "The dataset's normal/drowsy/aggressive behavior folders map directly to the repo's risk and fuzzy-logic framing.",
                "Road type and speed-limit context are necessary for thresholds because harsh acceleration on a city road and highway should not share one naive threshold.",
                "Vehicle events should remain separate from cabin events but fuse at the SessionSummary risk layer.",
            ],
            gaps=[
                "License is non-commercial and prohibits redistribution of raw or modified dataset data.",
                "Requires local acceptance of the dataset agreement before full raw analysis.",
                "Trip video should not be committed; publish only abstract metrics and charts.",
            ],
            demo_policy="Publish abstract metrics/charts and local commands; keep raw telemetry and trip video under `data/`.",
            next_benchmark=(
                "Run `uah-events` across all sessions, produce per-behavior event rates, and tune fuzzy thresholds by road type."
            ),
            source="http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/",
        ),
    ]
    return DatasetIntelligenceReport(
        title="AI Driver Safety Real Dataset Intelligence",
        thesis=(
            "The project should be evaluated as a fused driver-monitoring system: camera signals "
            "prove cabin behavior, physiological signals prove fatigue, and vehicle telemetry proves driving style."
        ),
        datasets=datasets,
        fused_solution=[
            "README proof path: NITYMED replaces generated visuals with real in-car yawning and microsleep footage.",
            "Vision benchmark path: YawDD validates real human yawn/mouth signals and the annotated-video/report workflow.",
            "Sensor path: DD-Database validates physiological drowsiness events for the heartbeat/sensor extension.",
            "Vehicle path: UAH-DriveSet validates fuzzy driving-style scoring from real car telemetry.",
            "Fusion path: driver-risk-fusion-v1 combines camera, physiological, and vehicle-risk events into one timeline.",
        ],
        readme_results=[
            {
                "artifact": "docs/sample-output/real-dataset-intelligence.json",
                "purpose": "Machine-readable dataset analysis for README and release notes.",
            },
            {
                "artifact": "docs/sample-output/real-dataset-intelligence.md",
                "purpose": "Human-readable intelligence report.",
            },
            {
                "artifact": "docs/screenshots/dataset-intelligence.png",
                "purpose": "Visual summary of which project signals each dataset validates.",
            },
        ],
    )


def write_dataset_intelligence_artifacts(
    output_json: str | Path,
    *,
    output_markdown: str | Path | None = None,
    output_chart: str | Path | None = None,
    dd_file_index: str | Path | None = None,
    yawdd_manifest: str | Path | None = None,
    nitymed_manifest: str | Path | None = None,
    uah_manifest: str | Path | None = None,
) -> dict[str, Path]:
    report = build_dataset_intelligence_report(
        dd_file_index=dd_file_index,
        yawdd_manifest=yawdd_manifest,
        nitymed_manifest=nitymed_manifest,
        uah_manifest=uah_manifest,
    )
    artifacts: dict[str, Path] = {}
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    artifacts["json"] = json_path
    if output_markdown:
        markdown_path = Path(output_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_dataset_intelligence_markdown(report), encoding="utf-8")
        artifacts["markdown"] = markdown_path
    if output_chart:
        chart_path = Path(output_chart)
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        render_dataset_intelligence_chart(report, chart_path)
        artifacts["chart"] = chart_path
    return artifacts


def render_dataset_intelligence_markdown(report: DatasetIntelligenceReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        report.thesis,
        "",
        "## Dataset Signal Coverage",
        "",
        "| Dataset | Validates | Data shape | Main gap |",
        "| --- | --- | --- | --- |",
    ]
    for dataset in report.datasets:
        shape = ", ".join(f"{key}: {value}" for key, value in dataset.public_data_shape.items())
        lines.append(
            f"| {dataset.name} | {', '.join(dataset.project_signals)} | {shape} | {dataset.gaps[0]} |"
        )
    lines.extend(["", "## Project Analysis", ""])
    for dataset in report.datasets:
        lines.extend([f"### {dataset.name}", ""])
        for item in dataset.analysis:
            lines.append(f"- {item}")
        lines.extend(["", f"Demo policy: {dataset.demo_policy}", ""])
    lines.extend(["## Fused Solution", ""])
    for item in report.fused_solution:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def render_dataset_intelligence_chart(
    report: DatasetIntelligenceReport,
    output_path: str | Path,
) -> None:
    width, height = 1280, 720
    image = np.full((height, width, 3), (246, 248, 247), dtype=np.uint8)
    _put_text(
        image, "Real Dataset Intelligence", (48, 70), scale=1.25, color=(22, 32, 29), thickness=2
    )
    _put_text(
        image,
        "Camera + physiological sensors + vehicle telemetry for one fused driver-risk timeline",
        (48, 112),
        scale=0.62,
        color=(72, 84, 79),
    )
    columns = [
        ("NITYMED", "Real demo video", (185, 72, 72)),
        ("YawDD", "Yawn benchmark", (58, 136, 211)),
        ("DD-Database", "Physiology sensors", (224, 132, 47)),
        ("UAH-DriveSet", "Car telemetry", (49, 150, 102)),
    ]
    card_w = 284
    card_h = 410
    y = 168
    for idx, (title, subtitle, color) in enumerate(columns):
        x = 36 + idx * 306
        cv2.rectangle(image, (x, y), (x + card_w, y + card_h), (255, 255, 255), -1)
        cv2.rectangle(image, (x, y), (x + card_w, y + card_h), (218, 225, 222), 2)
        cv2.rectangle(image, (x, y), (x + card_w, y + 12), color, -1)
        _put_text(image, title, (x + 24, y + 56), scale=0.78, color=(20, 29, 27), thickness=2)
        _put_text(image, subtitle, (x + 24, y + 90), scale=0.48, color=(80, 92, 87))
        dataset = report.datasets[idx]
        stats = _chart_stats(dataset)
        yy = y + 135
        for key, value in stats:
            label = f"{key.replace('_', ' ')}: {value}"
            _wrap_text(image, label, (x + 24, yy), max_width=235, scale=0.4, color=color)
            yy += 52
        _put_text(image, "Signals", (x + 24, y + 315), scale=0.5, color=(20, 29, 27), thickness=2)
        signal_text = ", ".join(dataset.project_signals[:3])
        _wrap_text(image, signal_text, (x + 24, y + 348), max_width=235, scale=0.4)
    footer_y = 640
    cv2.rectangle(image, (48, footer_y - 28), (1232, footer_y + 38), (232, 238, 235), -1)
    _put_text(
        image,
        "Fusion output: real cabin cues + sensor drowsiness + vehicle-risk events -> SessionSummary risk timeline",
        (72, footer_y + 12),
        scale=0.56,
        color=(28, 40, 36),
        thickness=2,
    )
    cv2.imwrite(str(output_path), image)


def _dd_stats(path: str | Path | None) -> dict[str, str | int | float]:
    defaults: dict[str, str | int | float] = {
        "subjects": 10,
        "trials": 20,
        "duration_hours": 40,
        "channels": "4 EEG, 2 EOG, 1 ECG",
    }
    if not path or not Path(path).exists():
        return defaults
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    files = data.get("files", [])
    annotation_files = [
        item for item in files if str(item.get("filename", "")).endswith("_annotations.edf")
    ]
    signal_files = [
        item
        for item in files
        if str(item.get("filename", "")).endswith(".edf")
        and not str(item.get("filename", "")).endswith("_annotations.edf")
    ]
    size_bytes = sum(int(item.get("size_bytes") or 0) for item in files)
    defaults.update(
        {
            "dryad_files": len(files),
            "signal_files": len(signal_files),
            "annotation_files": len(annotation_files),
            "indexed_mb": round(size_bytes / 1_000_000, 2),
        }
    )
    return defaults


def _yawdd_stats(path: str | Path | None) -> dict[str, str | int | float]:
    defaults: dict[str, str | int | float] = {
        "videos": 351,
        "mirror_camera_videos": 322,
        "dash_camera_videos": 29,
        "format": "640x480 RGB AVI, 30 fps",
    }
    if not path or not Path(path).exists():
        return defaults
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = data.get("scenario_counts", {})
    defaults.update(
        {
            "local_clips": int(data.get("clip_count", 0)),
            "local_yawn_clips": int(counts.get("yawning", 0)) if isinstance(counts, dict) else 0,
        }
    )
    return defaults


def _nitymed_stats(path: str | Path | None) -> dict[str, str | int | float]:
    defaults: dict[str, str | int | float] = {
        "videos": 130,
        "yawning_videos": 107,
        "microsleep_videos": 21,
        "format": "MP4, 25 fps, 720p/1080p",
    }
    if not path or not Path(path).exists():
        return defaults
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = data.get("scenario_counts", {})
    defaults.update(
        {
            "local_clips": int(data.get("clip_count", 0)),
            "local_yawning_clips": int(counts.get("yawning", 0)) if isinstance(counts, dict) else 0,
            "local_microsleep_clips": int(counts.get("microsleep", 0))
            if isinstance(counts, dict)
            else 0,
        }
    )
    return defaults


def _uah_stats(path: str | Path | None) -> dict[str, str | int | float]:
    defaults: dict[str, str | int | float] = {
        "duration_minutes": 500,
        "driver_vehicle_pairs": 6,
        "behaviors": "normal, drowsy, aggressive",
        "road_types": "motorway, secondary",
    }
    if not path or not Path(path).exists():
        return defaults
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults.update(
        {
            "local_sessions": int(data.get("session_count", 0)),
            "local_behaviors": ", ".join((data.get("behavior_counts") or {}).keys()),
        }
    )
    return defaults


def _chart_stats(dataset: DatasetIntelligence) -> list[tuple[str, str | int | float]]:
    preferred_keys = {
        "nitymed": ["videos", "yawning_videos", "microsleep_videos", "format"],
        "yawdd": ["videos", "mirror_camera_videos", "dash_camera_videos"],
        "dd-database": ["subjects", "trials", "duration_hours", "channels"],
        "uah-driveset": ["duration_minutes", "driver_vehicle_pairs", "behaviors", "road_types"],
    }
    keys = preferred_keys.get(dataset.key, list(dataset.public_data_shape.keys())[:4])
    return [
        (key, dataset.public_data_shape[key]) for key in keys if key in dataset.public_data_shape
    ]


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wrap_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    max_width: int,
    scale: float,
    color: tuple[int, int, int] = (68, 80, 75),
) -> None:
    words = text.split()
    line = ""
    y = origin[1]
    for word in words:
        candidate = f"{line} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width > max_width and line:
            _put_text(image, line, (origin[0], y), scale=scale, color=color)
            line = word
            y += 26
        else:
            line = candidate
    if line:
        _put_text(image, line, (origin[0], y), scale=scale, color=color)
