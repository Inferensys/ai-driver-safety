from __future__ import annotations

import json
from pathlib import Path

from driver_safety.datasets.dd_database import (
    build_dd_manifest,
    read_dd_annotation_events,
    write_dd_sensor_events,
)
from driver_safety.datasets.intelligence import write_dataset_intelligence_artifacts
from driver_safety.datasets.uah_driveset import build_uah_manifest, write_uah_vehicle_events
from driver_safety.datasets.yawdd import build_yawdd_manifest


def test_yawdd_manifest_tracks_yawning_clips_and_media_permissions(tmp_path: Path) -> None:
    root = tmp_path / "yawdd"
    video = root / "mirror-camera" / "23-FemaleNoGlasses-Talking&Yawning.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    participants = root / "ParticipantsInformation.csv"
    participants.write_text(
        "File Name,Allow picture to be shared publicly?\n"
        "23-FemaleNoGlasses-Talking&Yawning.avi,Yes\n",
        encoding="utf-8",
    )

    manifest = build_yawdd_manifest(root, participants_info=participants)

    assert manifest["clip_count"] == 1
    assert manifest["scenario_counts"] == {"yawning": 1}
    clip = manifest["clips"][0]
    assert clip["public_screenshot_allowed"] is True
    assert clip["subject_id"] == "23"


def test_dd_database_manifest_and_text_annotation_events(tmp_path: Path) -> None:
    root = tmp_path / "dd"
    root.mkdir()
    (root / "01M_1.edf").touch()
    (root / "01M_1_annotations.edf").touch()
    text_annotations = root / "01M_1_annotations.csv"
    text_annotations.write_text(
        "timestamp,duration,label\n12.5,35,drowsy button press\n90,3,awake\n",
        encoding="utf-8",
    )

    manifest = build_dd_manifest(root)
    events = read_dd_annotation_events(text_annotations)
    artifacts = write_dd_sensor_events(root, tmp_path / "run")
    exported = json.loads(artifacts["events"].read_text(encoding="utf-8"))

    assert manifest["recording_count"] == 1
    assert manifest["recordings"][0]["signal_file"] == "01M_1.edf"
    assert len(events) == 1
    assert events[0].signal == "sensor_drowsiness"
    assert exported[0]["metadata"]["dataset"] == "dd-database"


def test_uah_manifest_and_vehicle_sensor_events(tmp_path: Path) -> None:
    session = tmp_path / "uah" / "D1" / "normal" / "motorway" / "trip-001"
    session.mkdir(parents=True)
    (session / "PROC_LANE_DETECTION.txt").write_text(
        "0 0.10 0 3 1\n2 0.82 0 3 1\n",
        encoding="utf-8",
    )
    (session / "PROC_VEHICLE_DETECTION.txt").write_text("3 18 1.2 1 80\n", encoding="utf-8")
    (session / "RAW_ACCELEROMETERS.txt").write_text("4 1 0.50 0.10 1.00\n", encoding="utf-8")
    (session / "RAW_GPS.txt").write_text("5 135\n", encoding="utf-8")
    (session / "PROC_OPENSTREETMAP_DATA.txt").write_text("5 100\n", encoding="utf-8")

    manifest = build_uah_manifest(tmp_path / "uah")
    artifacts = write_uah_vehicle_events(tmp_path / "uah", tmp_path / "run")
    exported = json.loads(artifacts["events"].read_text(encoding="utf-8"))

    assert manifest["session_count"] == 1
    assert manifest["behavior_counts"] == {"normal": 1}
    assert {event["signal"] for event in exported} >= {
        "lane_drift",
        "short_time_to_collision",
        "hard_maneuver",
        "speeding",
    }


def test_dataset_intelligence_artifacts_include_project_signals(tmp_path: Path) -> None:
    dd_index = tmp_path / "dd-index.json"
    dd_index.write_text(
        json.dumps(
            {
                "files": [
                    {"filename": "01M_1_annotations.edf", "size_bytes": 1200},
                    {"filename": "01M_1.edf", "size_bytes": 12000000},
                ]
            }
        ),
        encoding="utf-8",
    )

    artifacts = write_dataset_intelligence_artifacts(
        tmp_path / "intelligence.json",
        output_markdown=tmp_path / "intelligence.md",
        output_chart=tmp_path / "intelligence.png",
        dd_file_index=dd_index,
    )
    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))

    assert artifacts["markdown"].exists()
    assert artifacts["chart"].exists()
    assert payload["datasets"][0]["key"] == "yawdd"
    assert "sensor_drowsiness" in payload["datasets"][1]["project_signals"]
