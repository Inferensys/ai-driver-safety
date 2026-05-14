from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from driver_safety.core.models import DetectionEvent, DriverState, Severity

UAH_SENSOR_FILES = (
    "RAW_ACCELEROMETERS.txt",
    "RAW_GPS.txt",
    "PROC_LANE_DETECTION.txt",
    "PROC_VEHICLE_DETECTION.txt",
    "PROC_OPENSTREETMAP_DATA.txt",
)


@dataclass(frozen=True, slots=True)
class UAHSession:
    path: str
    behavior: str
    road_type: str
    sensor_files: list[str]
    duration_seconds: float | None
    vehicle_event_count: int

    def to_dict(self) -> dict[str, str | int | float | list[str] | None]:
        return asdict(self)


def build_uah_manifest(
    input_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"UAH-DriveSet input directory does not exist: {root}")
    sessions = [
        _session_from_dir(session_dir, root)
        for session_dir in sorted(_find_session_dirs(root))
    ]
    behavior_counts: dict[str, int] = {}
    for session in sessions:
        behavior_counts[session.behavior] = behavior_counts.get(session.behavior, 0) + 1
    manifest: dict[str, object] = {
        "dataset": "uah-driveset",
        "input_dir": str(root),
        "session_count": len(sessions),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "sessions": [session.to_dict() for session in sessions],
        "media_policy": (
            "UAH-DriveSet is non-commercial research data. Keep raw telemetry and trip video under data/."
        ),
    }
    if output_path:
        _write_json(output_path, manifest)
    return manifest


def write_uah_vehicle_events(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events: list[DetectionEvent] = []
    for session_dir in sorted(_find_session_dirs(root)):
        session_events = generate_uah_vehicle_events(session_dir)
        for event in session_events:
            event.metadata["dataset"] = "uah-driveset"
            event.metadata["session_dir"] = str(session_dir.relative_to(root))
            events.append(event)
    events_path = output / "vehicle_events.json"
    summary_path = output / "vehicle_summary.json"
    events_path.write_text(
        json.dumps([event.to_dict() for event in events], indent=2),
        encoding="utf-8",
    )
    summary = {
        "dataset": "uah-driveset",
        "source": str(root),
        "event_count": len(events),
        "signals": ["lane_drift", "short_time_to_collision", "hard_maneuver", "speeding"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"events": events_path, "summary": summary_path}


def generate_uah_vehicle_events(session_dir: str | Path) -> list[DetectionEvent]:
    root = Path(session_dir)
    events: list[DetectionEvent] = []
    events.extend(_lane_drift_events(root / "PROC_LANE_DETECTION.txt"))
    events.extend(_short_ttc_events(root / "PROC_VEHICLE_DETECTION.txt"))
    events.extend(_hard_maneuver_events(root / "RAW_ACCELEROMETERS.txt"))
    events.extend(_speeding_events(root / "RAW_GPS.txt", root / "PROC_OPENSTREETMAP_DATA.txt"))
    return sorted(events, key=lambda event: (event.timestamp, event.signal))


def _find_session_dirs(root: Path) -> set[Path]:
    return {
        path.parent
        for sensor_file in UAH_SENSOR_FILES
        for path in root.rglob(sensor_file)
        if path.is_file()
    }


def _session_from_dir(session_dir: Path, root: Path) -> UAHSession:
    sensor_files = [file_name for file_name in UAH_SENSOR_FILES if (session_dir / file_name).exists()]
    return UAHSession(
        path=str(session_dir.relative_to(root)),
        behavior=_infer_behavior(session_dir),
        road_type=_infer_road_type(session_dir),
        sensor_files=sensor_files,
        duration_seconds=_infer_duration(session_dir),
        vehicle_event_count=len(generate_uah_vehicle_events(session_dir)),
    )


def _lane_drift_events(path: Path) -> list[DetectionEvent]:
    data = _load_table(path, min_columns=2)
    if data is None:
        return []
    events = []
    for timestamp, lane_offset in data[:, [0, 1]]:
        offset = abs(float(lane_offset))
        if offset < 0.65:
            continue
        score = min(1.0, offset / 1.5)
        events.append(
            DetectionEvent(
                timestamp=float(timestamp),
                frame_index=-1,
                signal="lane_drift",
                state=DriverState.DISTRACTED,
                score=round(score, 4),
                severity=Severity.CRITICAL if offset >= 1.0 else Severity.WARNING,
                message=f"Vehicle lane offset {lane_offset:.2f} m",
                metadata={"lane_offset_m": float(lane_offset)},
            )
        )
    return _thin_events(events, min_gap_seconds=2.0)


def _short_ttc_events(path: Path) -> list[DetectionEvent]:
    data = _load_table(path, min_columns=3)
    if data is None:
        return []
    events = []
    for timestamp, lead_distance, ttc in data[:, [0, 1, 2]]:
        ttc_value = float(ttc)
        if ttc_value <= 0.0 or ttc_value >= 2.0:
            continue
        score = min(1.0, (2.0 - ttc_value) / 2.0 + 0.45)
        events.append(
            DetectionEvent(
                timestamp=float(timestamp),
                frame_index=-1,
                signal="short_time_to_collision",
                state=DriverState.DISTRACTED,
                score=round(score, 4),
                severity=Severity.CRITICAL if ttc_value < 1.0 else Severity.WARNING,
                message=f"Short time to collision {ttc_value:.2f}s",
                metadata={"lead_distance_m": float(lead_distance), "time_to_collision_s": ttc_value},
            )
        )
    return _thin_events(events, min_gap_seconds=2.0)


def _hard_maneuver_events(path: Path) -> list[DetectionEvent]:
    data = _load_table(path, min_columns=5)
    if data is None:
        return []
    events = []
    for row in data:
        timestamp = float(row[0])
        accel_x = float(row[2])
        accel_y = float(row[3])
        accel_z = float(row[4])
        maneuver_g = max(abs(accel_x), abs(accel_y), abs(accel_z - 1.0))
        if maneuver_g < 0.35:
            continue
        score = min(1.0, maneuver_g / 0.8)
        events.append(
            DetectionEvent(
                timestamp=timestamp,
                frame_index=-1,
                signal="hard_maneuver",
                state=DriverState.DISTRACTED,
                score=round(score, 4),
                severity=Severity.CRITICAL if maneuver_g >= 0.55 else Severity.WARNING,
                message=f"Hard maneuver {maneuver_g:.2f} g",
                metadata={
                    "accel_x_g": accel_x,
                    "accel_y_g": accel_y,
                    "accel_z_g": accel_z,
                    "maneuver_g": maneuver_g,
                },
            )
        )
    return _thin_events(events, min_gap_seconds=2.0)


def _speeding_events(gps_path: Path, osm_path: Path) -> list[DetectionEvent]:
    gps = _load_table(gps_path, min_columns=2)
    osm = _load_table(osm_path, min_columns=2)
    if gps is None:
        return []
    events = []
    fallback_limit_kmh = 120.0
    for row in gps:
        timestamp = float(row[0])
        speed = float(row[1])
        limit = _nearest_osm_speed_limit(osm, timestamp) if osm is not None else fallback_limit_kmh
        if limit <= 0.0 or speed <= limit + 8.0:
            continue
        over_by = speed - limit
        score = min(1.0, over_by / 40.0 + 0.35)
        events.append(
            DetectionEvent(
                timestamp=timestamp,
                frame_index=-1,
                signal="speeding",
                state=DriverState.DISTRACTED,
                score=round(score, 4),
                severity=Severity.CRITICAL if over_by >= 25.0 else Severity.WARNING,
                message=f"Speed {speed:.1f} km/h over limit {limit:.1f} km/h",
                metadata={"speed_kmh": speed, "limit_kmh": limit},
            )
        )
    return _thin_events(events, min_gap_seconds=5.0)


def _nearest_osm_speed_limit(osm: np.ndarray, timestamp: float) -> float:
    index = int(np.abs(osm[:, 0] - timestamp).argmin())
    return float(osm[index, 1])


def _load_table(path: Path, min_columns: int) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.genfromtxt(path, dtype=float, delimiter=" ")
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < min_columns:
        return None
    return data


def _thin_events(events: list[DetectionEvent], min_gap_seconds: float) -> list[DetectionEvent]:
    thinned = []
    last_timestamp: float | None = None
    for event in sorted(events, key=lambda item: item.timestamp):
        if last_timestamp is not None and event.timestamp - last_timestamp < min_gap_seconds:
            continue
        thinned.append(event)
        last_timestamp = event.timestamp
    return thinned


def _infer_behavior(path: Path) -> str:
    joined = " ".join(part.lower() for part in path.parts)
    for behavior in ("drowsy", "aggressive", "normal"):
        if behavior in joined:
            return behavior
    return "unknown"


def _infer_road_type(path: Path) -> str:
    joined = " ".join(part.lower() for part in path.parts)
    if "motorway" in joined or "highway" in joined:
        return "motorway"
    if "secondary" in joined:
        return "secondary"
    return "unknown"


def _infer_duration(path: Path) -> float | None:
    durations = []
    for file_name in UAH_SENSOR_FILES:
        data = _load_table(path / file_name, min_columns=1)
        if data is not None and len(data) > 0:
            durations.append(float(np.nanmax(data[:, 0])))
    return round(max(durations), 3) if durations else None


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
