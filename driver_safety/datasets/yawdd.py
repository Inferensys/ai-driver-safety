from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass(frozen=True, slots=True)
class YawDDClip:
    path: str
    filename: str
    subject_id: str | None
    camera_view: str
    scenario: str
    public_screenshot_allowed: bool | None

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def build_yawdd_manifest(
    input_dir: str | Path,
    output_path: str | Path | None = None,
    participants_info: str | Path | None = None,
) -> dict[str, object]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"YawDD input directory does not exist: {root}")
    public_flags = _read_public_screenshot_flags(participants_info)
    clips = [
        _clip_from_path(video_path, root, public_flags)
        for video_path in sorted(root.rglob("*"))
        if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    scenario_counts: dict[str, int] = {}
    public_screenshot_counts = {"allowed": 0, "blocked": 0, "unknown": 0}
    for clip in clips:
        scenario_counts[clip.scenario] = scenario_counts.get(clip.scenario, 0) + 1
        if clip.public_screenshot_allowed is True:
            public_screenshot_counts["allowed"] += 1
        elif clip.public_screenshot_allowed is False:
            public_screenshot_counts["blocked"] += 1
        else:
            public_screenshot_counts["unknown"] += 1

    manifest: dict[str, object] = {
        "dataset": "yawdd",
        "input_dir": str(root),
        "clip_count": len(clips),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "public_screenshot_counts": public_screenshot_counts,
        "clips": [clip.to_dict() for clip in clips],
        "media_policy": (
            "Use videos locally for research/validation. Only publish screenshots for clips whose "
            "Participants Information row explicitly allows public sharing."
        ),
    }
    if output_path:
        _write_json(output_path, manifest)
    return manifest


def _clip_from_path(
    video_path: Path,
    root: Path,
    public_flags: dict[str, bool],
) -> YawDDClip:
    relative_path = video_path.relative_to(root)
    filename = video_path.name
    stem = video_path.stem
    tokens = stem.split("-")
    subject_id = tokens[0] if tokens and tokens[0].isdigit() else None
    lower_name = filename.lower()
    lower_parts = [part.lower() for part in relative_path.parts]
    if any("dash" in part for part in lower_parts):
        camera_view = "dash"
    elif any("mirror" in part or "rear" in part for part in lower_parts):
        camera_view = "mirror"
    else:
        camera_view = "unknown"
    if "yawn" in lower_name:
        scenario = "yawning"
    elif "talk" in lower_name or "sing" in lower_name:
        scenario = "talking_or_singing"
    elif "normal" in lower_name or "silent" in lower_name:
        scenario = "normal"
    else:
        scenario = "mixed_or_unlabeled"
    public_allowed = public_flags.get(filename)
    if public_allowed is None and subject_id is not None:
        public_allowed = public_flags.get(subject_id)
    return YawDDClip(
        path=str(relative_path),
        filename=filename,
        subject_id=subject_id,
        camera_view=camera_view,
        scenario=scenario,
        public_screenshot_allowed=public_allowed,
    )


def _read_public_screenshot_flags(path: str | Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    info_path = Path(path)
    if not info_path.exists():
        raise FileNotFoundError(f"Participants information file does not exist: {info_path}")
    with info_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        rows = list(csv.DictReader(handle, dialect=dialect))
    if not rows:
        return {}

    keys = list(rows[0].keys())
    flag_key = _find_key(keys, ("allow picture to be shared publicly", "public", "share"))
    file_key = _find_key(keys, ("file", "video", "filename", "clip"))
    subject_key = _find_key(keys, ("subject", "participant", "id"))
    if flag_key is None:
        return {}
    flags: dict[str, bool] = {}
    for row in rows:
        value = _boolish(row.get(flag_key, ""))
        if value is None:
            continue
        if file_key and row.get(file_key):
            flags[Path(row[file_key]).name] = value
        if subject_key and row.get(subject_key):
            flags[row[subject_key].strip()] = value
    return flags


def _find_key(keys: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_key(key): key for key in keys}
    for candidate in candidates:
        normalized_candidate = _normalize_key(candidate)
        for normalized_key, original_key in normalized.items():
            if normalized_candidate in normalized_key:
                return original_key
    return None


def _normalize_key(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def _boolish(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1", "allowed"}:
        return True
    if normalized in {"no", "n", "false", "0", "blocked", "not allowed"}:
        return False
    return None


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
