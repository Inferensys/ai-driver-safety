from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass(frozen=True, slots=True)
class NitymedClip:
    path: str
    filename: str
    scenario: str
    gender_group: str
    resolution_group: str
    expected_duration_seconds: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_nitymed_manifest(
    input_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"NITYMED input directory does not exist: {root}")
    clips = [
        _clip_from_path(video_path, root)
        for video_path in sorted(root.rglob("*"))
        if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    scenario_counts: dict[str, int] = {}
    resolution_counts: dict[str, int] = {}
    for clip in clips:
        scenario_counts[clip.scenario] = scenario_counts.get(clip.scenario, 0) + 1
        resolution_counts[clip.resolution_group] = (
            resolution_counts.get(clip.resolution_group, 0) + 1
        )

    manifest: dict[str, object] = {
        "dataset": "nitymed",
        "input_dir": str(root),
        "clip_count": len(clips),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "clips": [clip.to_dict() for clip in clips],
        "media_policy": (
            "NITYMED is listed as Creative Commons Attribution. Keep full videos under data/ "
            "and commit only short derived demo assets with clear citation."
        ),
    }
    if output_path:
        _write_json(output_path, manifest)
    return manifest


def _clip_from_path(video_path: Path, root: Path) -> NitymedClip:
    relative_path = video_path.relative_to(root)
    lower_parts = [part.lower() for part in relative_path.parts]
    lower_name = video_path.name.lower()

    if any("microsleep" in part for part in lower_parts) or "microsleep" in lower_name:
        scenario = "microsleep"
        expected_duration = "about 120"
    elif any("yawn" in part for part in lower_parts) or "yawn" in lower_name:
        scenario = "yawning"
        expected_duration = "15-25"
    else:
        scenario = "mixed_or_unlabeled"
        expected_duration = "unknown"

    if any("female" in part for part in lower_parts):
        gender_group = "female"
    elif any("male" in part for part in lower_parts):
        gender_group = "male"
    else:
        gender_group = "unknown"

    if any("hdtv720" in part or part == "720" for part in lower_parts):
        resolution_group = "hdtv720"
    elif any("full" in part or "1080" in part for part in lower_parts):
        resolution_group = "full_hd"
    else:
        resolution_group = "unknown"

    return NitymedClip(
        path=str(relative_path),
        filename=video_path.name,
        scenario=scenario,
        gender_group=gender_group,
        resolution_group=resolution_group,
        expected_duration_seconds=expected_duration,
    )


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
