from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from driver_safety.core.models import DetectionEvent, DriverState, Severity

DD_DRYAD_DOI = "10.5061/dryad.5tb2rbp9c"
DRYAD_API_BASE = "https://datadryad.org/api/v2"
DD_FILENAME_RE = re.compile(
    r"(?P<subject>\d{2})(?P<gender>[MF])_(?P<trial>\d)(?P<annotations>_annotations)?\.edf$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DDRecording:
    subject_id: str
    gender: str
    trial: int
    signal_file: str | None
    annotation_file: str | None
    annotation_event_count: int

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def fetch_dd_dryad_file_index() -> dict[str, object]:
    versions_url = f"{DRYAD_API_BASE}/datasets/doi%3A{quote(DD_DRYAD_DOI, safe='')}/versions"
    versions = _read_json_url(versions_url)
    version_items = versions.get("_embedded", {}).get("stash:versions", [])
    if not version_items:
        raise RuntimeError(f"No Dryad versions found for DOI {DD_DRYAD_DOI}")
    version = version_items[0]
    files_href = version["_links"]["stash:files"]["href"]
    files_url = _absolute_dryad_url(files_href)
    files: list[dict[str, str | int | None]] = []
    while files_url:
        page = _read_json_url(files_url)
        for item in page.get("_embedded", {}).get("stash:files", []):
            download_href = item.get("_links", {}).get("stash:download", {}).get("href")
            files.append(
                {
                    "filename": item.get("path") or item.get("filename"),
                    "size_bytes": item.get("size"),
                    "download_url": _absolute_dryad_url(download_href) if download_href else None,
                }
            )
        next_href = page.get("_links", {}).get("next", {}).get("href")
        files_url = _absolute_dryad_url(next_href) if next_href else ""
    return {
        "dataset": "dd-database",
        "doi": DD_DRYAD_DOI,
        "file_count": len(files),
        "files": files,
        "note": (
            "Download raw EDF files from Dryad into data/dd-database/raw. API download links can "
            "require browser/session handling; this index is kept for reproducibility."
        ),
    }


def build_dd_manifest(
    input_dir: str | Path,
    output_path: str | Path | None = None,
    dryad_index_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"DD-Database input directory does not exist: {root}")
    files_by_key: dict[tuple[str, str, int], dict[str, Path]] = {}
    for path in sorted(root.rglob("*.edf")):
        match = DD_FILENAME_RE.match(path.name)
        if not match:
            continue
        subject_id = match.group("subject")
        gender = match.group("gender").upper()
        trial = int(match.group("trial"))
        key = (subject_id, gender, trial)
        slot = "annotation_file" if match.group("annotations") else "signal_file"
        files_by_key.setdefault(key, {})[slot] = path

    recordings = []
    for (subject_id, gender, trial), files in sorted(files_by_key.items()):
        annotation_path = files.get("annotation_file")
        annotation_count = (
            len(read_dd_annotation_events(annotation_path)) if annotation_path is not None else 0
        )
        recordings.append(
            DDRecording(
                subject_id=subject_id,
                gender=gender,
                trial=trial,
                signal_file=_relative_or_none(files.get("signal_file"), root),
                annotation_file=_relative_or_none(annotation_path, root),
                annotation_event_count=annotation_count,
            )
        )

    manifest: dict[str, object] = {
        "dataset": "dd-database",
        "doi": DD_DRYAD_DOI,
        "input_dir": str(root),
        "recording_count": len(recordings),
        "recordings": [recording.to_dict() for recording in recordings],
        "sensor_channels": ["EEG", "EOG", "ECG"],
        "media_policy": "Raw EDF files stay under data/ and are not committed.",
    }
    if dryad_index_path:
        index = fetch_dd_dryad_file_index()
        _write_json(dryad_index_path, index)
        manifest["dryad_index"] = str(dryad_index_path)
    if output_path:
        _write_json(output_path, manifest)
    return manifest


def read_dd_annotation_events(annotation_path: str | Path) -> list[DetectionEvent]:
    path = Path(annotation_path)
    if path.suffix.lower() == ".edf":
        return _read_edf_annotations(path)
    return _read_text_annotations(path)


def write_dd_sensor_events(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events: list[DetectionEvent] = []
    for annotation_path in sorted(root.rglob("*_annotations.*")):
        for event in read_dd_annotation_events(annotation_path):
            event.metadata["dataset"] = "dd-database"
            event.metadata["annotation_file"] = str(annotation_path.relative_to(root))
            events.append(event)
    events_path = output / "sensor_events.json"
    summary_path = output / "sensor_summary.json"
    events_path.write_text(
        json.dumps([event.to_dict() for event in events], indent=2),
        encoding="utf-8",
    )
    summary = {
        "dataset": "dd-database",
        "source": str(root),
        "event_count": len(events),
        "signal": "sensor_drowsiness",
        "note": "Events are generated from DD-Database annotation files.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"events": events_path, "summary": summary_path}


def _read_edf_annotations(path: Path) -> list[DetectionEvent]:
    try:
        import pyedflib
    except Exception:
        return []
    reader = pyedflib.EdfReader(str(path))
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()
    events: list[DetectionEvent] = []
    for onset, duration, description in zip(onsets, durations, descriptions, strict=True):
        label = str(description).strip()
        if not label:
            continue
        score = 0.95 if "drows" in label.lower() or "sleep" in label.lower() else 0.75
        severity = Severity.CRITICAL if float(duration or 0.0) >= 30 else Severity.WARNING
        events.append(
            DetectionEvent(
                timestamp=float(onset),
                frame_index=-1,
                signal="sensor_drowsiness",
                state=DriverState.DROWSY,
                score=score,
                severity=severity,
                message=f"Physiology annotation: {label}",
                metadata={"duration_seconds": float(duration or 0.0)},
            )
        )
    return events


def _read_text_annotations(path: Path) -> list[DetectionEvent]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    rows = _read_annotation_rows(text)
    events: list[DetectionEvent] = []
    for index, row in enumerate(rows):
        timestamp = _float_from_row(row, ("timestamp", "time", "onset", "start"), default=0.0)
        duration = _float_from_row(row, ("duration", "length", "seconds"), default=0.0)
        label = _string_from_row(row, ("label", "event", "description", "state"), default="drowsy")
        if _looks_like_non_event(label):
            continue
        events.append(
            DetectionEvent(
                timestamp=timestamp,
                frame_index=-1,
                signal="sensor_drowsiness",
                state=DriverState.DROWSY,
                score=0.9,
                severity=Severity.CRITICAL if duration >= 30 else Severity.WARNING,
                message=f"Physiology annotation: {label}",
                metadata={"duration_seconds": duration, "annotation_row": index},
            )
        )
    return events


def _read_annotation_rows(text: str) -> list[dict[str, str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if rows and rows[0]:
        return rows
    parsed_rows = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        parsed_rows.append(
            {
                "timestamp": parts[0],
                "duration": parts[1] if len(parts) > 1 else "0",
                "label": " ".join(parts[2:]) if len(parts) > 2 else "drowsy",
            }
        )
    return parsed_rows


def _looks_like_non_event(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized in {"0", "none", "normal", "alert", "awake"}


def _float_from_row(row: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    normalized = {_normalize_key(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(_normalize_key(key))
        if value is None or value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def _string_from_row(row: dict[str, Any], keys: tuple[str, ...], default: str) -> str:
    normalized = {_normalize_key(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(_normalize_key(key))
        if value:
            return str(value)
    return default


def _normalize_key(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def _relative_or_none(path: Path | None, root: Path) -> str | None:
    return str(path.relative_to(root)) if path else None


def _read_json_url(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return data


def _absolute_dryad_url(href: str | None) -> str:
    if not href:
        return ""
    if href.startswith("http"):
        return href
    return f"https://datadryad.org{href}"


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
