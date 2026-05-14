from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from driver_safety.config import load_config
from driver_safety.datasets import (
    all_dataset_specs,
    build_dd_manifest,
    build_uah_manifest,
    build_yawdd_manifest,
    fetch_dd_dryad_file_index,
    write_dd_sensor_events,
    write_uah_vehicle_events,
)
from driver_safety.reporting.exports import write_events_csv, write_html_report
from driver_safety.runtime.runner import analyze_video, run_webcam

app = typer.Typer(no_args_is_help=True, help="AI Driver Safety local driver monitoring CLI.")
datasets_app = typer.Typer(no_args_is_help=True, help="Prepare real validation datasets.")
app.add_typer(datasets_app, name="datasets")
console = Console()


@app.command()
def analyze(
    video: Annotated[Path, typer.Option("--video", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("runs/demo"),
    config: Annotated[Path | None, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Analyze a video and export annotated video, events, summary, CSV, and HTML report."""
    cfg = load_config(config)
    artifacts = analyze_video(video, out, cfg)
    console.print("[bold green]Analysis complete[/bold green]")
    for name, path in artifacts.items():
        console.print(f"{name}: {path}")


@app.command()
def run(
    source: Annotated[str, typer.Option("--source")] = "webcam",
    webcam: Annotated[int, typer.Option("--webcam")] = 0,
    config: Annotated[Path | None, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Run realtime monitoring from a webcam source."""
    if source != "webcam":
        raise typer.BadParameter("Only --source webcam is supported for realtime mode.")
    cfg = load_config(config)
    cfg.runtime.display = True
    run_webcam(cfg, index=webcam)


@app.command()
def report(
    run_dir: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)] = Path("runs/demo"),
    format: Annotated[str, typer.Option("--format")] = "html,json,csv",
) -> None:
    """Regenerate report exports from a completed run directory."""
    events_path = run_dir / "events.json"
    summary_path = run_dir / "summary.json"
    if not events_path.exists() or not summary_path.exists():
        raise typer.BadParameter(f"Run directory must contain events.json and summary.json: {run_dir}")
    from driver_safety.core.models import DetectionEvent, DriverState, SessionSummary, Severity

    event_data = json.loads(events_path.read_text(encoding="utf-8"))
    events = [
        DetectionEvent(
            timestamp=item["timestamp"],
            frame_index=item["frame_index"],
            signal=item["signal"],
            state=DriverState(item["state"]),
            score=item["score"],
            severity=Severity(item["severity"]),
            message=item["message"],
            bbox=tuple(item["bbox"]) if item.get("bbox") else None,
            landmarks=[tuple(point) for point in item.get("landmarks", [])],
            metadata=item.get("metadata", {}),
            event_id=item.get("event_id", ""),
        )
        for item in event_data
    ]
    summary = SessionSummary(**json.loads(summary_path.read_text(encoding="utf-8")))
    requested = {part.strip() for part in format.split(",")}
    if "html" in requested:
        write_html_report(run_dir / "report.html", events=events, summary=summary)
        console.print(f"html: {run_dir / 'report.html'}")
    if "csv" in requested:
        write_events_csv(run_dir / "events.csv", events)
        console.print(f"csv: {run_dir / 'events.csv'}")
    if "json" in requested:
        console.print(f"json: {summary_path}, {events_path}")


@app.command()
def studio(
    run_dir: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)] = Path("runs/demo"),
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Serve the local FastAPI API for Studio review."""
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise typer.BadParameter(
            "Studio requires API dependencies. Install with `pip install ai-driver-safety[api]`."
        ) from exc
    from driver_safety.api.service import create_app

    uvicorn.run(create_app(run_dir), host=host, port=port)


@datasets_app.command("list")
def list_datasets() -> None:
    """Show the supported real-dataset validation tracks."""
    table = Table(title="AI Driver Safety real dataset tracks")
    table.add_column("Key")
    table.add_column("Use case")
    table.add_column("Policy")
    for spec in all_dataset_specs():
        table.add_row(spec.key, spec.use_case, spec.media_policy)
    console.print(table)


@datasets_app.command("prepare-yawdd")
def prepare_yawdd(
    input_dir: Annotated[Path, typer.Option("--input", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=True)] = Path(
        "data/manifests/yawdd.json"
    ),
    participants_info: Annotated[
        Path | None,
        typer.Option("--participants-info", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Index a local YawDD copy for real human yawning validation."""
    manifest = build_yawdd_manifest(input_dir, out, participants_info)
    console.print(f"YawDD clips indexed: {manifest['clip_count']}")
    console.print(f"manifest: {out}")


@datasets_app.command("prepare-dd")
def prepare_dd(
    input_dir: Annotated[Path, typer.Option("--input", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=True)] = Path(
        "data/manifests/dd-database.json"
    ),
    download_index: Annotated[Path | None, typer.Option("--download-index", file_okay=True)] = None,
) -> None:
    """Index the Dryad DD-Database physiological drowsiness EDF files."""
    manifest = build_dd_manifest(input_dir, out, dryad_index_path=download_index)
    console.print(f"DD-Database recordings indexed: {manifest['recording_count']}")
    console.print(f"manifest: {out}")
    if download_index:
        console.print(f"dryad file index: {download_index}")


@datasets_app.command("dd-index")
def dd_index(
    out: Annotated[Path, typer.Option("--out", file_okay=True)] = Path(
        "data/dd-database/dryad-files.json"
    ),
) -> None:
    """Fetch the Dryad DD-Database file index without downloading raw EDF files."""
    index = fetch_dd_dryad_file_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    console.print(f"DD-Database Dryad files indexed: {index['file_count']}")
    console.print(f"index: {out}")


@datasets_app.command("dd-events")
def dd_events(
    input_dir: Annotated[Path, typer.Option("--input", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "runs/dd-database-sensors"
    ),
) -> None:
    """Export drowsiness events from DD-Database annotation files."""
    artifacts = write_dd_sensor_events(input_dir, out)
    for name, path in artifacts.items():
        console.print(f"{name}: {path}")


@datasets_app.command("prepare-uah")
def prepare_uah(
    input_dir: Annotated[Path, typer.Option("--input", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=True)] = Path(
        "data/manifests/uah-driveset.json"
    ),
) -> None:
    """Index a local UAH-DriveSet copy for vehicle-sensor validation."""
    manifest = build_uah_manifest(input_dir, out)
    console.print(f"UAH-DriveSet sessions indexed: {manifest['session_count']}")
    console.print(f"manifest: {out}")


@datasets_app.command("uah-events")
def uah_events(
    input_dir: Annotated[Path, typer.Option("--input", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("runs/uah-driveset-sensors"),
) -> None:
    """Export vehicle-risk events from UAH-DriveSet telemetry files."""
    artifacts = write_uah_vehicle_events(input_dir, out)
    for name, path in artifacts.items():
        console.print(f"{name}: {path}")


if __name__ == "__main__":
    app()
