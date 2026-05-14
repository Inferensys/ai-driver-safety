from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from driver_safety.config import load_config
from driver_safety.reporting.exports import write_events_csv, write_html_report
from driver_safety.runtime.runner import analyze_video, run_webcam

app = typer.Typer(no_args_is_help=True, help="AI Driver Safety local driver monitoring CLI.")
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
    run_dir: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)] = Path(
        "runs/demo"
    ),
    format: Annotated[str, typer.Option("--format")] = "html,json,csv",
) -> None:
    """Regenerate report exports from a completed run directory."""
    events_path = run_dir / "events.json"
    summary_path = run_dir / "summary.json"
    if not events_path.exists() or not summary_path.exists():
        raise typer.BadParameter(
            f"Run directory must contain events.json and summary.json: {run_dir}"
        )
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


if __name__ == "__main__":
    app()
