from __future__ import annotations

import csv
import json
from pathlib import Path

from jinja2 import Template

from driver_safety.core.models import DetectionEvent, SessionSummary


def export_run_artifacts(
    output_dir: str | Path,
    *,
    events: list[DetectionEvent],
    summary: SessionSummary,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "events.json"
    summary_path = output / "summary.json"
    csv_path = output / "events.csv"
    html_path = output / "report.html"
    events_path.write_text(
        json.dumps([event.to_dict() for event in events], indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    write_events_csv(csv_path, events)
    write_html_report(html_path, events=events, summary=summary)
    return {
        "events": events_path,
        "summary": summary_path,
        "csv": csv_path,
        "html": html_path,
    }


def write_events_csv(path: str | Path, events: list[DetectionEvent]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "timestamp",
                "frame_index",
                "signal",
                "state",
                "score",
                "severity",
                "message",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "timestamp": f"{event.timestamp:.3f}",
                    "frame_index": event.frame_index,
                    "signal": event.signal,
                    "state": event.state.value,
                    "score": f"{event.score:.4f}",
                    "severity": event.severity.value,
                    "message": event.message,
                }
            )


def write_html_report(
    path: str | Path,
    *,
    events: list[DetectionEvent],
    summary: SessionSummary,
) -> None:
    rendered = Template(REPORT_TEMPLATE).render(
        events=[event.to_dict() for event in events],
        summary=summary.to_dict(),
    )
    Path(path).write_text(rendered, encoding="utf-8")


REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Driver Safety Report</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f4f6f5; color: #17201d; }
    main { max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; border-bottom: 1px solid #d9dfdd; padding-bottom: 20px; }
    h1 { margin: 0; font-size: 30px; line-height: 1.1; }
    .muted { color: #61706b; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 24px 0; }
    .metric, .panel { background: #fff; border: 1px solid #dde4e1; border-radius: 8px; padding: 16px; }
    .metric strong { display: block; font-size: 26px; }
    .timeline { height: 120px; display: flex; align-items: end; gap: 2px; border-bottom: 1px solid #ccd4d1; }
    .bar { width: 5px; background: #2f9f73; min-height: 2px; }
    .bar.warn { background: #df8b32; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid #e2e8e5; text-align: left; vertical-align: top; }
    th { color: #40514b; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    @media (max-width: 780px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } header { display: block; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>AI Driver Safety Session Report</h1>
      <p class="muted">{{ summary.source }}</p>
    </div>
    <div class="muted">Session {{ summary.session_id }}</div>
  </header>
  <section class="grid">
    <div class="metric"><span class="muted">Duration</span><strong>{{ summary.duration_seconds }}s</strong></div>
    <div class="metric"><span class="muted">Frames</span><strong>{{ summary.processed_frames }}</strong></div>
    <div class="metric"><span class="muted">Unsafe Interval</span><strong>{{ summary.longest_unsafe_interval_seconds }}s</strong></div>
    <div class="metric"><span class="muted">Events</span><strong>{{ events|length }}</strong></div>
  </section>
  <section class="panel">
    <h2>Risk Timeline</h2>
    <div class="timeline">
    {% for point in summary.risk_timeline[::3] %}
      <div class="bar {% if point.risk_score >= 0.45 %}warn{% endif %}" style="height: {{ 10 + point.risk_score * 100 }}px"></div>
    {% endfor %}
    </div>
  </section>
  <section class="panel" style="margin-top: 18px;">
    <h2>Events</h2>
    <table>
      <thead><tr><th>Time</th><th>Signal</th><th>Severity</th><th>Score</th><th>Message</th></tr></thead>
      <tbody>
      {% for event in events[:120] %}
        <tr>
          <td>{{ "%.2f"|format(event.timestamp) }}s</td>
          <td>{{ event.signal }}</td>
          <td>{{ event.severity }}</td>
          <td>{{ "%.2f"|format(event.score) }}</td>
          <td>{{ event.message }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
</main>
</body>
</html>
"""

