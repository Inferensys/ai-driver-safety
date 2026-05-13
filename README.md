# AI Driver Safety

AI Driver Safety is a local-first **Intelligent Driver Monitoring System** for autonomous and assisted driving use cases. It analyzes camera or video input for drowsiness, eye closure, yawning, distraction, phone use, missing-face states, and driver risk over time.

The project keeps the original framing: computer vision, driver activity recognition, heart-rate and driving-style extension points, fuzzy/risk scoring, and real-time alerts. The revamp turns the old script collection into a package, CLI, report generator, and review dashboard.

> This is an open-source reference application, not certified automotive safety software.

## What It Does

- Analyzes a webcam or video file frame by frame.
- Emits typed driver-state events: `attentive`, `eyes_closed`, `drowsy`, `yawning`, `distracted`, `phone_use`, and `face_missing`.
- Renders an annotated MP4 with overlays.
- Exports `events.json`, `summary.json`, `events.csv`, and `report.html`.
- Provides a local Studio dashboard for reviewing sessions and event timelines.
- Supports MediaPipe Face Landmarker and optional ONNX object detection hooks while keeping model weights out of the repo.

## Quickstart

```bash
git clone https://github.com/prasad-kumkar/ai-driver-safety.git
cd ai-driver-safety
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Generate the repo-safe synthetic demo and run analysis:

```bash
python scripts/make_demo_assets.py
ai-driver-safety analyze --video samples/demo-driving.mp4 --config configs/synthetic-demo.yaml --out runs/demo
```

Open the report:

```bash
open runs/demo/report.html
```

Run with a real video:

```bash
python scripts/download_models.py --mediapipe-face
python -m pip install -e ".[vision,onnx,api]"
ai-driver-safety analyze --video path/to/driving-video.mp4 --config configs/default.yaml --out runs/real-video
```

Run webcam mode:

```bash
ai-driver-safety run --source webcam --config configs/default.yaml
```

## CLI

```bash
ai-driver-safety analyze --video samples/demo-driving.mp4 --out runs/demo
ai-driver-safety run --source webcam --config configs/default.yaml
ai-driver-safety report --run runs/demo --format html,json,csv
ai-driver-safety studio --run runs/demo
```

## Python API

```python
from driver_safety import create_pipeline, load_config
from driver_safety.core import FramePacket

config = load_config("configs/default.yaml")
pipeline = create_pipeline(config)
result = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))
```

## Project Structure

```text
driver_safety/
  core/        events, scoring, smoothing, alert policies
  vision/      face landmarks, eye closure, yawn, head offset, object detector hooks
  io/          video/webcam sources and overlays
  runtime/     video and webcam processing loops
  reporting/   JSON, CSV, and HTML exports
  api/         optional FastAPI service
apps/studio/   React/Vite dashboard
configs/       threshold and runtime profiles
legacy/        original scripts and assets
docs/          architecture, datasets, deployment, plugin guide
```

## How It Works

1. **Data acquisition**: webcam, video file, or future RTSP source.
2. **Pre-processing**: frame timestamps, optional frame skipping, detector setup.
3. **Vision signals**: face landmarks, eye aspect ratio, mouth/yawn score, head offset, missing-face state, optional phone/object detection.
4. **Driver state classification**: smoothed signals become driver-state events.
5. **Risk scoring**: weighted fuzzy-style risk score is computed per frame.
6. **Alerting and reporting**: event timeline, annotated video, JSON/CSV/HTML exports.

## Studio Dashboard

```bash
cd apps/studio
npm install
npm run dev
```

The Studio dashboard is a local review surface for event timelines, signal scores, current driver state, alert stack, and exported artifacts.

## Models

Model weights are not part of the active runtime tree.

```bash
python scripts/download_models.py --mediapipe-face
```

Optional ONNX object detector models should be placed under `models/` and referenced from config:

```yaml
object_detector:
  enabled: true
  provider: onnx
  model_path: models/driver-objects.onnx
  labels_path: models/driver-objects.labels
```

## Validation Datasets

Use public datasets for validation and benchmarking, subject to each dataset's license and access rules:

- NTHU Driver Drowsiness Detection Dataset
- YawDD
- State Farm Distracted Driver Detection
- Drive&Act

See `docs/datasets.md`.

## Development

```bash
ruff check .
mypy driver_safety
pytest
```

## Safety Note

AI Driver Safety can support research, prototyping, and product demos. It should not be presented as a certified automotive safety system, and it should not be used as the only safety layer in a real vehicle.

