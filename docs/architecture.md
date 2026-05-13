# Architecture

AI Driver Safety is organized around a deterministic frame pipeline.

## Flow

```text
FrameSource -> DriverSafetyPipeline -> DetectionEvent[] -> SessionRecorder -> Reports
                         |                       |
                         |                       +-> AlertPolicy
                         +-> FaceDetector / ObjectDetector
```

## Components

- `driver_safety.core`: typed events, driver states, risk scoring, smoothing, alert cooldowns.
- `driver_safety.vision`: detector interfaces, MediaPipe adapter, Haar fallback, synthetic detector, signal metrics, optional ONNX object detection.
- `driver_safety.io`: video/webcam sources and annotated overlay writer.
- `driver_safety.runtime`: video and webcam loops.
- `driver_safety.reporting`: JSON, CSV, and HTML exports.
- `driver_safety.api`: optional FastAPI service for local review.
- `apps/studio`: React/Vite dashboard for session review.

## Detector Strategy

The runtime depends on interfaces instead of one model implementation:

- `FaceLandmarkDetector.detect(frame) -> FaceObservation[]`
- `ObjectDetector.detect(frame) -> ObjectObservation[]`

The default config uses `vision.provider: auto`, which attempts MediaPipe Face Landmarker when the model is present, then falls back to OpenCV Haar face detection for basic face presence. The synthetic detector is for demo and tests.

## Event Model

Each event includes:

- timestamp
- frame index
- signal
- state
- score
- severity
- message
- optional bbox and landmarks
- metadata

This keeps the pipeline usable from CLI, API, reports, and dashboards without coupling those surfaces to detector-specific outputs.

