# Edge Deployment

AI Driver Safety is designed to run locally first. The default reliable path is CPU video analysis with OpenCV and MediaPipe. ONNX Runtime is the optional deployment path for object detectors and edge accelerators.

## CPU Profile

Use:

```bash
ai-driver-safety analyze --video samples/demo-driving.mp4 --config configs/edge-cpu.yaml --out runs/edge-cpu
```

`configs/edge-cpu.yaml` processes every second frame and writes output at 15 FPS.

## MediaPipe Model

```bash
python scripts/download_models.py --mediapipe-face
```

## ONNX Object Detector

Place exported detector assets under `models/`:

```text
models/
  driver-objects.onnx
  driver-objects.labels
```

Then enable:

```yaml
object_detector:
  enabled: true
  provider: onnx
```

## YOLO11 Export Direction

YOLO11 can be used to train phone/object distraction detectors, then exported to ONNX. Keep the runtime interface ONNX-based so the package can run across CPU, GPU, and edge accelerators without binding the core runtime to a training framework.

## Runtime Metrics

Every run summary includes:

- source FPS
- average latency
- p95 latency
- estimated runtime FPS
- face detector provider
- object detector provider

