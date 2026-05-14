# Edge Deployment

AI Driver Safety is designed to run locally first. The default reliable path is CPU video analysis with OpenCV and MediaPipe. ONNX Runtime is the optional deployment path for object detectors and edge accelerators.

## CPU Profile

Use:

```bash
ai-driver-safety analyze --video data/approved-demo/driver-yawning.mp4 --config configs/edge-cpu.yaml --out runs/edge-cpu
```

`configs/edge-cpu.yaml` processes every second frame and writes output at 15 FPS.

## MediaPipe Model

```bash
python scripts/download_models.py --mediapipe-face
```

## ONNX Object Detector

Export detector assets under `models/`:

```text
models/
  driver-objects.onnx
  driver-objects.labels
```

```bash
python -m pip install ultralytics onnx onnxslim onnxruntime
python scripts/download_models.py --phone-detector --phone-model yolo11s.pt
```

Then enable:

```yaml
object_detector:
  enabled: true
  provider: onnx
```

For the phone-use demo, use `configs/phone-demo.yaml`. The config keeps the core runtime ONNX-based and maps COCO `cell phone` detections into the `phone_use` signal.

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
- fusion model name
