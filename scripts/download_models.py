#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download optional AI Driver Safety model assets.")
    parser.add_argument(
        "--mediapipe-face", action="store_true", help="Download Face Landmarker task."
    )
    parser.add_argument(
        "--phone-detector",
        action="store_true",
        help="Export a YOLO11 COCO phone/object detector to ONNX.",
    )
    parser.add_argument(
        "--phone-model",
        default="yolo11s.pt",
        help="Ultralytics model to export for phone/object detection.",
    )
    parser.add_argument("--out", default="models", help="Model output directory.")
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mediapipe_face:
        target = output_dir / "face_landmarker.task"
        print(f"Downloading MediaPipe Face Landmarker to {target}")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, target)
    if args.phone_detector:
        _export_phone_detector(output_dir, args.phone_model)
    if not args.mediapipe_face and not args.phone_detector:
        parser.print_help()


def _export_phone_detector(output_dir: Path, model_name: str) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Phone detector export needs Ultralytics export dependencies. Run:\n"
            "  python -m pip install ultralytics onnx onnxslim"
        ) from exc

    target = (output_dir / "driver-objects.onnx").resolve()
    labels_target = (output_dir / "driver-objects.labels").resolve()
    cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        model = YOLO(model_name)
        exported = Path(model.export(format="onnx", imgsz=640, simplify=True, opset=12))
        exported_path = exported.resolve() if not exported.is_absolute() else exported
        if exported_path != target.resolve():
            shutil.copy2(exported_path, target)
    finally:
        os.chdir(cwd)
    labels_target.write_text("\n".join(COCO_LABELS) + "\n", encoding="utf-8")
    print(f"Exported phone/object detector to {target}")
    print(f"Wrote COCO labels to {labels_target}")


if __name__ == "__main__":
    main()
