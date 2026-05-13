#!/usr/bin/env python
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download optional AI Driver Safety model assets.")
    parser.add_argument("--mediapipe-face", action="store_true", help="Download Face Landmarker task.")
    parser.add_argument("--out", default="models", help="Model output directory.")
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mediapipe_face:
        target = output_dir / "face_landmarker.task"
        print(f"Downloading MediaPipe Face Landmarker to {target}")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, target)
    if not args.mediapipe_face:
        parser.print_help()


if __name__ == "__main__":
    main()

