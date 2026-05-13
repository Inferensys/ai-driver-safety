# Contributing

AI Driver Safety is a local-first driver monitoring project. Contributions should improve reliability, reproducibility, and safety clarity before adding new model complexity.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For the optional MediaPipe and ONNX paths:

```bash
python -m pip install -e ".[vision,onnx,api,dev]"
python scripts/download_models.py --mediapipe-face
```

## Checks

```bash
ruff check .
mypy driver_safety
pytest
```

## Contribution Rules

- Do not commit large model weights or private driving footage.
- Keep datasets behind documented download scripts or links unless redistribution rights are clear.
- Keep safety claims conservative: this is a driver monitoring reference application, not certified automotive safety software.
- Add tests for new detectors, scoring logic, configs, report formats, and CLI behavior.
- Prefer explicit config over hardcoded thresholds.

