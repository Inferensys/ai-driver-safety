from pathlib import Path

import pytest

from driver_safety.config import load_config


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.vision.provider == "auto"
    assert config.thresholds.eye_aspect_ratio > 0


def test_invalid_config_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("vision:\n  process_every_n_frames: 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
