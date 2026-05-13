# mypy: disable-error-code=untyped-decorator
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from driver_safety.config import load_config
from driver_safety.runtime.runner import analyze_video


def create_app(run_dir: str | Path = "runs/demo") -> Any:
    try:
        from fastapi import FastAPI, File, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "FastAPI dependencies are missing. Install with `pip install ai-driver-safety[api]`."
        ) from exc

    run_path = Path(run_dir)
    app = FastAPI(title="AI Driver Safety API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/sessions/current/events")
    def events() -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], _read_json(run_path / "events.json", default=[]))

    @app.get("/sessions/current/summary")
    def summary() -> dict[str, Any]:
        return cast(dict[str, Any], _read_json(run_path / "summary.json", default={}))

    video_upload = File(...)

    @app.post("/sessions/current/videos")
    async def upload_video(file: UploadFile = video_upload) -> dict[str, str]:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / file.filename
            tmp_path.write_bytes(await file.read())
            artifacts = analyze_video(tmp_path, run_path, load_config())
        return {key: str(value) for key, value in artifacts.items()}

    return app


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
