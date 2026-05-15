"""Run-state persistence: read/write data/run_state.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Phase name source-of-truth (lifted unchanged from run.py to preserve schema).
PHASES = ["encoder", "watermark", "extract"]
OPTIONAL_PHASES = ["pretrain", "generate-negative", "token-channel-train", "build-entropy-profile"]
ALL_PHASES = PHASES + OPTIONAL_PHASES

DEFAULT_STATE_FILE = Path("data/run_state.json")


class RunStateManager:
    """Persistent phase completion tracker.

    Schema is preserved byte-for-byte from the original `RunState` class in run.py
    so existing data/run_state.json files load cleanly.
    """

    def __init__(self, path: Path = DEFAULT_STATE_FILE) -> None:
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {phase: {"done": False} for phase in ALL_PHASES}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def is_done(self, phase: str) -> bool:
        return self._data.get(phase, {}).get("done", False)

    def get(self, phase: str, key: str) -> str | None:
        return self._data.get(phase, {}).get(key)

    def mark_done(self, phase: str, **kwargs) -> None:
        self._data[phase] = {
            "done": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._save()

    def reset(self) -> None:
        self._data = {phase: {"done": False} for phase in ALL_PHASES}
        self._save()

    def status(self) -> dict:
        return {
            phase: self._data.get(phase, {"done": False})
            for phase in ALL_PHASES
        }
