"""统一运行入口：支持全流程、单阶段运行与断点续跑。

用法：
    python run.py                          # 全流程
    python run.py --phase encoder          # 只跑阶段一
    python run.py --status                 # 查看状态
    python run.py --reset                  # 清除断点
    python run.py --phase encoder --force  # 强制重跑
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = ["encoder", "watermark", "extract"]
DEFAULT_STATE_FILE = Path("data/run_state.json")


class RunState:
    """断点状态管理：读写 data/run_state.json。"""

    def __init__(self, path: Path = DEFAULT_STATE_FILE):
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {phase: {"done": False} for phase in PHASES}

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
        self._data = {phase: {"done": False} for phase in PHASES}
        self._save()

    def status(self) -> dict:
        return {
            phase: {
                "done": self._data.get(phase, {}).get("done", False),
                **{k: v for k, v in self._data.get(phase, {}).items() if k != "done"},
            }
            for phase in PHASES
        }
