from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "merge_sawr_final_code.py"


def _load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_sawr_final_code", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_merge_final_code_files_sanitizes_and_overwrites_by_id(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps(
            {
                "id": "HumanEval/1",
                "dataset": "humaneval",
                "prompt": "def old():\n",
                "final_code": "def old():\n    return 1\n",
                "retry_trace": ["forbidden"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": "def zero():\n",
                "generated_code": "def zero():\n    return 0\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "id": "HumanEval/1",
                "dataset": "humaneval",
                "prompt": "def new():\n",
                "final_code": "def new():\n    return 2\n",
                "audit": {"forbidden": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    module = _load_merge_module()
    rows, analysis = module.merge_final_code_files([first, second])

    assert [row["id"] for row in rows] == ["HumanEval/0", "HumanEval/1"]
    assert rows[0] == {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def zero():\n",
        "final_code": "def zero():\n    return 0\n",
    }
    assert rows[1] == {
        "id": "HumanEval/1",
        "dataset": "humaneval",
        "prompt": "def new():\n",
        "final_code": "def new():\n    return 2\n",
    }
    assert analysis["merged_count"] == 2
    assert analysis["duplicate_count"] == 1
