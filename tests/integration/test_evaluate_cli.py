from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "_wfcllm_evaluate", ROOT / "scripts" / "evaluate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_defaults_to_posthoc_pass_at_1(tmp_path, capsys) -> None:
    source = tmp_path / "correctness.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"task_id": "a", "is_correct": True},
                {"task_id": "a", "is_correct": False},
                {"task_id": "b", "is_correct": True},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "pass_report_posthoc.json"

    assert _module().main([str(source), "--output", str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "pass@1"
    assert payload["k"] == 1
    assert payload["value"] == pytest.approx(0.75)
    assert payload["passed_count"] == 2
    assert payload["total_count"] == 3
    assert payload["posthoc_only"] is True
    assert payload["not_used_for_detection"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_evaluate_rejects_removed_pass_at_k_option(tmp_path) -> None:
    source = tmp_path / "correctness.jsonl"
    source.write_text(
        json.dumps({"task_id": "a", "is_correct": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        _module().main([str(source), "--k", "2"])


def test_evaluate_rejects_non_boolean_correctness(tmp_path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        json.dumps({"task_id": "a", "is_correct": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="boolean is_correct"):
        _module().main([str(source)])
