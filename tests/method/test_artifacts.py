from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.method.artifacts import (
    FinalCodeRecord,
    RunPaths,
    build_run_paths,
    write_jsonl_rows,
)


def test_final_code_record_exact_four_fields() -> None:
    record = FinalCodeRecord(
        id="HumanEval/0",
        dataset="humaneval",
        prompt="def foo():\n",
        final_code="def foo():\n    return 1\n",
    )

    assert record.to_dict() == {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
    }


def test_final_code_record_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="id"):
        FinalCodeRecord(id="", dataset="humaneval", prompt="", final_code="")


def test_build_run_paths_uses_new_layout(tmp_path: Path) -> None:
    paths = build_run_paths(tmp_path, "20260708_153000_evidence_retry_seed7x3")

    assert isinstance(paths, RunPaths)
    assert (
        paths.final_code_input
        == tmp_path
        / "20260708_153000_evidence_retry_seed7x3"
        / "inputs"
        / "final_code.jsonl"
    )
    assert (
        paths.audit_log
        == tmp_path
        / "20260708_153000_evidence_retry_seed7x3"
        / "generation"
        / "audit.jsonl"
    )
    assert (
        paths.detector_integrity
        == tmp_path
        / "20260708_153000_evidence_retry_seed7x3"
        / "audit"
        / "detector_input_integrity.json"
    )


def test_write_jsonl_rows_rejects_non_json_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON-safe"):
        write_jsonl_rows(tmp_path / "rows.jsonl", [{"score": float("nan")}])


def test_write_jsonl_rows_writes_utf8_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl_rows(path, [{"id": "a", "text": "中文"}])

    assert json.loads(path.read_text(encoding="utf-8").strip()) == {
        "id": "a",
        "text": "中文",
    }
