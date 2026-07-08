from __future__ import annotations

import json
from pathlib import Path

from wfcllm.audit.detector_input_integrity import audit_detector_input_file


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_detector_input_file_accepts_four_field_rows(tmp_path: Path) -> None:
    path = tmp_path / "final_code.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "final_code": "def foo():\n    return 1\n",
            }
        ],
    )

    report = audit_detector_input_file(path)

    assert report["ok"] is True
    assert report["records_checked"] == 1
    assert report["errors"] == []


def test_audit_detector_input_file_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "final_code.jsonl"
    path.write_text(
        "\n"
        + json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "final_code": "def foo():\n    return 1\n",
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    report = audit_detector_input_file(path)

    assert report["ok"] is True
    assert report["records_checked"] == 1
    assert report["errors"] == []


def test_audit_detector_input_file_reports_generated_code(tmp_path: Path) -> None:
    path = tmp_path / "final_code.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "generated_code": "    return 1\n",
            }
        ],
    )

    report = audit_detector_input_file(path)

    assert report["ok"] is False
    assert report["records_checked"] == 1
    assert report["errors"][0]["line"] == 1
    assert "generated_code" in report["errors"][0]["error"]


def test_audit_detector_input_file_reports_json_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "final_code.jsonl"
    path.write_text('{"id": "HumanEval/0"\n', encoding="utf-8")

    report = audit_detector_input_file(path)

    assert report["ok"] is False
    assert report["records_checked"] == 1
    assert report["errors"][0]["line"] == 1
    assert "Expecting" in report["errors"][0]["error"]
