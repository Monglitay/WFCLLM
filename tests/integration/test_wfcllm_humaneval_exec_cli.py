from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.wfcllm_humaneval_exec as exec_cli


def _row(sample_id: str, code: str) -> dict[str, str]:
    return {
        "id": sample_id,
        "dataset": "humaneval",
        "prompt": "def target(x):\n",
        "final_code": code,
    }


def _case():
    return SimpleNamespace(
        test_code="def check(candidate):\n    assert candidate(2) == 3",
        entry_point="target",
    )


def test_evaluate_row_executes_saved_final_code() -> None:
    result = exec_cli.evaluate_row(
        _row("HumanEval/0", "def target(x):\n    return x + 1\n"),
        _case(),
        timeout=2.0,
    )

    assert result["id"] == "HumanEval/0"
    assert result["passed"] is True
    assert result["reason"] == "passed"


def test_evaluate_row_rejects_non_strict_and_syntax_invalid_rows() -> None:
    invalid = _row("HumanEval/0", "def target(x):\n    return (\n")
    result = exec_cli.evaluate_row(invalid, _case(), timeout=2.0)
    assert result["passed"] is False
    assert result["reason"] == "syntax_error"

    with_extra = {**_row("HumanEval/0", "def target(x):\n    return x\n"), "label": 1}
    result = exec_cli.evaluate_row(with_extra, _case(), timeout=2.0)
    assert result["passed"] is False
    assert result["reason"] == "invalid_final_code_contract"


def test_exec_cli_writes_versioned_results_and_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "final.jsonl"
    result_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _row("HumanEval/0", "def target(x):\n    return x + 1\n"),
                _row("HumanEval/1", "def target(x):\n    return x\n"),
            )
        ),
        encoding="utf-8",
    )
    cases = {"HumanEval/0": _case(), "HumanEval/1": _case()}

    with patch("scripts.wfcllm_humaneval_exec.load_test_cases", return_value=cases):
        rc = exec_cli.main(
            [
                "--input",
                str(input_path),
                "--dataset",
                "humaneval",
                "--dataset-path",
                str(tmp_path),
                "--output-results",
                str(result_path),
                "--output-summary",
                str(summary_path),
                "--timeout",
                "2",
            ]
        )

    assert rc == 0
    results = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert [row["passed"] for row in results] == [True, False]
    assert all(row["schema_version"] == "wfcllm-execution-result/v1" for row in results)
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "wfcllm-execution-summary/v1"
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["pass_at_1"] == 0.5
    assert summary["reason_counts"] == {"passed": 1, "failed_tests": 1}


def test_exec_cli_records_timeout_without_stopping_other_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "final.jsonl"
    result_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps(
            _row("HumanEval/0", "def target(x):\n    while True:\n        pass\n")
        )
        + "\n",
        encoding="utf-8",
    )

    with patch(
        "scripts.wfcllm_humaneval_exec.load_test_cases",
        return_value={"HumanEval/0": _case()},
    ):
        rc = exec_cli.main(
            [
                "--input",
                str(input_path),
                "--output-results",
                str(result_path),
                "--output-summary",
                str(summary_path),
                "--timeout",
                "0.1",
            ]
        )

    assert rc == 0
    result = json.loads(result_path.read_text())
    assert result["passed"] is False
    assert result["reason"] == "timeout"
