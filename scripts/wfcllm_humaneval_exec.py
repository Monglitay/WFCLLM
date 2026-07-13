#!/usr/bin/env python
"""Execute strict saved final-code rows against local benchmark tests."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.datasets.loaders.local import load_test_cases  # noqa: E402
from wfcllm.detection.pipeline import (  # noqa: E402
    validate_final_code_detector_input_record,
)

EXECUTION_RESULT_SCHEMA_VERSION = "wfcllm-execution-result/v1"
EXECUTION_SUMMARY_SCHEMA_VERSION = "wfcllm-execution-summary/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run saved final code against local HumanEval/MBPP tests.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--dataset-path", default="data/datasets")
    parser.add_argument("--output-results", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at line {exc.lineno}") from exc
    return rows


def _run_python_source(
    source: str,
    timeout: float,
) -> tuple[bool, str, str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(source)
        temp_path = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        diagnostic = stderr[-2000:] if stderr else stdout[-2000:]
        return (
            result.returncode == 0,
            "passed" if result.returncode == 0 else "failed_tests",
            diagnostic,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return False, "timeout", stderr[-1800:]
    finally:
        temp_path.unlink(missing_ok=True)


def _result(
    *,
    sample_id: str,
    passed: bool,
    reason: str,
    stderr_tail: str = "",
) -> dict[str, Any]:
    return {
        "artifact_type": "wfcllm_execution_result",
        "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
        "id": sample_id,
        "passed": passed,
        "reason": reason,
        "stderr_tail": stderr_tail,
    }


def evaluate_row(
    row: dict[str, Any],
    test_case: Any,
    timeout: float,
) -> dict[str, Any]:
    sample_id = str(row.get("id", ""))
    try:
        validate_final_code_detector_input_record(row)
    except ValueError as exc:
        return _result(
            sample_id=sample_id,
            passed=False,
            reason="invalid_final_code_contract",
            stderr_tail=str(exc),
        )
    final_code = str(row["final_code"])
    if not final_code:
        return _result(
            sample_id=sample_id,
            passed=False,
            reason="missing_final_code",
        )
    try:
        ast.parse(final_code)
    except SyntaxError as exc:
        return _result(
            sample_id=sample_id,
            passed=False,
            reason="syntax_error",
            stderr_tail=f"{exc.__class__.__name__}: {exc}",
        )
    if test_case is None:
        return _result(
            sample_id=sample_id,
            passed=False,
            reason="missing_test_case",
        )
    if test_case.entry_point is not None:
        test_source = f"{test_case.test_code}\ncheck({test_case.entry_point})\n"
    else:
        test_source = str(test_case.test_code)
    passed, reason, stderr_tail = _run_python_source(
        f"{final_code}\n{test_source}\n",
        timeout,
    )
    return _result(
        sample_id=sample_id,
        passed=passed,
        reason=reason,
        stderr_tail=stderr_tail,
    )


def _write_results(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_summary(
    *,
    input_path: str | Path,
    dataset: str,
    timeout: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for row in results:
        reason = str(row["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    passed = sum(bool(row["passed"]) for row in results)
    total = len(results)
    return {
        "artifact_type": "wfcllm_execution_summary",
        "schema_version": EXECUTION_SUMMARY_SCHEMA_VERSION,
        "input": str(Path(input_path)),
        "input_sha256": _sha256(input_path),
        "dataset": dataset,
        "timeout_seconds": timeout,
        "total": total,
        "passed": passed,
        "pass_at_1": passed / total if total else 0.0,
        "reason_counts": reason_counts,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        rows = _load_jsonl(args.input)
        test_cases = load_test_cases(args.dataset, args.dataset_path)
        results = [
            evaluate_row(
                row,
                test_cases.get(str(row.get("id", ""))),
                args.timeout,
            )
            for row in rows
        ]
        _write_results(args.output_results, results)
        summary = _build_summary(
            input_path=args.input,
            dataset=args.dataset,
            timeout=args.timeout,
            results=results,
        )
        summary_path = Path(args.output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
