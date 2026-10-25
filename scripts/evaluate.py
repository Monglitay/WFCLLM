#!/usr/bin/env python
"""Posthoc Pass@1 reporting for a completed Gate-only reproduction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.evaluation.code_execution import compute_pass_at_1, load_jsonl_records  # noqa: E402
from wfcllm.datasets.loaders.local import load_test_cases  # noqa: E402

_STRICT_GENERATED_KEYS = frozenset({"id", "dataset", "prompt", "final_code"})
_PYTHON_EXECUTION_TIMEOUT_SECONDS = 10


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute posthoc Pass@1")
    parser.add_argument("inputs", nargs="*", help="posthoc correctness JSONL")
    parser.add_argument(
        "--generated-input",
        help="strict four-field generated final_code JSONL to evaluate first",
    )
    parser.add_argument("--dataset", help="dataset name for --generated-input mode")
    parser.add_argument("--language", choices=("python", "cpp", "java", "js"), default="python")
    parser.add_argument(
        "--dataset-path",
        default="data/datasets",
        help="local dataset cache root for --generated-input mode",
    )
    parser.add_argument(
        "--correctness-output",
        help="posthoc correctness JSONL path written by --generated-input mode",
    )
    parser.add_argument(
        "--sandbox-stdlib-path",
        help="read-only standard-library snapshot required by Python posthoc execution",
    )
    parser.add_argument("--output", help="optional pass_report_posthoc.json path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    inputs = list(args.inputs)
    if args.generated_input is not None:
        if args.dataset is None:
            raise ValueError("--dataset is required with --generated-input")
        if args.correctness_output is None:
            raise ValueError("--correctness-output is required with --generated-input")
        if args.language == "python" and args.sandbox_stdlib_path is None:
            raise ValueError("--sandbox-stdlib-path is required with --generated-input")
        if args.language == "cpp":
            if args.dataset != "humanevalpack":
                raise ValueError("C++ evaluation requires dataset=humanevalpack")
            _write_cpp_correctness(
                generated_input=Path(args.generated_input), dataset_path=args.dataset_path,
                correctness_output=Path(args.correctness_output),
            )
        elif args.language in {"java", "js"}:
            if args.dataset != "humanevalpack":
                raise ValueError("Java/JavaScript evaluation requires dataset=humanevalpack")
            _write_native_correctness(
                generated_input=Path(args.generated_input), dataset_path=args.dataset_path,
                correctness_output=Path(args.correctness_output), language=args.language,
            )
        else:
            _write_generated_correctness(
                generated_input=Path(args.generated_input),
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                correctness_output=Path(args.correctness_output),
                sandbox_stdlib_path=Path(args.sandbox_stdlib_path),
            )
        inputs.append(args.correctness_output)
    if not inputs:
        raise ValueError("provide correctness JSONL input or --generated-input")

    rows: list[dict] = []
    for path in inputs:
        rows.extend(load_jsonl_records(path))
    payload = {
        "schema_version": "wfcllm-posthoc-pass-report/v1",
        "metric": "pass@1",
        "k": 1,
        "value": compute_pass_at_1(rows),
        "sample_count": len(rows),
        "passed_count": sum(row["is_correct"] is True for row in rows),
        "total_count": len(rows),
        "posthoc_only": True,
        "not_used_for_generation": True,
        "not_used_for_retry": True,
        "not_used_for_selection": True,
        "not_used_for_calibration": True,
        "not_used_for_detection": True,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _write_generated_correctness(
    *,
    generated_input: Path,
    dataset: str,
    dataset_path: str,
    correctness_output: Path,
    sandbox_stdlib_path: Path,
) -> None:
    if not sandbox_stdlib_path.is_file():
        raise ValueError(f"sandbox stdlib snapshot does not exist: {sandbox_stdlib_path}")
    generated_rows = _load_strict_generated_rows(generated_input)
    test_cases = load_test_cases(dataset, dataset_path)
    correctness_output.parent.mkdir(parents=True, exist_ok=True)
    with correctness_output.open("w", encoding="utf-8") as handle:
        for row in generated_rows:
            task_id = row["id"]
            if task_id not in test_cases:
                raise ValueError(f"generated task id missing from dataset tests: {task_id}")
            case = test_cases[task_id]
            result = _evaluate_python_row(
                final_code=row["final_code"],
                test_code=case.test_code,
                entry_point=case.entry_point,
            )
            output_row = {
                "dataset": dataset,
                "error": result["error"],
                "id": task_id,
                "is_correct": result["is_correct"],
                "posthoc_only": True,
                "task_id": task_id,
            }
            handle.write(
                json.dumps(output_row, sort_keys=True, allow_nan=False) + "\n"
            )


def _write_cpp_correctness(
    *, generated_input: Path, dataset_path: str, correctness_output: Path,
) -> None:
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter
    from wfcllm.evaluation.cpp_execution import evaluate_cpp_row, require_cpp_toolchain

    toolchain = require_cpp_toolchain()
    rows = _load_strict_generated_rows(generated_input)
    cases = {
        str(row["task_id"]): str(row["test"])
        for row in HumanEvalPackAdapter(dataset_path)._load_split("cpp")
    }
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("C++ generated input requires one completion per task ID")
    for row in rows:
        if row["dataset"] != "humanevalpack" or row["id"] not in cases:
            raise ValueError(f"C++ input task is missing from original tests: {row['id']}")
    correctness_output.parent.mkdir(parents=True, exist_ok=True)
    with correctness_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = evaluate_cpp_row(
                final_code=row["final_code"], test_code=cases[row["id"]], toolchain=toolchain,
            )
            payload = {**result, "id": row["id"], "task_id": row["id"],
                       "dataset": "humanevalpack", "language": "cpp", "posthoc_only": True}
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


def _write_native_correctness(
    *, generated_input: Path, dataset_path: str, correctness_output: Path, language: str,
) -> None:
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter
    from wfcllm.evaluation.humanevalpack_execution import evaluate_native_row, require_native_toolchain

    toolchain = require_native_toolchain(language)
    rows = _load_strict_generated_rows(generated_input)
    cases = {
        str(row["task_id"]): str(row["test"])
        for row in HumanEvalPackAdapter(dataset_path)._load_split(language)
    }
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("native generated input requires one completion per task ID")
    for row in rows:
        if row["dataset"] != "humanevalpack" or row["id"] not in cases:
            raise ValueError(f"native input task is missing from original tests: {row['id']}")
    correctness_output.parent.mkdir(parents=True, exist_ok=True)
    with correctness_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = evaluate_native_row(
                final_code=row["final_code"], test_code=cases[row["id"]], toolchain=toolchain,
            )
            payload = {**result, "id": row["id"], "task_id": row["id"],
                       "dataset": "humanevalpack", "language": language, "posthoc_only": True}
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


def _load_strict_generated_rows(path: Path) -> list[dict[str, str]]:
    rows = load_jsonl_records(path)
    strict_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        keys = set(row)
        if keys != _STRICT_GENERATED_KEYS:
            raise ValueError(
                f"generated input row {index} must contain exactly "
                f"{sorted(_STRICT_GENERATED_KEYS)}, got {sorted(keys)}"
            )
        strict_row: dict[str, str] = {}
        for key in sorted(_STRICT_GENERATED_KEYS):
            value = row[key]
            if not isinstance(value, str):
                raise ValueError(f"generated input row {index} field {key} must be a string")
            strict_row[key] = value
        strict_rows.append(strict_row)
    return strict_rows


def _evaluate_python_row(
    *,
    final_code: str,
    test_code: str,
    entry_point: str | None,
) -> dict[str, Any]:
    script = _python_check_script(final_code, test_code, entry_point)
    with tempfile.TemporaryDirectory(prefix="wfcllm-posthoc-") as tmp:
        path = Path(tmp) / "check.py"
        path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=_PYTHON_EXECUTION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"is_correct": False, "error": "execution timed out"}
    if completed.returncode == 0:
        return {"is_correct": True, "error": None}
    stderr = completed.stderr.strip().splitlines()
    stdout = completed.stdout.strip().splitlines()
    message = (stderr or stdout or [f"exit status {completed.returncode}"])[-1]
    return {"is_correct": False, "error": message[:500]}


def _python_check_script(
    final_code: str,
    test_code: str,
    entry_point: str | None,
) -> str:
    payload = {
        "entry_point": entry_point,
        "final_code": final_code,
        "test_code": test_code,
    }
    return textwrap.dedent(
        f"""
        import json

        payload = json.loads({json.dumps(json.dumps(payload, allow_nan=False))})
        namespace = {{}}
        exec(payload["final_code"], namespace)
        exec(payload["test_code"], namespace)
        entry_point = payload["entry_point"]
        if entry_point is not None:
            candidate = namespace.get(entry_point)
            if not callable(candidate):
                raise AssertionError(f"missing callable entry point: {{entry_point}}")
            check = namespace.get("check")
            if not callable(check):
                raise AssertionError("missing callable check function")
            check(candidate)
        """
    ).lstrip()


if __name__ == "__main__":
    raise SystemExit(main())
