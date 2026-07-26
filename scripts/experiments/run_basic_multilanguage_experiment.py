#!/usr/bin/env python3
"""Run a lightweight end-to-end multi-language experiment.

This path is intentionally simple: it verifies dataset/language wiring and
artifact contracts without loading generation models or training a gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.audit import audit_detector_input_file  # noqa: E402
from wfcllm.cli.experiment_preflight import load_and_validate_experiment_config  # noqa: E402
from wfcllm.datasets import get as get_dataset  # noqa: E402
from wfcllm.generation.outputs import write_final_code_rows  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run a basic WFCLLM experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language", choices=["cpp", "java"], required=True)
    parser.add_argument("--dataset", choices=["humanevalpack"], required=True)
    parser.add_argument("--profile", choices=["fast", "full"], required=True)
    parser.add_argument("--dataset-path", default="data/datasets")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--sample-offset", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        run_basic_experiment(args)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


def run_basic_experiment(args: argparse.Namespace) -> None:
    if args.sample_limit <= 0:
        raise ValueError("sample-limit must be positive")
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be non-negative")

    load_and_validate_experiment_config(
        args.config,
        args.language,
        args.dataset,
        args.profile,
    )
    adapter = get_dataset(args.dataset)
    try:
        adapter = type(adapter)(dataset_path=args.dataset_path)
    except TypeError:
        pass
    if not adapter.supports(args.language):
        raise ValueError(
            f"dataset {args.dataset!r} does not support language={args.language!r}"
        )

    samples = list(adapter.iter_samples(language=args.language))
    selected = samples[args.sample_offset : args.sample_offset + args.sample_limit]
    if not selected:
        raise ValueError("no samples selected for basic experiment")

    run_dir = args.run_dir
    inputs_dir = run_dir / "inputs"
    generation_dir = run_dir / "generation"
    calibration_dir = run_dir / "calibration"
    detection_dir = run_dir / "detection"
    reports_dir = run_dir / "reports"
    audit_dir = run_dir / "audit"
    for path in (inputs_dir, generation_dir, calibration_dir, detection_dir, reports_dir, audit_dir):
        path.mkdir(parents=True, exist_ok=True)

    final_rows = [
        {
            "id": sample.task_id,
            "dataset": args.dataset,
            "prompt": sample.prompt,
            "final_code": _basic_final_code(sample.prompt, sample.canonical_solution),
        }
        for sample in selected
    ]
    write_final_code_rows(inputs_dir / "final_code.jsonl", final_rows)

    generation_rows = [
        {
            "id": row["id"],
            "dataset": args.dataset,
            "language": args.language,
            "basic_experiment": True,
            "model_loaded": False,
            "source": "canonical_solution",
        }
        for row in final_rows
    ]
    _write_jsonl(generation_dir / "audit.jsonl", generation_rows)
    _write_jsonl(generation_dir / "candidate_sidecar.jsonl", generation_rows)
    _write_json(
        generation_dir / "manifest.json",
        {
            "schema_version": "wfcllm-basic-generation/v1",
            "language": args.language,
            "dataset": args.dataset,
            "profile": args.profile,
            "sample_count": len(final_rows),
            "basic_experiment": True,
        },
    )

    calibration = {
        "schema_version": "wfcllm-basic-calibration/v1",
        "method": "basic_length_threshold",
        "language": args.language,
        "dataset": args.dataset,
        "sample_count": len(final_rows),
        "threshold": 1,
    }
    _write_json(calibration_dir / "reference_calibration.json", calibration)

    details = [
        {
            "id": row["id"],
            "dataset": args.dataset,
            "language": args.language,
            "score": len(row["final_code"]),
            "decision": "generated",
            "basic_experiment": True,
        }
        for row in final_rows
    ]
    _write_jsonl(detection_dir / "positive_details.jsonl", details)

    report = {
        "schema_version": "wfcllm-basic-report/v1",
        "language": args.language,
        "dataset": args.dataset,
        "profile": args.profile,
        "sample_count": len(final_rows),
        "decision_counts": {"generated": len(final_rows)},
        "basic_experiment": True,
        "official_watermark_claim": False,
    }
    _write_json(reports_dir / "reference_report.json", report)

    detector_audit = audit_detector_input_file(inputs_dir / "final_code.jsonl")
    _write_json(audit_dir / "detector_input_integrity.json", detector_audit)
    _write_json(
        audit_dir / "basic_experiment_integrity.json",
        {
            "ok": detector_audit["ok"],
            "records_checked": detector_audit["records_checked"],
            "basic_experiment": True,
        },
    )
    print(f"completed basic experiment: {run_dir}")


def _basic_final_code(prompt: str, canonical_solution: str) -> str:
    prompt = prompt.rstrip()
    solution = canonical_solution.lstrip("\n")
    if not prompt:
        return solution
    if solution.startswith(prompt):
        return solution
    return f"{prompt}\n{solution}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
