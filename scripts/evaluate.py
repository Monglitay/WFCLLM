#!/usr/bin/env python
"""Posthoc Pass@1 reporting for a completed Gate-only reproduction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.evaluation.code_execution import compute_pass_at_1, load_jsonl_records  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute posthoc Pass@1")
    parser.add_argument("inputs", nargs="+", help="posthoc correctness JSONL")
    parser.add_argument("--output", help="optional pass_report_posthoc.json path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows: list[dict] = []
    for path in args.inputs:
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


if __name__ == "__main__":
    raise SystemExit(main())
