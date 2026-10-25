#!/usr/bin/env python
"""Build fixed secondary length-filtered detection diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfcllm.evaluation.length_filtered_detection import (
    build_length_filtered_detection_report,
    load_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build fixed WFCLLM length-filtered detection diagnostics."
    )
    parser.add_argument("--positive-final-code", type=Path, required=True)
    parser.add_argument("--positive-details", type=Path, required=True)
    parser.add_argument("--negative-final-code", type=Path)
    parser.add_argument("--negative-details", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    negative_final_code = (
        load_jsonl(args.negative_final_code) if args.negative_final_code else None
    )
    negative_details = load_jsonl(args.negative_details) if args.negative_details else None
    report = build_length_filtered_detection_report(
        positive_final_code_rows=load_jsonl(args.positive_final_code),
        positive_detail_rows=load_jsonl(args.positive_details),
        negative_final_code_rows=negative_final_code,
        negative_detail_rows=negative_details,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
