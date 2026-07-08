#!/usr/bin/env python
"""Merge and sanitize SAWR final-code JSONL candidate files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge SAWR final-code JSONL files by id without executing code.",
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--analysis-json", required=True)
    return parser


def merge_final_code_files(
    input_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    source_by_id: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    duplicate_count = 0

    for input_index, path in enumerate(input_paths):
        rows = _load_jsonl(path)
        source_counts[str(path)] = len(rows)
        for row in rows:
            sample_id = str(row.get("id") or row.get("task_id") or "")
            if not sample_id:
                raise ValueError(f"row is missing id: {path}")
            if sample_id in rows_by_id:
                duplicate_count += 1
            rows_by_id[sample_id] = _sanitized_row(row)
            source_by_id[sample_id] = f"{input_index}:{path}"

    merged = [rows_by_id[sample_id] for sample_id in sorted(rows_by_id)]
    analysis = {
        "artifact_type": "sawr_final_code_merge_analysis",
        "input_paths": [str(path) for path in input_paths],
        "source_counts": source_counts,
        "merged_count": len(merged),
        "duplicate_count": duplicate_count,
        "source_by_id": source_by_id,
    }
    return merged, analysis


def _sanitized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("task_id") or ""),
        "dataset": str(row.get("dataset") or "humaneval"),
        "prompt": str(row.get("prompt") or ""),
        "final_code": str(row.get("final_code") or row.get("generated_code") or ""),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        rows, analysis = merge_final_code_files([Path(path) for path in args.input])
        _write_jsonl(Path(args.output_jsonl), rows)
        _write_json(Path(args.analysis_json), analysis)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] merged SAWR final-code rows saved to {args.output_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
