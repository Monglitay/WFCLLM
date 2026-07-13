#!/usr/bin/env python
"""Assemble disjoint pilot and remaining final-code rows into a closed full arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "wfcllm-full-assembly/v1"
FINAL_CODE_FIELDS = frozenset({"id", "dataset", "prompt", "final_code"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly assemble a complete HumanEval final-code-only arm.",
    )
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--remaining", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    return parser


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_rows(path: str | Path, *, label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{label} row {line_number} must be an object")
                if set(value) != FINAL_CODE_FIELDS:
                    raise ValueError(
                        f"{label} row {line_number} must contain exactly the "
                        "final-code-only fields id, dataset, prompt, final_code"
                    )
                if any(not isinstance(value[field], str) for field in FINAL_CODE_FIELDS):
                    raise ValueError(
                        f"{label} row {line_number} final-code-only fields must be strings"
                    )
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL in {label} at line {exc.lineno}") from exc
    return rows


def _numeric_id(sample_id: str) -> int:
    prefix = "HumanEval/"
    if not sample_id.startswith(prefix):
        raise ValueError(f"unexpected HumanEval id: {sample_id}")
    suffix = sample_id[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError(f"unexpected HumanEval id: {sample_id}")
    return int(suffix)


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_count <= 0:
        raise ValueError("expected_count must be positive")
    pilot_rows = _load_rows(args.pilot, label="pilot")
    remaining_rows = _load_rows(args.remaining, label="remaining")
    by_id: dict[str, dict[str, str]] = {}
    for row in [*pilot_rows, *remaining_rows]:
        sample_id = row["id"]
        if sample_id in by_id:
            raise ValueError(f"duplicate id across assembly inputs: {sample_id}")
        by_id[sample_id] = row

    expected_ids = {f"HumanEval/{index}" for index in range(args.expected_count)}
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            "full arm is not closed; "
            f"missing={sorted(expected_ids - actual_ids, key=_numeric_id)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )

    ordered_rows = [by_id[f"HumanEval/{index}"] for index in range(args.expected_count)]
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
            for row in ordered_rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(ordered_rows),
        "pilot_count": len(pilot_rows),
        "remaining_count": len(remaining_rows),
        "pilot_sha256": _sha256(args.pilot),
        "remaining_sha256": _sha256(args.remaining),
        "output_sha256": _sha256(output_path),
        "ordered_ids": [row["id"] for row in ordered_rows],
        "final_code_only_fields": sorted(FINAL_CODE_FIELDS),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = assemble(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
