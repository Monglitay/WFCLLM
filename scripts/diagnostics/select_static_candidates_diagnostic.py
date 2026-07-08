#!/usr/bin/env python
"""Select candidates with static diagnostic quality proxies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.diagnostics.static_selector import select_static_candidates  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select between baseline and candidate final-code rows with static "
            "diagnostic quality proxies."
        ),
    )
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--baseline-details", required=True)
    parser.add_argument("--candidate-details", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument(
        "--policy",
        choices=["candidate_if_syntax_signature_sufficient"],
        default="candidate_if_syntax_signature_sufficient",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        rows, analysis = select_static_candidates(
            baseline_rows=_load_jsonl(Path(args.baseline_jsonl)),
            candidate_rows=_load_jsonl(Path(args.candidate_jsonl)),
            baseline_details=_load_jsonl(Path(args.baseline_details)),
            candidate_details=_load_jsonl(Path(args.candidate_details)),
            policy=args.policy,
        )
        _write_jsonl(Path(args.output_jsonl), rows)
        _write_json(Path(args.analysis_json), analysis)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"[done] static diagnostic selected rows saved to {args.output_jsonl}", file=sys.stderr)
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
