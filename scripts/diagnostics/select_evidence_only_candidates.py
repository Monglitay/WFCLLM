#!/usr/bin/env python
"""Run diagnostic candidate selection and mark analysis as non-official."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.diagnostics.evidence_selector import (  # noqa: E402
    select_candidate_rows,
    selection_analysis_markdown,
)
from wfcllm.diagnostics.static_selector import RANKING_MODES  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select among diagnostic candidate final-code JSONL files.",
    )
    parser.add_argument("--candidate-jsonl", action="append", required=True)
    parser.add_argument("--candidate-details", action="append", default=[])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--analysis-md", required=True)
    parser.add_argument("--min-detector-score", type=float, default=None)
    parser.add_argument("--max-score-drop-vs-baseline", type=float, default=None)
    parser.add_argument("--min-proxy-windows", type=int, default=0)
    parser.add_argument("--min-scoreable-contexts", type=int, default=0)
    parser.add_argument("--require-not-insufficient", action="store_true")
    parser.add_argument("--require-proxy-ge-baseline", action="store_true")
    parser.add_argument("--require-public-doctest-passed", action="store_true")
    parser.add_argument("--reject-suspicious-tail", action="store_true")
    parser.add_argument(
        "--ranking-mode",
        default="quality_first",
        choices=sorted(RANKING_MODES),
        help=(
            "Candidate ranking after gates. This diagnostic path is not an "
            "official method and analysis output is marked accordingly."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        selected, analysis = select_candidate_rows(
            [Path(path) for path in args.candidate_jsonl],
            [Path(path) for path in args.candidate_details],
            min_detector_score=args.min_detector_score,
            max_score_drop_vs_baseline=args.max_score_drop_vs_baseline,
            min_proxy_windows=args.min_proxy_windows,
            min_scoreable_contexts=args.min_scoreable_contexts,
            require_not_insufficient=args.require_not_insufficient,
            require_proxy_ge_baseline=args.require_proxy_ge_baseline,
            require_public_doctest_passed=args.require_public_doctest_passed,
            reject_suspicious_tail=args.reject_suspicious_tail,
            ranking_mode=args.ranking_mode,
        )
        _write_jsonl(Path(args.output_jsonl), selected)
        _write_json(Path(args.analysis_json), analysis)
        _write_markdown(Path(args.analysis_md), analysis)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"[done] diagnostic selected rows saved to {args.output_jsonl}", file=sys.stderr)
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(selection_analysis_markdown(analysis), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
