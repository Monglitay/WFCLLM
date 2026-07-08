#!/usr/bin/env python
"""Select SAWR final-code candidates using prompt-visible quality gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.sawr.selection import (  # noqa: E402
    CandidateSelectionFeatures,
    RANKING_MODES,
    evaluate_candidate_quality,
    select_best_candidate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select among SAWR candidate final-code JSONL files.",
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
            "Candidate ranking after gates. quality_first keeps the original "
            "syntax/public/evidence priority; public_then_detector ranks by "
            "strict code-only detector score after public gates; detector_first "
            "uses detector score before public doctest pass status."
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
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] selected SAWR candidates saved to {args.output_jsonl}", file=sys.stderr)
    return 0


def select_candidate_rows(
    candidate_paths: list[Path],
    detail_paths: list[Path],
    *,
    min_detector_score: float | None = None,
    max_score_drop_vs_baseline: float | None = None,
    min_proxy_windows: int = 0,
    min_scoreable_contexts: int = 0,
    require_not_insufficient: bool = False,
    require_proxy_ge_baseline: bool = False,
    require_public_doctest_passed: bool = False,
    reject_suspicious_tail: bool = False,
    ranking_mode: str = "quality_first",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidate_paths:
        raise ValueError("at least one --candidate-jsonl is required")
    if detail_paths and len(detail_paths) != len(candidate_paths):
        raise ValueError("--candidate-details count must match --candidate-jsonl count")

    candidate_rows = [_load_jsonl(path) for path in candidate_paths]
    detail_features = [_load_detail_features(path) for path in detail_paths]
    while len(detail_features) < len(candidate_rows):
        detail_features.append({})

    sample_ids = sorted({str(row.get("id") or "") for rows in candidate_rows for row in rows})
    selected_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        rows_for_id = [
            row
            for rows in candidate_rows
            for row in rows
            if str(row.get("id") or "") == sample_id
        ]
        if not rows_for_id:
            continue
        features: list[CandidateSelectionFeatures] = []
        row_by_candidate: dict[int, dict[str, Any]] = {}
        for candidate_index, rows in enumerate(candidate_rows):
            row = _first_row_for_id(rows, sample_id)
            if row is None:
                continue
            detail = detail_features[candidate_index].get(sample_id, {})
            baseline_detail = detail_features[0].get(sample_id, {}) if detail_features else {}
            row_by_candidate[candidate_index] = row
            features.append(
                evaluate_candidate_quality(
                    row,
                    detector_score=_float_feature(detail.get("score"), default=0.0),
                    candidate_index=candidate_index,
                    scoreable_contexts=_int_feature(detail.get("scoreable_contexts")),
                    proxy_windows=_int_feature(detail.get("proxy_windows")),
                    insufficient_evidence=bool(
                        detail.get("insufficient_evidence", False)
                    ),
                    baseline_detector_score=(
                        _float_feature(baseline_detail.get("score"), default=0.0)
                        if baseline_detail
                        else None
                    ),
                    baseline_proxy_windows=(
                        _int_feature(baseline_detail.get("proxy_windows"))
                        if baseline_detail
                        else None
                    ),
                )
            )
        eligible = _eligible_features(
            features,
            min_detector_score=min_detector_score,
            max_score_drop_vs_baseline=max_score_drop_vs_baseline,
            min_proxy_windows=min_proxy_windows,
            min_scoreable_contexts=min_scoreable_contexts,
            require_not_insufficient=require_not_insufficient,
            require_proxy_ge_baseline=require_proxy_ge_baseline,
            require_public_doctest_passed=require_public_doctest_passed,
            reject_suspicious_tail=reject_suspicious_tail,
        )
        selected = select_best_candidate(eligible, ranking_mode=ranking_mode)
        selected_row = row_by_candidate[selected.candidate_index]
        selected_rows.append(_sanitized_output_row(selected_row))
        per_sample.append(
            {
                "id": sample_id,
                "selected_candidate_index": selected.candidate_index,
                "selection_reason": selected.selection_reason,
                "candidates": [feature.__dict__ for feature in features],
            }
        )

    analysis = {
        "artifact_type": "sawr_candidate_selection_analysis",
        "candidate_paths": [str(path) for path in candidate_paths],
        "detail_paths": [str(path) for path in detail_paths],
        "sample_count": len(selected_rows),
        "selected_by_candidate_index": _count_selected(per_sample),
        "quality_counts": _quality_counts(per_sample),
        "policy": {
            "min_detector_score": min_detector_score,
            "max_score_drop_vs_baseline": max_score_drop_vs_baseline,
            "min_proxy_windows": min_proxy_windows,
            "min_scoreable_contexts": min_scoreable_contexts,
            "require_not_insufficient": require_not_insufficient,
            "require_proxy_ge_baseline": require_proxy_ge_baseline,
            "require_public_doctest_passed": require_public_doctest_passed,
            "reject_suspicious_tail": reject_suspicious_tail,
            "ranking_mode": ranking_mode,
        },
        "per_sample": per_sample,
    }
    return selected_rows, analysis


def _sanitized_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("task_id") or ""),
        "dataset": str(row.get("dataset") or "humaneval"),
        "prompt": str(row.get("prompt") or ""),
        "final_code": str(row.get("final_code") or row.get("generated_code") or ""),
    }


def _eligible_features(
    features: list[CandidateSelectionFeatures],
    *,
    min_detector_score: float | None,
    max_score_drop_vs_baseline: float | None,
    min_proxy_windows: int,
    min_scoreable_contexts: int,
    require_not_insufficient: bool,
    require_proxy_ge_baseline: bool,
    require_public_doctest_passed: bool,
    reject_suspicious_tail: bool,
) -> list[CandidateSelectionFeatures]:
    if not features:
        return []
    baseline = min(features, key=lambda feature: feature.candidate_index)
    eligible = [baseline]
    for feature in features:
        if feature.candidate_index == baseline.candidate_index:
            continue
        if min_detector_score is not None and feature.detector_score < min_detector_score:
            continue
        if (
            max_score_drop_vs_baseline is not None
            and feature.score_delta_vs_baseline is not None
            and feature.score_delta_vs_baseline < -max_score_drop_vs_baseline
        ):
            continue
        if feature.proxy_windows < min_proxy_windows:
            continue
        if feature.scoreable_contexts < min_scoreable_contexts:
            continue
        if require_not_insufficient and feature.insufficient_evidence:
            continue
        if (
            require_proxy_ge_baseline
            and feature.proxy_delta_vs_baseline is not None
            and feature.proxy_delta_vs_baseline < 0
        ):
            continue
        if require_public_doctest_passed and not feature.public_doctest_passed:
            continue
        if feature.public_doctest_timeout or feature.public_doctest_parse_error:
            continue
        if reject_suspicious_tail and feature.suspicious_tail:
            continue
        eligible.append(feature)
    return eligible


def _load_detail_features(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id") or ""): row
        for row in _load_jsonl(path)
    }


def _float_feature(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_feature(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_row_for_id(rows: list[dict[str, Any]], sample_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("id") or "") == sample_id:
            return row
    return None


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
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SAWR Candidate Selection",
        "",
        f"Selected rows: {analysis['sample_count']}",
        "",
        "## Selected by Candidate Index",
        "",
    ]
    for key, count in sorted(analysis["selected_by_candidate_index"].items()):
        lines.append(f"- candidate {key}: {count}")
    lines.extend(["", "## Quality Counts", ""])
    for key, count in sorted(analysis["quality_counts"].items()):
        lines.append(f"- {key}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_selected(per_sample: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_sample:
        key = str(row["selected_candidate_index"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _quality_counts(per_sample: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_sample:
        selected_index = row["selected_candidate_index"]
        selected = next(
            candidate
            for candidate in row["candidates"]
            if candidate["candidate_index"] == selected_index
        )
        for key in (
            "syntax_valid",
            "target_function_present",
            "signature_compatible",
            "prompt_doctest_passed",
        ):
            if selected[key]:
                counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
