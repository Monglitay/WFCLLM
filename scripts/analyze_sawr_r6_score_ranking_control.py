#!/usr/bin/env python
"""Analyze R6 score-ranking positives against negative high-tail control."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    positive_payload = read_json(Path(args.positive_gate_analysis))
    selected_payload = read_json(Path(args.selected_samples))
    reference_rows = read_jsonl(Path(args.reference_negative_details))
    selected_ids = [
        require_str(row.get("id"), "id")
        for row in selected_payload["selected_samples"]
    ]

    positive = analyze_positive_score_ranking(positive_payload, args.threshold)
    negative = analyze_negative_attempts(root, selected_ids, args.threshold)
    reference = analyze_reference_negative(reference_rows, selected_ids, args.threshold)
    projection = build_tpr_projection(
        official_positive_count=args.official_positive_count,
        official_tp=args.official_tp,
        selected_fn_count=positive["all"]["selected_fn_count"],
        selected_fn_recovered=positive["all"]["fn_recovered_by_r6_best"],
    )

    decision = build_decision(
        positive=positive,
        negative=negative,
        reference=reference,
        projection=projection,
    )
    payload = {
        "artifact_type": "sawr_r6_score_ranking_control",
        "schema_version": "sawr-r6-score-ranking-control/v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": args.threshold,
        "inputs": {
            "root": str(root),
            "positive_gate_analysis": args.positive_gate_analysis,
            "selected_samples": args.selected_samples,
            "reference_negative_details": args.reference_negative_details,
        },
        "positive_score_ranking": positive,
        "negative_score_ranking_control": negative,
        "reference_negative_control": reference,
        "tpr_projection": projection,
        "decision": decision,
    }
    write_json(analysis_dir / "r6_score_ranking_control.json", payload)
    report = render_report(payload)
    write_text(analysis_dir / "R6_SCORE_RANKING_CONTROL.md", report)
    write_text(root / "SAWR_R6_SCORE_RANKING_CONTROL_REPORT.md", report)
    write_text(root / "manifest.txt", render_manifest(payload))
    write_text(root / "RUN_PLAN.md", render_run_plan(payload))
    print(
        json.dumps(
            {
                "positive_best_detected": positive["all"]["r6_best_detected"],
                "negative_best_fp": negative["all"]["best_score_fp_count"],
                "negative_best_count": negative["all"]["count"],
                "projected_tpr_if_transferable": projection["projected_tpr_if_fn_recovery_transfers"],
                "decision": decision["recommendation"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--positive-gate-analysis", required=True)
    parser.add_argument("--selected-samples", required=True)
    parser.add_argument("--reference-negative-details", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--official-positive-count", type=int, default=164)
    parser.add_argument("--official-tp", type=int, default=39)
    return parser


def analyze_positive_score_ranking(
    payload: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    rows = payload["per_sample"]
    by_cohort = {"all": rows}
    for cohort in sorted({row["cohort"] for row in rows}):
        by_cohort[cohort] = [row for row in rows if row["cohort"] == cohort]
    out = {
        cohort: summarize_positive_rows(cohort_rows, threshold)
        for cohort, cohort_rows in by_cohort.items()
    }
    out["per_sample"] = [
        {
            "id": row["id"],
            "cohort": row["cohort"],
            "baseline_score": row.get("baseline_score"),
            "baseline_detected": bool(row.get("baseline_detected")),
            "r6_best_score": row.get("best_score_oracle", {}).get("score"),
            "r6_best_seed": row.get("best_score_oracle", {}).get("seed"),
            "r6_best_detected": bool(row.get("best_score_oracle", {}).get("detected")),
            "r6_loose_score": row.get("r6_loose_score", {}).get("score"),
            "r6_loose_detected": bool(row.get("r6_loose_score", {}).get("detected")),
        }
        for row in rows
    ]
    return out


def summarize_positive_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected_fn = [row for row in rows if not row.get("baseline_detected")]
    fn_recovered = [
        row for row in selected_fn
        if bool(row.get("best_score_oracle", {}).get("detected"))
    ]
    best_scores = [
        maybe_float(row.get("best_score_oracle", {}).get("score"))
        for row in rows
    ]
    baseline_scores = [maybe_float(row.get("baseline_score")) for row in rows]
    return {
        "count": len(rows),
        "baseline_detected": sum(1 for row in rows if row.get("baseline_detected")),
        "r6_best_detected": sum(
            1 for row in rows if row.get("best_score_oracle", {}).get("detected")
        ),
        "selected_fn_count": len(selected_fn),
        "fn_recovered_by_r6_best": len(fn_recovered),
        "fn_recovery_rate": safe_div(len(fn_recovered), len(selected_fn)),
        "baseline_score_quantiles": quantiles(baseline_scores),
        "r6_best_score_quantiles": quantiles(best_scores),
        "above_threshold_rate": safe_div(
            sum(1 for value in best_scores if value is not None and value >= threshold),
            len(rows),
        ),
    }


def analyze_negative_attempts(
    root: Path,
    selected_ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    attempts = load_negative_attempts(root)
    per_sample: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        candidates = [attempt[sample_id] for attempt in attempts if sample_id in attempt]
        candidates.sort(key=lambda row: row["seed"])
        best = max(candidates, key=lambda row: row.get("score") or -math.inf) if candidates else None
        per_sample.append(
            {
                "id": sample_id,
                "candidate_count": len(candidates),
                "candidate_scores": [row.get("score") for row in candidates],
                "best_seed": best.get("seed") if best else None,
                "best_score": best.get("score") if best else None,
                "best_fp": bool(best and (best.get("score") or 0.0) >= threshold),
                "best_scoreable_contexts": best.get("scoreable_contexts") if best else None,
                "best_proxy_windows_count": best.get("proxy_windows_count") if best else None,
                "best_direct_statements_count": best.get("direct_statements_count") if best else None,
                "best_insufficient_evidence": best.get("insufficient_evidence") if best else None,
                "best_code_chars": best.get("code_chars") if best else None,
            }
        )
    all_summary = summarize_negative_rows(per_sample, threshold)
    return {
        "attempt_count": len(attempts),
        "all": all_summary,
        "per_sample": per_sample,
        "score_threshold": threshold,
    }


def load_negative_attempts(root: Path) -> list[dict[str, dict[str, Any]]]:
    attempts: list[dict[str, dict[str, Any]]] = []
    for attempt_dir in sorted(root.glob("negative_attempt_*")):
        seed = int(attempt_dir.name.rsplit("_", 1)[-1])
        final_files = sorted((attempt_dir / "generation").glob("*negative_final*.jsonl"))
        detail_files = sorted((attempt_dir / "detection/gamma025").glob("*details*.jsonl"))
        if not final_files or not detail_files:
            continue
        final_by_id = {row["id"]: row for row in read_jsonl(final_files[-1])}
        detail_by_id = {row["id"]: row for row in read_jsonl(detail_files[-1])}
        rows: dict[str, dict[str, Any]] = {}
        for sample_id, detail in detail_by_id.items():
            final = final_by_id.get(sample_id, {})
            rows[sample_id] = {
                "seed": seed,
                "score": maybe_float(detail.get("score")),
                "insufficient_evidence": boolish(detail.get("insufficient_evidence")),
                "scoreable_contexts": int_or_none(detail.get("scoreable_contexts")),
                "proxy_windows_count": count_value(detail.get("proxy_windows")),
                "direct_statements_count": count_value(detail.get("direct_statements")),
                "code_chars": len(str(final.get("final_code", ""))),
            }
        attempts.append(rows)
    return attempts


def summarize_negative_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scores = [maybe_float(row.get("best_score")) for row in rows]
    fp_count = sum(1 for score in scores if score is not None and score >= threshold)
    return {
        "count": len(rows),
        "best_score_fp_count": fp_count,
        "best_score_fpr_proxy": safe_div(fp_count, len(rows)),
        "candidate_count_quantiles": quantiles([row.get("candidate_count") for row in rows]),
        "best_score_quantiles": quantiles(scores),
        "best_scoreable_contexts_quantiles": quantiles(
            [row.get("best_scoreable_contexts") for row in rows]
        ),
        "best_proxy_windows_count_quantiles": quantiles(
            [row.get("best_proxy_windows_count") for row in rows]
        ),
        "best_code_chars_quantiles": quantiles([row.get("best_code_chars") for row in rows]),
    }


def analyze_reference_negative(
    rows: list[dict[str, Any]],
    selected_ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("id") in set(selected_ids)]
    false_positives = [row for row in rows if (maybe_float(row.get("score")) or 0.0) >= threshold]
    selected_false_positives = [
        row for row in selected if (maybe_float(row.get("score")) or 0.0) >= threshold
    ]
    return {
        "full_reference_count": len(rows),
        "full_reference_fp_count": len(false_positives),
        "full_reference_fpr": safe_div(len(false_positives), len(rows)),
        "selected_reference_count": len(selected),
        "selected_reference_fp_count": len(selected_false_positives),
        "selected_reference_fpr_proxy": safe_div(len(selected_false_positives), len(selected)),
        "score_quantiles": quantiles([row.get("score") for row in rows]),
        "selected_score_quantiles": quantiles([row.get("score") for row in selected]),
        "false_positives": [
            compact_detail_row(row, threshold)
            for row in sorted(false_positives, key=lambda item: maybe_float(item.get("score")) or -math.inf, reverse=True)
        ],
        "selected_rows": [
            compact_detail_row(row, threshold)
            for row in sorted(selected, key=lambda item: str(item.get("id")))
        ],
    }


def build_tpr_projection(
    *,
    official_positive_count: int,
    official_tp: int,
    selected_fn_count: int,
    selected_fn_recovered: int,
) -> dict[str, Any]:
    official_fn = official_positive_count - official_tp
    recovery_rate = safe_div(selected_fn_recovered, selected_fn_count)
    projected_tp = official_tp + (official_fn * recovery_rate if recovery_rate is not None else 0.0)
    return {
        "official_positive_count": official_positive_count,
        "official_tp": official_tp,
        "official_tpr": safe_div(official_tp, official_positive_count),
        "official_fn": official_fn,
        "selected_fn_count": selected_fn_count,
        "selected_fn_recovered": selected_fn_recovered,
        "selected_fn_recovery_rate": recovery_rate,
        "projected_tp_if_fn_recovery_transfers": projected_tp,
        "projected_tpr_if_fn_recovery_transfers": safe_div(projected_tp, official_positive_count),
        "caveat": (
            "Projection is diagnostic only because the 12-sample set is enriched for "
            "known false negatives and strong true positives."
        ),
    }


def build_decision(
    *,
    positive: dict[str, Any],
    negative: dict[str, Any],
    reference: dict[str, Any],
    projection: dict[str, Any],
) -> dict[str, Any]:
    positive_recovery = positive["all"]["fn_recovery_rate"] or 0.0
    negative_fpr = negative["all"]["best_score_fpr_proxy"]
    negative_fp_count = negative["all"]["best_score_fp_count"]
    reference_fpr = reference["full_reference_fpr"]
    reasons: list[str] = []
    if positive_recovery >= 0.2:
        reasons.append(
            f"R6 best-of-seed recovered {positive['all']['fn_recovered_by_r6_best']}/"
            f"{positive['all']['selected_fn_count']} selected FNs."
        )
    else:
        reasons.append("R6 best-of-seed recovered fewer than 20% of selected FNs.")
    if negative_fp_count == 0:
        reasons.append("No selected unwatermarked negative best-of-seed candidate crossed the main threshold.")
    else:
        reasons.append(
            f"Selected unwatermarked negative best-of-seed had {negative_fp_count}/"
            f"{negative['all']['count']} threshold crossings."
        )
    if reference_fpr is not None:
        reasons.append(f"Full canonical reference baseline FPR remains {reference_fpr:.4f}.")

    if positive_recovery >= 0.2 and negative_fp_count == 0:
        recommendation = "go_30_sample_r6_score_ranking_control"
        next_step = (
            "Scale to 30 samples with matched positive and unwatermarked negative "
            "multi-seed pools before any full164 run."
        )
    elif positive_recovery >= 0.2 and negative_fp_count <= 1:
        recommendation = "cautious_scale_with_larger_negative_control"
        next_step = (
            "Run a larger negative high-tail control first; one FP in 12 is already "
            "above a strict 5% target as a point estimate."
        )
    else:
        recommendation = "no_go_r6_as_current_gate"
        next_step = (
            "Do not scale R6 score-ranking yet; prefer R7 stronger embedding/gamma "
            "or context-null calibrated statistic work."
        )
    return {
        "recommendation": recommendation,
        "next_step": next_step,
        "reasons": reasons,
        "negative_best_score_fpr_proxy": negative_fpr,
        "projected_tpr_if_transferable": projection["projected_tpr_if_fn_recovery_transfers"],
    }


def compact_detail_row(row: dict[str, Any], threshold: float) -> dict[str, Any]:
    score = maybe_float(row.get("score"))
    return {
        "id": row.get("id"),
        "score": score,
        "fp": bool(score is not None and score >= threshold),
        "scoreable_contexts": int_or_none(row.get("scoreable_contexts")),
        "proxy_windows_count": count_value(row.get("proxy_windows")),
        "direct_statements_count": count_value(row.get("direct_statements")),
        "insufficient_evidence": boolish(row.get("insufficient_evidence")),
        "context_pattern": summarize_context_pattern(row),
    }


def summarize_context_pattern(row: dict[str, Any]) -> str:
    contexts = row.get("context_summaries")
    if not isinstance(contexts, list) or not contexts:
        return ""
    scored = []
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        score = maybe_float(ctx.get("context_score"))
        if score is None:
            continue
        scored.append((score, ctx))
    if not scored:
        return ""
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[0][1]
    parts = []
    for key in ("node_type", "parent_node_type", "evidence_type", "window_type"):
        value = top.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def render_report(payload: dict[str, Any]) -> str:
    pos = payload["positive_score_ranking"]
    neg = payload["negative_score_ranking_control"]
    ref = payload["reference_negative_control"]
    proj = payload["tpr_projection"]
    decision = payload["decision"]
    lines = [
        "# SAWR R6 Score-Ranking Control Report",
        "",
        "## Executive Summary",
        "",
        (
            f"- Positive selected set: baseline detected {pos['all']['baseline_detected']}/"
            f"{pos['all']['count']}; R6 best-of-3 detected {pos['all']['r6_best_detected']}/"
            f"{pos['all']['count']}."
        ),
        (
            f"- Selected FN recovery: {pos['all']['fn_recovered_by_r6_best']}/"
            f"{pos['all']['selected_fn_count']} "
            f"({format_float(pos['all']['fn_recovery_rate'])})."
        ),
        (
            f"- Unwatermarked negative best-of-3 threshold crossings: "
            f"{neg['all']['best_score_fp_count']}/{neg['all']['count']} "
            f"({format_float(neg['all']['best_score_fpr_proxy'])})."
        ),
        (
            f"- Canonical reference baseline FPR: {ref['full_reference_fp_count']}/"
            f"{ref['full_reference_count']} ({format_float(ref['full_reference_fpr'])})."
        ),
        f"- Recommendation: `{decision['recommendation']}`. {decision['next_step']}",
        "",
        "## TPR Projection",
        "",
        (
            f"Official TPR stays {proj['official_tp']}/{proj['official_positive_count']} = "
            f"{format_float(proj['official_tpr'])}. If the selected-FN recovery transferred "
            f"to all {proj['official_fn']} official FNs, the diagnostic projection would be "
            f"{format_float(proj['projected_tpr_if_fn_recovery_transfers'])}. "
            "This is not a reported metric; it is a small-sample planning estimate."
        ),
        "",
        "## Positive Score-Ranking",
        "",
        table(
            [
                [
                    cohort,
                    summary["count"],
                    summary["baseline_detected"],
                    summary["r6_best_detected"],
                    summary["selected_fn_count"],
                    summary["fn_recovered_by_r6_best"],
                    format_float(summary["fn_recovery_rate"]),
                    format_float(summary["r6_best_score_quantiles"]["p50"]),
                    format_float(summary["r6_best_score_quantiles"]["max"]),
                ]
                for cohort, summary in pos.items()
                if cohort != "per_sample"
            ],
            [
                "cohort",
                "n",
                "baseline_det",
                "r6_best_det",
                "selected_fn",
                "fn_recovered",
                "fn_recovery",
                "best_p50",
                "best_max",
            ],
        ),
        "",
        "## Negative High-Tail Control",
        "",
        table(
            [
                [
                    row["id"],
                    row["candidate_count"],
                    row["best_seed"],
                    format_float(row["best_score"]),
                    row["best_fp"],
                    row["best_scoreable_contexts"],
                    row["best_proxy_windows_count"],
                    row["best_direct_statements_count"],
                    row["best_code_chars"],
                ]
                for row in neg["per_sample"]
            ],
            [
                "id",
                "candidates",
                "best_seed",
                "best_score",
                "fp",
                "scoreable",
                "proxy",
                "direct",
                "code_chars",
            ],
        ),
        "",
        "## Reference False Positives",
        "",
        (
            f"Full canonical reference false positives at threshold "
            f"{payload['threshold']}: {ref['full_reference_fp_count']}/"
            f"{ref['full_reference_count']}."
        ),
        "",
        table(
            [
                [
                    row["id"],
                    format_float(row["score"]),
                    row["scoreable_contexts"],
                    row["proxy_windows_count"],
                    row["direct_statements_count"],
                    row["insufficient_evidence"],
                    row["context_pattern"],
                ]
                for row in ref["false_positives"]
            ],
            [
                "id",
                "score",
                "scoreable",
                "proxy",
                "direct",
                "insufficient",
                "top_context_pattern",
            ],
        ),
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.extend(
        [
            (
                "- R6 score-ranking is only useful if the same best-of-N selection does "
                "not raise unwatermarked negative high-tail scores enough to break the "
                "5% FPR target."
            ),
            (
                "- This run is still a selected 12-sample diagnostic. It is suitable for "
                "choosing the next probe, not for replacing the official full164 metric."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_manifest(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "SAWR R6 score-ranking control manifest",
            f"created_at: {payload['created_at']}",
            f"threshold: {payload['threshold']}",
            f"positive_gate_analysis: {payload['inputs']['positive_gate_analysis']}",
            f"selected_samples: {payload['inputs']['selected_samples']}",
            f"reference_negative_details: {payload['inputs']['reference_negative_details']}",
            "missing_artifacts: none detected by analysis script",
            "production_code_modified: no",
            "detector_decision_logic_modified: no",
            "",
        ]
    )


def render_run_plan(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R6 Score-Ranking Control Run Plan",
            "",
            "1. Reuse existing 3-seed positive SAWR final-code candidates from the R5/R6 gate probe.",
            "2. Generate matched 3-seed unwatermarked negative final-code candidates for the same 12 IDs.",
            "3. Score every negative candidate with the unchanged gamma=0.25 final-code-only detector.",
            "4. Select the highest score per sample for positives and negatives.",
            "5. Compare positive FN recovery against unwatermarked negative high-tail crossings.",
            "",
            "No production detector logic, default threshold, default statistic, or detector input contract is changed.",
            "",
        ]
    )


def table(rows: list[list[Any]], headers: list[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return format_float(value)
        if value is None:
            return ""
        return str(value)

    all_rows = [headers] + [[fmt(value) for value in row] for row in rows]
    widths = [max(len(row[index]) for row in all_rows) for index in range(len(headers))]
    lines = [
        "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(all_rows[0])) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in all_rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def quantiles(values: list[Any]) -> dict[str, float | None]:
    clean = sorted(value for value in (maybe_float(item) for item in values) if value is not None)
    if not clean:
        return {key: None for key in ("min", "p25", "p50", "p75", "max")}
    return {
        "min": clean[0],
        "p25": percentile(clean, 0.25),
        "p50": percentile(clean, 0.50),
        "p75": percentile(clean, 0.75),
        "max": clean[-1],
    }


def percentile(sorted_values: list[float], frac: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = frac * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - pos) + sorted_values[high] * (pos - low)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def count_value(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return int_or_none(value)


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_float(value: Any) -> str:
    number = maybe_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
