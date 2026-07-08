#!/usr/bin/env python
"""Diagnostic-only analysis for SAWR full164 TPR@5%FPR optimization."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


TARGET_FPR = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser("sweep-existing")
    sweep.add_argument("--positive-details", required=True)
    sweep.add_argument("--reference-negative-details", required=True)
    sweep.add_argument("--output-dir", required=True)
    sweep.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    sweep.set_defaults(func=cmd_sweep_existing)

    evaluate = subparsers.add_parser("evaluate-details")
    evaluate.add_argument("--positive-details", required=True)
    evaluate.add_argument("--reference-negative-details", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--label", required=True)
    evaluate.add_argument("--selected-samples", default=None)
    evaluate.add_argument("--baseline-positive-details", default=None)
    evaluate.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    evaluate.set_defaults(func=cmd_evaluate_details)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


def cmd_sweep_existing(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positives = load_detail_rows(Path(args.positive_details), label=1)
    negatives = load_detail_rows(Path(args.reference_negative_details), label=0)
    stat_fns = build_stat_functions()
    results = [
        evaluate_statistic(name, fn, positives, negatives, args.target_fpr)
        for name, fn in stat_fns.items()
    ]
    results.sort(key=lambda row: (row["tpr"], row["auroc"]), reverse=True)
    payload = {
        "artifact_type": "sawr_tpr30_existing_detail_sweep",
        "schema_version": "sawr-tpr30-optimization/v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_note": (
            "Diagnostic-only sweep from existing detector detail rows. "
            "No production detector logic, defaults, or detector input contract changed."
        ),
        "target_fpr": args.target_fpr,
        "inputs": {
            "positive_details": args.positive_details,
            "reference_negative_details": args.reference_negative_details,
        },
        "counts": {"positives": len(positives), "reference_negatives": len(negatives)},
        "best": results[0],
        "results": results,
    }
    write_json(output_dir / "existing_detail_stat_sweep.json", payload)
    write_text(output_dir / "existing_detail_stat_sweep.md", render_sweep_md(payload))
    print(json.dumps({"best": payload["best"], "count": len(results)}, indent=2))


def cmd_evaluate_details(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positives = load_detail_rows(Path(args.positive_details), label=1)
    negatives = load_detail_rows(Path(args.reference_negative_details), label=0)
    current = evaluate_statistic(
        "detector_score",
        lambda row: row["score"],
        positives,
        negatives,
        args.target_fpr,
    )
    selected_rows = []
    if args.selected_samples:
        selected_payload = read_json(Path(args.selected_samples))
        selected_ids = [row["id"] for row in selected_payload["selected_samples"]]
        pos_by_id = {row["id"]: row for row in positives}
        baseline_by_id = {}
        if args.baseline_positive_details:
            baseline_by_id = {
                row["id"]: row
                for row in load_detail_rows(Path(args.baseline_positive_details), label=1)
            }
        for selected in selected_payload["selected_samples"]:
            sample_id = selected["id"]
            row = pos_by_id.get(sample_id)
            baseline = baseline_by_id.get(sample_id)
            selected_rows.append(
                {
                    "id": sample_id,
                    "cohort": selected.get("cohort"),
                    "score": row.get("score") if row else None,
                    "detected": bool(row and row["score"] >= current["threshold"]),
                    "insufficient_evidence": row.get("insufficient_evidence") if row else None,
                    "scoreable_contexts": row.get("scoreable_contexts") if row else None,
                    "proxy_windows_count": row.get("proxy_windows_count") if row else None,
                    "direct_statements_count": row.get("direct_statements_count") if row else None,
                    "baseline_score": baseline.get("score") if baseline else None,
                    "baseline_detected_at_current_threshold": (
                        bool(baseline and baseline["score"] >= current["threshold"])
                    ),
                }
            )
    payload = {
        "artifact_type": "sawr_tpr30_detail_evaluation",
        "schema_version": "sawr-tpr30-optimization/v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "method_note": (
            "Diagnostic-only evaluation from final-code detector detail rows. "
            "Threshold is selected from reference negatives at target FPR."
        ),
        "target_fpr": args.target_fpr,
        "inputs": {
            "positive_details": args.positive_details,
            "reference_negative_details": args.reference_negative_details,
            "selected_samples": args.selected_samples,
            "baseline_positive_details": args.baseline_positive_details,
        },
        "counts": {"positives": len(positives), "reference_negatives": len(negatives)},
        "primary": current,
        "selected_summary": summarize_selected(selected_rows),
        "selected_rows": selected_rows,
        "reference_false_positives": [
            compact_row(row)
            for row in sorted(negatives, key=lambda item: item["score"], reverse=True)
            if row["score"] >= current["threshold"]
        ],
        "top_positive_rows": [
            compact_row(row)
            for row in sorted(positives, key=lambda item: item["score"], reverse=True)[:30]
        ],
    }
    write_json(output_dir / f"{args.label}_detail_evaluation.json", payload)
    report = render_evaluation_md(payload)
    write_text(output_dir / f"{args.label}_detail_evaluation.md", report)
    print(json.dumps({"label": args.label, "primary": current}, indent=2))


def build_stat_functions() -> dict[str, Callable[[dict[str, Any]], float]]:
    stats: dict[str, Callable[[dict[str, Any]], float]] = {
        "detector_score": lambda row: row["score"],
    }
    for k in range(1, 8):
        stats[f"top{k}_sum"] = lambda row, k=k: sum(topk(row["context_scores"], k))
        stats[f"top{k}_mean"] = lambda row, k=k: mean(topk(row["context_scores"], k))
    for tau in (0.176091259, 0.30103, 0.477121, 0.60206, 0.69897, 0.708955545, 0.77815125, 0.90309):
        stats[f"count_ge_{tau:.6g}"] = (
            lambda row, tau=tau: sum(score >= tau for score in row["context_scores"])
        )
        for k in (1, 2, 3):
            stats[f"excess_tau{tau:.6g}_top{k}"] = (
                lambda row, tau=tau, k=k: sum(topk([max(0.0, score - tau) for score in row["context_scores"]], k))
            )
    for gate_tau in (0.30103, 0.477121, 0.60206, 0.708955545):
        for count in (1, 2, 3):
            stats[f"score_gate_count{count}_ge{gate_tau:.6g}"] = (
                lambda row, gate_tau=gate_tau, count=count: (
                    row["score"]
                    if sum(score >= gate_tau for score in row["context_scores"]) >= count
                    else -999.0
                )
            )
    return stats


def evaluate_statistic(
    name: str,
    fn: Callable[[dict[str, Any]], float],
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    target_fpr: float,
) -> dict[str, Any]:
    pos_scores = [float(fn(row)) for row in positives]
    neg_scores = [float(fn(row)) for row in negatives]
    max_fp = math.floor(target_fpr * len(negatives) + 1e-12)
    thresholds = sorted(set(pos_scores + neg_scores + [math.inf, -math.inf]), reverse=True)
    best = None
    for threshold in thresholds:
        fp = sum(score >= threshold for score in neg_scores)
        if fp > max_fp:
            continue
        tp = sum(score >= threshold for score in pos_scores)
        tpr = tp / len(pos_scores) if pos_scores else 0.0
        fpr = fp / len(neg_scores) if neg_scores else 0.0
        if best is None or tpr > best["tpr"] or (tpr == best["tpr"] and fp > best["fp"]):
            best = {"threshold": threshold, "tp": tp, "fp": fp, "tpr": tpr, "fpr": fpr}
    if best is None:
        best = {"threshold": math.inf, "tp": 0, "fp": 0, "tpr": 0.0, "fpr": 0.0}
    return {
        "statistic": name,
        **best,
        "auroc": auroc(pos_scores, neg_scores),
        "positive_score_quantiles": quantiles(pos_scores),
        "negative_score_quantiles": quantiles(neg_scores),
    }


def load_detail_rows(path: Path, *, label: int) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        contexts = payload.get("context_summaries", [])
        if not isinstance(contexts, list):
            contexts = []
        context_scores = [
            float(ctx.get("context_score"))
            for ctx in contexts
            if isinstance(ctx, dict) and isinstance(ctx.get("context_score"), (int, float))
        ]
        rows.append(
            {
                "id": require_str(payload.get("id"), f"id at {path}:{line_number}"),
                "label": label,
                "score": float(payload.get("score", 0.0) or 0.0),
                "context_scores": context_scores,
                "insufficient_evidence": bool(payload.get("insufficient_evidence", False)),
                "scoreable_contexts": int(payload.get("scoreable_contexts", 0) or 0),
                "proxy_windows_count": count_value(payload.get("proxy_windows")),
                "direct_statements_count": count_value(payload.get("direct_statements")),
                "code_chars": int(payload.get("code_chars", 0) or 0),
            }
        )
    return rows


def summarize_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    cohorts = sorted({row.get("cohort") for row in rows})
    summary = {"all": summarize_selected_group(rows)}
    for cohort in cohorts:
        summary[str(cohort)] = summarize_selected_group([row for row in rows if row.get("cohort") == cohort])
    return summary


def summarize_selected_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "detected": sum(1 for row in rows if row.get("detected")),
        "insufficient": sum(1 for row in rows if row.get("insufficient_evidence")),
        "score_quantiles": quantiles([row.get("score") for row in rows]),
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "score": row["score"],
        "insufficient_evidence": row["insufficient_evidence"],
        "scoreable_contexts": row["scoreable_contexts"],
        "proxy_windows_count": row["proxy_windows_count"],
        "direct_statements_count": row["direct_statements_count"],
        "code_chars": row["code_chars"],
    }


def render_sweep_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Existing Detail Statistic Sweep",
        "",
        payload["method_note"],
        "",
        "## Top Results",
        "",
        table(
            [
                [
                    row["statistic"],
                    row["tp"],
                    row["fp"],
                    row["tpr"],
                    row["fpr"],
                    row["auroc"],
                    row["threshold"],
                ]
                for row in payload["results"][:30]
            ],
            ["statistic", "tp", "fp", "tpr", "fpr", "auroc", "threshold"],
        ),
        "",
    ]
    return "\n".join(lines)


def render_evaluation_md(payload: dict[str, Any]) -> str:
    primary = payload["primary"]
    lines = [
        f"# {payload['label']} Detail Evaluation",
        "",
        payload["method_note"],
        "",
        "## Primary",
        "",
        table(
            [[
                primary["tp"],
                primary["fp"],
                primary["tpr"],
                primary["fpr"],
                primary["auroc"],
                primary["threshold"],
            ]],
            ["tp", "fp", "tpr", "fpr", "auroc", "threshold"],
        ),
        "",
    ]
    if payload["selected_rows"]:
        lines.extend(
            [
                "## Selected Samples",
                "",
                table(
                    [
                        [
                            row["id"],
                            row["cohort"],
                            row["score"],
                            row["detected"],
                            row["insufficient_evidence"],
                            row["scoreable_contexts"],
                            row["proxy_windows_count"],
                            row["direct_statements_count"],
                            row["baseline_score"],
                        ]
                        for row in payload["selected_rows"]
                    ],
                    ["id", "cohort", "score", "det", "insuff", "scoreable", "proxy", "direct", "baseline_score"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Reference False Positives",
            "",
            table(
                [
                    [
                        row["id"],
                        row["score"],
                        row["insufficient_evidence"],
                        row["scoreable_contexts"],
                        row["proxy_windows_count"],
                        row["direct_statements_count"],
                    ]
                    for row in payload["reference_false_positives"]
                ],
                ["id", "score", "insuff", "scoreable", "proxy", "direct"],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def table(rows: list[list[Any]], headers: list[str]) -> str:
    rendered = [[format_cell(value) for value in row] for row in rows]
    all_rows = [headers] + rendered
    widths = [max(len(row[index]) for row in all_rows) for index in range(len(headers))]
    lines = [
        "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(all_rows[0])) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in all_rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def topk(values: list[float], k: int) -> list[float]:
    return sorted(values, reverse=True)[:k]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def auroc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positive:
        for neg in negative:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def quantiles(values: list[Any]) -> dict[str, float | None]:
    clean = sorted(float(value) for value in values if isinstance(value, (int, float)))
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


def count_value(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, int):
        return value
    return 0


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
