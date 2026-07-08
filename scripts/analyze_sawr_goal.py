#!/usr/bin/env python
"""Build SAWR AUROC/pass bottleneck reports for the full164 goal."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--bestgen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    baseline_dir = Path(args.baseline_dir)
    bestgen_dir = Path(args.bestgen_dir)
    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    positive_details = _load_jsonl(
        baseline_dir / "detection/proxy_penalized_alpha04_sanitized/positive_details.jsonl"
    )
    reference_details = _load_jsonl(
        baseline_dir / "detection/proxy_penalized_alpha04_sanitized/reference_negative_details.jsonl"
    )
    model_negative_details = _load_jsonl(
        baseline_dir / "detection/proxy_penalized_alpha04_sanitized/model_negative_details.jsonl"
    )
    positive_inputs = _load_jsonl(baseline_dir / "inputs/positive_final_code_sanitized.jsonl")
    reference_inputs = _load_jsonl(baseline_dir / "inputs/reference_negative_sanitized.jsonl")
    execution_rows = _load_jsonl(bestgen_dir / "analysis/execution_results.jsonl")
    old_positive_details = _load_jsonl(bestgen_dir / "detection/gamma025/all_positive_details.jsonl")
    old_negative_details = _load_jsonl(bestgen_dir / "detection/gamma025/all_negative_details.jsonl")
    audit_summary = _load_json(bestgen_dir / "analysis/audit_hit_summary.json")
    generation_summary = _load_json(bestgen_dir / "analysis/generation_summary.json")

    auroc_report = _build_auroc_report(
        positive_details,
        reference_details,
        model_negative_details,
    )
    pass_report = _build_pass_report(
        positive_inputs,
        positive_details,
        execution_rows,
        audit_summary,
        generation_summary,
    )
    delta_report = _build_delta_report(
        positive_details,
        reference_details,
        old_positive_details,
        old_negative_details,
    )
    integrity_report = _build_input_integrity_report(positive_inputs, reference_inputs)

    _write_json(analysis_dir / "auroc_bottleneck.json", auroc_report)
    _write_md(analysis_dir / "auroc_bottleneck.md", _auroc_markdown(auroc_report))
    _write_json(analysis_dir / "pass_failure_taxonomy.json", pass_report)
    _write_md(analysis_dir / "pass_failure_taxonomy.md", _pass_markdown(pass_report))
    _write_json(analysis_dir / "per_sample_baseline_vs_final_delta.json", delta_report)
    _write_jsonl(
        analysis_dir / "per_sample_baseline_vs_final_delta.jsonl",
        delta_report["rows"],
    )
    _write_md(
        analysis_dir / "per_sample_baseline_vs_final_delta.md",
        _delta_markdown(delta_report),
    )
    _write_json(analysis_dir / "detector_input_integrity_goal.json", integrity_report)
    _write_md(
        analysis_dir / "detector_input_integrity_goal.md",
        _integrity_markdown(integrity_report),
    )
    print(json.dumps({"analysis_dir": str(analysis_dir)}, ensure_ascii=False, indent=2))
    return 0


def _build_auroc_report(
    positive_details: list[dict[str, Any]],
    reference_details: list[dict[str, Any]],
    model_negative_details: list[dict[str, Any]],
) -> dict[str, Any]:
    pos_scores = _scores(positive_details)
    ref_scores = _scores(reference_details)
    model_scores = _scores(model_negative_details)
    threshold = float(positive_details[0]["threshold_5fpr"]) if positive_details else 0.0
    fp_rows = [row for row in reference_details if float(row["score"]) >= threshold]
    fn_rows = [row for row in positive_details if float(row["score"]) < threshold]
    inversion_count, tie_count = _pairwise_inversions(pos_scores, ref_scores)
    total_pairs = len(pos_scores) * len(ref_scores)
    variants = [
        _variant("all_full164", positive_details, reference_details),
        _variant(
            "scoreable_only_exclude_insufficient",
            [row for row in positive_details if not row.get("insufficient_evidence")],
            [row for row in reference_details if not row.get("insufficient_evidence")],
        ),
        _variant(
            "drop_positive_insufficient_only",
            [row for row in positive_details if not row.get("insufficient_evidence")],
            reference_details,
        ),
        _variant(
            "drop_negative_insufficient_only",
            positive_details,
            [row for row in reference_details if not row.get("insufficient_evidence")],
        ),
    ]
    return {
        "artifact_type": "sawr_auroc_bottleneck",
        "positive_n": len(positive_details),
        "reference_negative_n": len(reference_details),
        "model_negative_n": len(model_negative_details),
        "threshold": threshold,
        "auroc": _auc(pos_scores, ref_scores),
        "model_negative_auroc": _auc(pos_scores, model_scores),
        "tpr_at_5fpr": _tpr_at_fpr(pos_scores, ref_scores, 0.05),
        "partial_auc_at_5fpr": _partial_auc(pos_scores, ref_scores, 0.05),
        "fp_count_at_threshold": len(fp_rows),
        "fn_count_at_threshold": len(fn_rows),
        "pairwise_negative_above_positive": inversion_count,
        "pairwise_ties": tie_count,
        "pairwise_inversion_rate": inversion_count / total_pairs if total_pairs else 0.0,
        "positive_score_quantiles": _quantiles(pos_scores),
        "reference_score_quantiles": _quantiles(ref_scores),
        "model_negative_score_quantiles": _quantiles(model_scores),
        "variants": variants,
        "false_positives": _compact_detail_rows(fp_rows, descending=True),
        "lowest_positive_false_negatives": _compact_detail_rows(fn_rows, descending=False)[:30],
    }


def _build_pass_report(
    positive_inputs: list[dict[str, Any]],
    positive_details: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    audit_summary: dict[str, Any],
    generation_summary: dict[str, Any],
) -> dict[str, Any]:
    detail_by_id = {str(row["id"]): row for row in positive_details}
    input_by_id = {str(row["id"]): row for row in positive_inputs}
    audit_hits = audit_summary.get("per_sample_accepted_hit_count") or {}
    rows = []
    reason_counts: Counter[str] = Counter()
    taxonomy_counts: Counter[str] = Counter()
    score_by_reason: dict[str, list[float]] = defaultdict(list)
    for exec_row in execution_rows:
        sample_id = str(exec_row["id"])
        passed = bool(exec_row.get("passed"))
        reason = str(exec_row.get("reason") or ("passed" if passed else "unknown"))
        code = str(input_by_id.get(sample_id, {}).get("final_code") or "")
        detail = detail_by_id.get(sample_id, {})
        taxonomy = _classify_pass_failure(code, reason)
        reason_counts[reason] += 1
        taxonomy_counts[taxonomy] += 1
        score = float(detail.get("score", 0.0))
        score_by_reason[reason].append(score)
        rows.append(
            {
                "id": sample_id,
                "passed": passed,
                "reason": reason,
                "taxonomy": taxonomy,
                "syntax_valid": _syntax_valid(code),
                "code_chars": len(code),
                "detector_score": score,
                "scoreable_contexts": int(detail.get("scoreable_contexts", 0) or 0),
                "proxy_windows": int(detail.get("proxy_windows", 0) or 0),
                "insufficient_evidence": bool(detail.get("insufficient_evidence", False)),
                "accepted_hits": int(audit_hits.get(sample_id, 0) or 0),
            }
        )
    passed_count = sum(1 for row in rows if row["passed"])
    return {
        "artifact_type": "sawr_pass_bottleneck",
        "total": len(rows),
        "passed": passed_count,
        "pass_at_1": passed_count / len(rows) if rows else 0.0,
        "reason_counts": dict(reason_counts),
        "taxonomy_counts": dict(taxonomy_counts),
        "score_by_reason_mean": {
            reason: sum(values) / len(values) for reason, values in score_by_reason.items()
        },
        "generation_summary": generation_summary,
        "audit_accepted_samples": audit_summary.get("accepted_samples"),
        "audit_accepted_generation_time_window_rows": audit_summary.get(
            "accepted_generation_time_window_rows"
        ),
        "rows": rows,
        "syntax_error_examples": [row for row in rows if row["reason"] == "syntax_error"][:20],
        "failed_test_examples": [row for row in rows if row["reason"] == "failed_tests"][:20],
    }


def _build_delta_report(
    final_positive: list[dict[str, Any]],
    final_negative: list[dict[str, Any]],
    old_positive: list[dict[str, Any]],
    old_negative: list[dict[str, Any]],
) -> dict[str, Any]:
    final_by_id = {("positive", str(row["id"])): row for row in final_positive}
    final_by_id.update({("reference_negative", str(row["id"])): row for row in final_negative})
    old_by_id = {("positive", str(row["id"])): row for row in old_positive}
    old_by_id.update({("reference_negative", str(row["id"])): row for row in old_negative})
    rows = []
    for key in sorted(set(final_by_id) | set(old_by_id)):
        final = final_by_id.get(key, {})
        old = old_by_id.get(key, {})
        rows.append(
            {
                "kind": key[0],
                "id": key[1],
                "old_score": old.get("score"),
                "final_score": final.get("score"),
                "delta": (
                    float(final["score"]) - float(old["score"])
                    if "score" in final and "score" in old
                    else None
                ),
                "old_insufficient": old.get("insufficient_evidence"),
                "final_insufficient": final.get("insufficient_evidence"),
                "final_scoreable_contexts": final.get("scoreable_contexts"),
                "final_proxy_windows": final.get("proxy_windows"),
            }
        )
    return {
        "artifact_type": "sawr_detector_delta",
        "rows": rows,
        "positive_count": len(final_positive),
        "reference_negative_count": len(final_negative),
    }


def _build_input_integrity_report(
    positive_inputs: list[dict[str, Any]],
    reference_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = {"id", "dataset", "prompt", "final_code", "generated_code"}
    forbidden = {
        "watermark_params",
        "blocks",
        "detector_score",
        "z_score",
        "p_value",
        "score",
        "logits",
        "sampling_trace",
        "retry_trace",
        "rollback_trace",
        "audit",
        "audit_event_id",
    }
    result = {}
    for name, rows in (("positive", positive_inputs), ("reference_negative", reference_inputs)):
        fields = sorted({field for row in rows for field in row})
        forbidden_hits = sorted(
            field
            for field in fields
            if field in forbidden or field.startswith("generation_") or field.startswith("audit_")
        )
        unknown = sorted(set(fields) - allowed)
        result[name] = {
            "rows": len(rows),
            "fields": fields,
            "unknown_fields": unknown,
            "forbidden_hits": forbidden_hits,
            "sha256": _sha256_rows(rows),
        }
    result["status"] = (
        "pass"
        if all(not item["unknown_fields"] and not item["forbidden_hits"] for item in result.values())
        else "fail"
    )
    return {"artifact_type": "sawr_detector_input_integrity", **result}


def _classify_pass_failure(code: str, reason: str) -> str:
    if reason == "passed":
        return "passed"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax_error"
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return "missing_target_function"
    lowered = code.lower()
    if "todo" in lowered or "notimplemented" in lowered or "pass\n" in lowered:
        return "placeholder_or_incomplete"
    if reason == "failed_tests":
        return "semantic_or_edge_case_failure"
    return reason


def _variant(name: str, positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> dict[str, Any]:
    pos_scores = _scores(positives)
    neg_scores = _scores(negatives)
    return {
        "name": name,
        "positive_n": len(positives),
        "negative_n": len(negatives),
        "auroc": _auc(pos_scores, neg_scores),
        "tpr_at_5fpr": _tpr_at_fpr(pos_scores, neg_scores, 0.05),
    }


def _compact_detail_rows(rows: list[dict[str, Any]], *, descending: bool) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=descending)
    return [
        {
            "id": str(row["id"]),
            "score": float(row["score"]),
            "scoreable_contexts": int(row.get("scoreable_contexts", 0) or 0),
            "proxy_windows": int(row.get("proxy_windows", 0) or 0),
            "insufficient_evidence": bool(row.get("insufficient_evidence", False)),
        }
        for row in ordered
    ]


def _scores(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["score"]) for row in rows if "score" in row]


def _auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return 0.0
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _tpr_at_fpr(positive: list[float], negative: list[float], target_fpr: float) -> float:
    if not positive or not negative:
        return 0.0
    best = 0.0
    for threshold in sorted(set(positive + negative), reverse=True):
        fpr = sum(score >= threshold for score in negative) / len(negative)
        if fpr <= target_fpr:
            best = max(best, sum(score >= threshold for score in positive) / len(positive))
    return best


def _partial_auc(positive: list[float], negative: list[float], max_fpr: float) -> float:
    if not positive or not negative:
        return 0.0
    points = [(0.0, 0.0)]
    for threshold in sorted(set(positive + negative), reverse=True):
        fpr = sum(score >= threshold for score in negative) / len(negative)
        tpr = sum(score >= threshold for score in positive) / len(positive)
        if fpr <= max_fpr:
            points.append((fpr, tpr))
    points.append((max_fpr, points[-1][1]))
    points = sorted(set(points))
    area = 0.0
    for (left_fpr, left_tpr), (right_fpr, right_tpr) in zip(points, points[1:]):
        area += (right_fpr - left_fpr) * (left_tpr + right_tpr) / 2
    return area / max_fpr if max_fpr else 0.0


def _pairwise_inversions(positive: list[float], negative: list[float]) -> tuple[int, int]:
    inversions = 0
    ties = 0
    for pos in positive:
        for neg in negative:
            if neg > pos:
                inversions += 1
            elif neg == pos:
                ties += 1
    return inversions, ties


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def _quantile(ordered: list[float], q: float) -> float:
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _sha256_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _auroc_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AUROC Bottleneck",
        "",
        f"- AUROC: {report['auroc']:.10f}",
        f"- TPR@5%FPR: {report['tpr_at_5fpr']:.10f}",
        f"- FP at threshold: {report['fp_count_at_threshold']}",
        f"- FN at threshold: {report['fn_count_at_threshold']}",
        f"- Pairwise inversion rate: {report['pairwise_inversion_rate']:.6f}",
        f"- Model-negative AUROC: {report['model_negative_auroc']:.10f}",
        "",
        "## Variants",
        "",
        "| variant | positive n | negative n | AUROC | TPR@5%FPR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["variants"]:
        lines.append(
            f"| {row['name']} | {row['positive_n']} | {row['negative_n']} | "
            f"{row['auroc']:.6f} | {row['tpr_at_5fpr']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _pass_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pass Bottleneck",
        "",
        f"- pass@1: {report['pass_at_1']:.10f}",
        f"- passed: {report['passed']}/{report['total']}",
        "",
        "## Reasons",
        "",
    ]
    for key, count in sorted(report["reason_counts"].items()):
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Taxonomy", ""])
    for key, count in sorted(report["taxonomy_counts"].items()):
        lines.append(f"- {key}: {count}")
    return "\n".join(lines) + "\n"


def _delta_markdown(report: dict[str, Any]) -> str:
    return (
        "# Detector Delta\n\n"
        f"- positive rows: {report['positive_count']}\n"
        f"- reference negative rows: {report['reference_negative_count']}\n"
    )


def _integrity_markdown(report: dict[str, Any]) -> str:
    lines = ["# Detector Input Integrity", "", f"Status: `{report['status']}`", ""]
    for name in ("positive", "reference_negative"):
        item = report[name]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- rows: {item['rows']}")
        lines.append(f"- fields: `{', '.join(item['fields'])}`")
        lines.append(f"- forbidden hits: {item['forbidden_hits']}")
        lines.append(f"- unknown fields: {item['unknown_fields']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
