#!/usr/bin/env python
"""Diagnostic-only SAWR sparse statistic and conformal calibration analysis.

This script reads existing final-code detector detail artifacts. It does not
change production detector logic, default thresholds, default sample scores, or
the detector input contract. Diagnostic sidecars are used only for explanation
and oracle upper-bound analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_REPO_ROOT = Path("/root/autodl-tmp/WFCLLM_sawr_detect_codex_20260616")
DEFAULT_E1_ROOT = (
    DEFAULT_REPO_ROOT / "data/diagnostics/sawr_e1_e2_sidecar_probe_20260616"
)
DEFAULT_FULL164_ROOT = (
    DEFAULT_REPO_ROOT / "data/diagnostics/sawr_full164_bestgen_20260616"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_REPO_ROOT / "data/diagnostics/sawr_sparse_stat_conformal_20260616"
)
TARGET_FPR = 0.05
ALPHA_CONTEXT = 0.05
LN10 = math.log(10.0)
P_EPS = 1e-15
NORMAL = statistics.NormalDist()

STATISTIC_ORDER = [
    "current_mean",
    "context_max",
    "top2_mean",
    "top3_mean",
    "top2_sum",
    "top3_sum",
    "count_sig_contexts",
    "tippett_min_p",
    "fisher",
    "stouffer",
    "cauchy",
    "berk_jones",
    "higher_criticism",
]
STABILITY_STATISTICS = [
    "current_mean",
    "context_max",
    "top2_sum",
    "cauchy",
    "berk_jones",
    "count_sig_contexts",
]
BASELINE_COMPARISONS = [
    {
        "run": "current detector split positive test",
        "positive_n": 82,
        "negative_n": 82,
        "tpr": 0.219512,
        "fpr": 0.121951,
        "auroc": 0.611392,
    },
    {
        "run": "current detector all-positive",
        "positive_n": 164,
        "negative_n": 82,
        "tpr": 0.268293,
        "fpr": 0.121951,
        "auroc": 0.633998,
    },
    {
        "run": "old 164 d8/g0.25",
        "positive_n": 82,
        "negative_n": 82,
        "tpr": 0.0366,
        "fpr": 0.0244,
        "auroc": 0.4926,
    },
    {
        "run": "old 164 d4/g0.5",
        "positive_n": 82,
        "negative_n": 82,
        "tpr": 0.0610,
        "fpr": 0.0122,
        "auroc": 0.5073,
    },
    {
        "run": "20-sample t03_p4_g025",
        "positive_n": 20,
        "negative_n": 82,
        "tpr": 0.4000,
        "fpr": 0.1220,
        "auroc": 0.6354,
    },
]


class ConformalDecision(NamedTuple):
    score: float
    p_conformal: float
    is_detected: bool
    is_detected_without_gate: bool
    insufficient_evidence: bool


STATISTIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "current_mean": {
        "formula": "mean_i(context_score_i)",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "sample-level strict conformal p-value from negative calibration scores",
        "production_candidate": "baseline only; this is the current production sample aggregation",
        "risk": "sparse hits can be diluted by many zero or weak contexts",
    },
    "context_max": {
        "formula": "max_i(context_score_i)",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "sample-level strict conformal p-value from negative calibration scores",
        "production_candidate": "experimental candidate only",
        "risk": "single-context false positive tail may be unstable with small negative calibration",
    },
    "top2_mean": {
        "formula": "mean of the largest two context_score values; if n<2 use available contexts",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "sample-level strict conformal p-value from negative calibration scores",
        "production_candidate": "experimental candidate only",
        "risk": "choice of k must be pre-registered or validated on larger negatives",
    },
    "top3_mean": {
        "formula": "mean of the largest three context_score values; if n<3 use available contexts",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "sample-level strict conformal p-value from negative calibration scores",
        "production_candidate": "experimental candidate only",
        "risk": "choice of k must be pre-registered or validated on larger negatives",
    },
    "top2_sum": {
        "formula": "sum of the largest two context_score values",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "sample-level strict conformal p-value from negative calibration scores",
        "production_candidate": "experimental candidate only",
        "risk": "can correlate with number of contexts unless calibrated carefully",
    },
    "top3_sum": {
        "formula": "sum of the largest three context_score values",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "sample-level strict conformal p-value from negative calibration scores",
        "production_candidate": "experimental candidate only",
        "risk": "can correlate with number of contexts unless calibrated carefully",
    },
    "count_sig_contexts": {
        "formula": "count_i[p_i <= alpha_context], alpha_context=0.05 predefined diagnostic grid",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "predefined context alpha plus sample-level strict conformal p-value",
        "production_candidate": "experimental candidate only",
        "risk": "ties and context dependence make empirical calibration mandatory",
    },
    "tippett_min_p": {
        "formula": "min_i(p_i), reported on -log10 scale as max_i(context_score_i)",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "sample-level strict conformal p-value from negative calibration scores",
        "production_candidate": "experimental candidate only",
        "risk": "equivalent to context_max on the current context_score scale",
    },
    "fisher": {
        "formula": "-2 * sum_i(log(p_i)); p_i = 10 ** (-context_score_i)",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "empirical/conformal only; no chi-square analytic p-value is used",
        "production_candidate": "experimental candidate only",
        "risk": "depends on context count and dependence, so analytic null is invalid here",
    },
    "stouffer": {
        "formula": "sum_i(Phi^-1(1-p_i)) / sqrt(n_contexts)",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "empirical/conformal only; no independence assumption",
        "production_candidate": "experimental candidate only",
        "risk": "dense weak-signal statistic may underperform sparse recovered hits",
    },
    "cauchy": {
        "formula": "mean_i(tan((0.5 - p_i) * pi))",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "empirical/conformal only; analytic p-value is not used",
        "production_candidate": "experimental candidate only",
        "risk": "very tail-sensitive; small negative calibration may make FPR unstable",
    },
    "berk_jones": {
        "formula": "max_i n * KL(i/n || p_(i)) for sorted p_(i) with i/n > p_(i)",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "empirical/conformal only",
        "production_candidate": "experimental candidate only",
        "risk": "sparse-mixture diagnostic; needs larger negative corpus before production",
    },
    "higher_criticism": {
        "formula": "max_i sqrt(n) * (i/n - p_(i)) / sqrt(p_(i)*(1-p_(i)))",
        "final_code_only": True,
        "uses_diagnostic_sidecar": False,
        "calibration_method": "empirical/conformal only",
        "production_candidate": "optional diagnostic only",
        "risk": "unstable for tiny context counts; included only with clamped finite p-values",
    },
    "recovered_hit_oracle": {
        "formula": "1 if E1/E2 sidecar shows any final proxy recovered hit for the sample",
        "final_code_only": False,
        "uses_diagnostic_sidecar": True,
        "calibration_method": "not calibrated as detector; oracle upper bound only",
        "production_candidate": "not a detector claim",
        "risk": "uses diagnostic sidecar/oracle information unavailable to production detector input",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run diagnostic-only SAWR sparse statistic conformal analysis.",
    )
    parser.add_argument("--e1-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--full164-root", type=Path, default=DEFAULT_FULL164_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    parser.add_argument("--alpha-context", type=float, default=ALPHA_CONTEXT)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--stability-seeds", default="0,1,2,3,4,20260616")
    args = parser.parse_args()

    run_analysis(
        e1_root=args.e1_root,
        full164_root=args.full164_root,
        output_root=args.output_root,
        target_fpr=args.target_fpr,
        alpha_context=args.alpha_context,
        seed=args.seed,
        stability_seeds=[int(item) for item in args.stability_seeds.split(",")],
    )
    return 0


def run_analysis(
    *,
    e1_root: Path,
    full164_root: Path,
    output_root: Path,
    target_fpr: float,
    alpha_context: float,
    seed: int,
    stability_seeds: list[int],
) -> None:
    analysis_dir = output_root / "analysis"
    diagnostics_dir = output_root / "diagnostics"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    run_plan_path = write_run_plan(output_root, e1_root, full164_root, target_fpr, alpha_context, seed)
    audit_path = write_input_artifact_audit(analysis_dir, e1_root, full164_root)
    definitions_path = write_statistic_definitions(analysis_dir)

    e1_results = run_20sample_analysis(
        e1_root=e1_root,
        output_root=output_root,
        target_fpr=target_fpr,
        alpha_context=alpha_context,
        seed=seed,
    )
    full164_results = run_full164_analysis(
        full164_root=full164_root,
        output_root=output_root,
        target_fpr=target_fpr,
        alpha_context=alpha_context,
    )
    stability_results = run_calibration_stability(
        full164_root=full164_root,
        output_root=output_root,
        target_fpr=target_fpr,
        alpha_context=alpha_context,
        seeds=stability_seeds,
    )
    stop_go_path = write_stop_go_decision(
        analysis_dir,
        full164_results=full164_results,
        stability_results=stability_results,
        target_fpr=target_fpr,
    )
    final_report_path = write_final_report(
        output_root,
        e1_root=e1_root,
        full164_root=full164_root,
        e1_results=e1_results,
        full164_results=full164_results,
        stability_results=stability_results,
        target_fpr=target_fpr,
    )
    write_manifest(
        output_root,
        [
            run_plan_path,
            audit_path,
            definitions_path,
            output_root / "analysis/statistic_results_20sample.json",
            output_root / "analysis/statistic_results_20sample.csv",
            output_root / "analysis/statistic_results_20sample.md",
            output_root / "analysis/statistic_results_full164.json",
            output_root / "analysis/statistic_results_full164.csv",
            output_root / "analysis/statistic_results_full164.md",
            output_root / "analysis/calibration_stability.json",
            output_root / "analysis/calibration_stability.md",
            output_root / "analysis/loss_reclassification_20sample.json",
            output_root / "analysis/loss_reclassification_20sample.md",
            stop_go_path,
            output_root / "diagnostics/context_score_cache_20sample.jsonl",
            output_root / "diagnostics/context_score_cache_full164.jsonl",
            output_root / "diagnostics/conformal_thresholds.json",
            final_report_path,
        ],
        e1_root=e1_root,
        full164_root=full164_root,
    )


def compute_statistic(
    row: dict[str, Any],
    statistic: str,
    *,
    alpha_context: float,
) -> float:
    scores = context_scores(row)
    if statistic == "current_mean":
        return sum(scores) / len(scores) if scores else safe_float(row.get("score"))
    if statistic in {"context_max", "tippett_min_p"}:
        return max(scores) if scores else 0.0
    if statistic == "top2_mean":
        return topk_mean(scores, 2)
    if statistic == "top3_mean":
        return topk_mean(scores, 3)
    if statistic == "top2_sum":
        return topk_sum(scores, 2)
    if statistic == "top3_sum":
        return topk_sum(scores, 3)
    if statistic == "count_sig_contexts":
        return float(sum(1 for p_value in context_p_values(row) if p_value <= alpha_context))
    if statistic == "fisher":
        return 2.0 * LN10 * sum(scores)
    if statistic == "stouffer":
        p_values = context_p_values(row)
        if not p_values:
            return 0.0
        z_values = [NORMAL.inv_cdf(1.0 - clamp_p(p_value)) for p_value in p_values]
        return sum(z_values) / math.sqrt(len(z_values))
    if statistic == "cauchy":
        p_values = context_p_values(row)
        if not p_values:
            return 0.0
        return sum(math.tan((0.5 - clamp_p(p_value)) * math.pi) for p_value in p_values) / len(p_values)
    if statistic == "berk_jones":
        return berk_jones(context_p_values(row))
    if statistic == "higher_criticism":
        return higher_criticism(context_p_values(row))
    raise ValueError(f"unknown statistic: {statistic}")


def conformal_p_value(score: float, calibration_scores: list[float]) -> float:
    if not calibration_scores:
        return 1.0
    upper_tail_count = sum(1 for calibration_score in calibration_scores if calibration_score >= score)
    return (1.0 + upper_tail_count) / (1.0 + len(calibration_scores))


def detect_with_conformal(
    *,
    score: float,
    calibration_scores: list[float],
    target_fpr: float,
    insufficient_evidence: bool,
) -> ConformalDecision:
    p_conformal = conformal_p_value(score, calibration_scores)
    detected_without_gate = p_conformal <= target_fpr
    return ConformalDecision(
        score=score,
        p_conformal=p_conformal,
        is_detected=detected_without_gate and not insufficient_evidence,
        is_detected_without_gate=detected_without_gate,
        insufficient_evidence=insufficient_evidence,
    )


def run_20sample_analysis(
    *,
    e1_root: Path,
    output_root: Path,
    target_fpr: float,
    alpha_context: float,
    seed: int,
) -> dict[str, Any]:
    detection_dir = e1_root / "detection/gamma025"
    analysis_dir = output_root / "analysis"
    diagnostics_dir = output_root / "diagnostics"

    positives = load_jsonl(detection_dir / "positive_details.jsonl")
    negatives_all = load_jsonl(detection_dir / "negative_details.jsonl")
    negative_calibration, negative_test, split_manifest = split_by_task(
        negatives_all,
        calibration_ratio=0.5,
        seed=seed,
    )
    write_json(diagnostics_dir / "split_manifest_20sample.json", split_manifest)
    write_context_cache(
        diagnostics_dir / "context_score_cache_20sample.jsonl",
        [
            ("20sample_positive", "positive", positives),
            ("20sample_negative_calibration", "negative_calibration", negative_calibration),
            ("20sample_negative_test", "negative_test", negative_test),
        ],
        alpha_context=alpha_context,
    )

    rows = [
        evaluate_statistic(
            setting="20sample_probe",
            statistic=statistic,
            positives=positives,
            negatives=negative_test,
            calibration_negatives=negative_calibration,
            target_fpr=target_fpr,
            alpha_context=alpha_context,
            uses_diagnostic_sidecar=False,
            note=(
                "20-sample negative calibration/test split is post-hoc from "
                "negative_details; context_score values come from the original "
                "E1/E2 calibration artifact."
            ),
        )
        for statistic in STATISTIC_ORDER
    ]
    oracle = recovered_hit_oracle(
        positives=positives,
        proxy_recall_rows=load_jsonl(e1_root / "analysis/proxy_recall_results.jsonl"),
    )
    rows.append(oracle_result_row("20sample_probe", oracle, len(positives)))

    result_payload = {
        "target_fpr": target_fpr,
        "alpha_context": alpha_context,
        "setting": "20sample_probe",
        "positive_count": len(positives),
        "negative_calibration_count": len(negative_calibration),
        "negative_test_count": len(negative_test),
        "split_manifest": split_manifest,
        "strictness_note": (
            "Sample-level conformal split is task-disjoint and uses held-out "
            "negative test rows. Because this probe did not persist separately "
            "scored negative calibration/test details, its context_score fields "
            "were precomputed under the original 164-negative calibration "
            "artifact. Treat 20-sample FPR as diagnostic; full164 is the primary "
            "strict held-out FPR evidence."
        ),
        "oracle": oracle,
        "results": rows,
    }
    write_json(analysis_dir / "statistic_results_20sample.json", result_payload)
    write_csv(analysis_dir / "statistic_results_20sample.csv", flatten_result_rows(rows))
    write_results_markdown(
        analysis_dir / "statistic_results_20sample.md",
        title="20-Sample E1/E2 Probe Sparse Statistic Results",
        rows=rows,
        intro=[
            "Primary rule: p_conformal <= 0.05 and insufficient_evidence is false.",
            "The recovered_hit_oracle row uses sidecar/oracle information and is not a final-code detector claim.",
        ],
    )
    loss_payload = build_loss_reclassification(
        e1_root=e1_root,
        statistic_rows=rows,
        positives=positives,
        target_fpr=target_fpr,
    )
    write_json(analysis_dir / "loss_reclassification_20sample.json", loss_payload)
    write_loss_reclassification_markdown(
        analysis_dir / "loss_reclassification_20sample.md",
        loss_payload,
    )
    return result_payload


def run_full164_analysis(
    *,
    full164_root: Path,
    output_root: Path,
    target_fpr: float,
    alpha_context: float,
) -> dict[str, Any]:
    detection_dir = full164_root / "detection/gamma025"
    analysis_dir = output_root / "analysis"
    diagnostics_dir = output_root / "diagnostics"

    negative_calibration = load_jsonl(detection_dir / "negative_calibration_details.jsonl")
    negative_test = load_jsonl(detection_dir / "negative_test_details.jsonl")
    positive_split = load_jsonl(detection_dir / "positive_details.jsonl")
    positive_all = load_jsonl(detection_dir / "all_positive_details.jsonl")
    all_negative = load_jsonl(detection_dir / "all_negative_details.jsonl")

    write_context_cache(
        diagnostics_dir / "context_score_cache_full164.jsonl",
        [
            ("full164_split_positive", "positive", positive_split),
            ("full164_all_positive", "positive", positive_all),
            ("full164_negative_calibration", "negative_calibration", negative_calibration),
            ("full164_negative_test", "negative_test", negative_test),
        ],
        alpha_context=alpha_context,
    )

    rows: list[dict[str, Any]] = []
    for setting, positives in (
        ("full164_split_positive", positive_split),
        ("full164_all_positive", positive_all),
    ):
        for statistic in STATISTIC_ORDER:
            rows.append(
                evaluate_statistic(
                    setting=setting,
                    statistic=statistic,
                    positives=positives,
                    negatives=negative_test,
                    calibration_negatives=negative_calibration,
                    target_fpr=target_fpr,
                    alpha_context=alpha_context,
                    uses_diagnostic_sidecar=False,
                    note=(
                        "Full164 has no proxy-window sidecar; this is a "
                        "context-summary-level statistic using detector details."
                    ),
                )
            )

    threshold_payload = {
        "target_fpr": target_fpr,
        "alpha_context": alpha_context,
        "threshold_definition": (
            "Primary rule is conformal p-value. threshold_exclusive is the "
            "calibration score boundary above which p_conformal can pass the "
            "target under the observed calibration scores; ties are resolved by "
            "the p-value rule."
        ),
        "20sample": load_json(output_root / "analysis/statistic_results_20sample.json")["results"],
        "full164": rows,
    }
    write_json(diagnostics_dir / "conformal_thresholds.json", threshold_payload)

    result_payload = {
        "target_fpr": target_fpr,
        "alpha_context": alpha_context,
        "setting": "full164_bestgen",
        "positive_split_count": len(positive_split),
        "positive_all_count": len(positive_all),
        "negative_calibration_count": len(negative_calibration),
        "negative_test_count": len(negative_test),
        "all_negative_count": len(all_negative),
        "proxy_window_level_available": False,
        "context_summary_level_note": (
            "Full164 details include context_summaries with context_score and "
            "context_raw, but no proxy_window_diagnostics. Cauchy/Berk-Jones "
            "therefore combine context-summary p-value approximations "
            "p=10**(-context_score), not exact proxy-window-level p-values."
        ),
        "baseline_comparisons": BASELINE_COMPARISONS,
        "results": rows,
    }
    write_json(analysis_dir / "statistic_results_full164.json", result_payload)
    write_csv(analysis_dir / "statistic_results_full164.csv", flatten_result_rows(rows))
    write_results_markdown(
        analysis_dir / "statistic_results_full164.md",
        title="Full164 Bestgen Sparse Statistic Results",
        rows=rows,
        intro=[
            "Primary rule: p_conformal <= 0.05 and insufficient_evidence is false.",
            "Full164 does not contain proxy-window sidecars, so all sparse combinations are context-summary-level diagnostics.",
            "Baseline comparison values are copied from FULL164_BESTGEN_REPORT.md.",
        ],
        baseline_rows=BASELINE_COMPARISONS,
    )
    return result_payload


def run_calibration_stability(
    *,
    full164_root: Path,
    output_root: Path,
    target_fpr: float,
    alpha_context: float,
    seeds: list[int],
) -> dict[str, Any]:
    detection_dir = full164_root / "detection/gamma025"
    analysis_dir = output_root / "analysis"
    positives = load_jsonl(detection_dir / "all_positive_details.jsonl")
    negatives = load_jsonl(detection_dir / "all_negative_details.jsonl")

    split_rows: list[dict[str, Any]] = []
    for seed in seeds:
        negative_calibration, negative_test, split_manifest = split_by_task(
            negatives,
            calibration_ratio=0.5,
            seed=seed,
        )
        for statistic in STABILITY_STATISTICS:
            result = evaluate_statistic(
                setting=f"full164_stability_seed_{seed}",
                statistic=statistic,
                positives=positives,
                negatives=negative_test,
                calibration_negatives=negative_calibration,
                target_fpr=target_fpr,
                alpha_context=alpha_context,
                uses_diagnostic_sidecar=False,
                note="Repeated task-disjoint split stability over full164 all_negative_details.",
            )
            split_rows.append(
                {
                    "seed": seed,
                    "statistic": statistic,
                    "calibration_negative_count": len(negative_calibration),
                    "heldout_negative_count": len(negative_test),
                    "calibration_task_count": split_manifest["calibration_task_count"],
                    "heldout_task_count": split_manifest["test_task_count"],
                    "threshold_exclusive": result["threshold_exclusive"],
                    "tpr": result["tpr"],
                    "fpr": result["observed_fpr"],
                    "tpr_without_gate": result["tpr_without_gate"],
                    "fpr_without_gate": result["observed_fpr_without_gate"],
                    "auroc": result["auroc"],
                    "false_positive_count": len(result["false_positive_ids"]),
                    "detected_positive_count": len(result["detected_positive_ids"]),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for statistic in STABILITY_STATISTICS:
        rows = [row for row in split_rows if row["statistic"] == statistic]
        fprs = [safe_float(row["fpr"]) for row in rows]
        tprs = [safe_float(row["tpr"]) for row in rows]
        thresholds = [safe_float(row["threshold_exclusive"]) for row in rows]
        summary_rows.append(
            {
                "statistic": statistic,
                "runs": len(rows),
                "fpr_mean": mean(fprs),
                "fpr_std": stdev(fprs),
                "fpr_min": min(fprs) if fprs else 0.0,
                "fpr_max": max(fprs) if fprs else 0.0,
                "tpr_mean": mean(tprs),
                "tpr_std": stdev(tprs),
                "tpr_min": min(tprs) if tprs else 0.0,
                "tpr_max": max(tprs) if tprs else 0.0,
                "threshold_mean": mean(thresholds),
                "threshold_std": stdev(thresholds),
                "threshold_min": min(thresholds) if thresholds else 0.0,
                "threshold_max": max(thresholds) if thresholds else 0.0,
                "calibration_negative_count": rows[0]["calibration_negative_count"] if rows else 0,
                "heldout_negative_count": rows[0]["heldout_negative_count"] if rows else 0,
                "has_fpr_gt_005": any(value > target_fpr for value in fprs),
                "has_fpr_gt_010": any(value > 0.10 for value in fprs),
                "passes_005_all_splits": bool(fprs) and max(fprs) <= target_fpr,
                "meaningfully_above_current_all_positive_tpr": (
                    bool(tprs) and mean(tprs) > 0.268293 + 0.05
                ),
            }
        )

    payload = {
        "target_fpr": target_fpr,
        "alpha_context": alpha_context,
        "seeds": seeds,
        "positive_set": "full164_all_positive_details",
        "negative_source": "full164_all_negative_details",
        "statistics": STABILITY_STATISTICS,
        "split_rows": split_rows,
        "summary": summary_rows,
        "interpretation_note": (
            "Only 82 calibration negatives per split are available. A 5% "
            "tail decision is controlled by a handful of calibration samples, "
            "so threshold variance and held-out FPR excursions should not be "
            "over-interpreted."
        ),
    }
    write_json(analysis_dir / "calibration_stability.json", payload)
    write_stability_markdown(analysis_dir / "calibration_stability.md", payload)
    return payload


def evaluate_statistic(
    *,
    setting: str,
    statistic: str,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    calibration_negatives: list[dict[str, Any]],
    target_fpr: float,
    alpha_context: float,
    uses_diagnostic_sidecar: bool,
    note: str,
) -> dict[str, Any]:
    calibration_scores = [
        compute_statistic(row, statistic, alpha_context=alpha_context)
        for row in calibration_negatives
    ]
    positive_eval = score_and_decide(
        positives,
        statistic=statistic,
        calibration_scores=calibration_scores,
        target_fpr=target_fpr,
        alpha_context=alpha_context,
        label="positive",
    )
    negative_eval = score_and_decide(
        negatives,
        statistic=statistic,
        calibration_scores=calibration_scores,
        target_fpr=target_fpr,
        alpha_context=alpha_context,
        label="negative_test",
    )
    positive_scores = [item["score"] for item in positive_eval]
    negative_scores = [item["score"] for item in negative_eval]
    positive_p_values = [item["p_conformal"] for item in positive_eval]
    negative_p_values = [item["p_conformal"] for item in negative_eval]
    detected_positive_ids = [
        item["id"] for item in positive_eval if item["is_detected"]
    ]
    false_positive_ids = [
        item["id"] for item in negative_eval if item["is_detected"]
    ]
    false_negative_ids = [
        item["id"] for item in positive_eval if not item["is_detected"]
    ]
    high_score_insufficient = [
        {
            "label": item["label"],
            "id": item["id"],
            "score": item["score"],
            "p_conformal": item["p_conformal"],
            "proxy_windows": item["proxy_windows"],
            "scoreable_contexts": item["scoreable_contexts"],
        }
        for item in [*positive_eval, *negative_eval]
        if item["is_detected_without_gate"] and item["insufficient_evidence"]
    ]
    definition = STATISTIC_DEFINITIONS[statistic]
    return {
        "setting": setting,
        "statistic": statistic,
        "formula": definition["formula"],
        "uses_final_code_only": definition["final_code_only"],
        "uses_diagnostic_sidecar": uses_diagnostic_sidecar,
        "calibration_method": definition["calibration_method"],
        "production_readiness": definition["production_candidate"],
        "risk": definition["risk"],
        "target_fpr": target_fpr,
        "alpha_context": alpha_context,
        "threshold_exclusive": threshold_exclusive(calibration_scores, target_fpr),
        "threshold_rule": "primary decision uses p_conformal <= target_fpr; threshold is diagnostic",
        "conformal_p_value_rule": "(1 + count(calibration_score >= sample_score)) / (1 + n_calibration)",
        "positive_count": len(positive_eval),
        "negative_count": len(negative_eval),
        "calibration_negative_count": len(calibration_scores),
        "detected_positive_count": len(detected_positive_ids),
        "false_positive_count": len(false_positive_ids),
        "false_negative_count": len(false_negative_ids),
        "tpr": rate(len(detected_positive_ids), len(positive_eval)),
        "observed_fpr": rate(len(false_positive_ids), len(negative_eval)),
        "tpr_without_gate": rate(
            sum(1 for item in positive_eval if item["is_detected_without_gate"]),
            len(positive_eval),
        ),
        "observed_fpr_without_gate": rate(
            sum(1 for item in negative_eval if item["is_detected_without_gate"]),
            len(negative_eval),
        ),
        "auroc": auroc(positive_scores, negative_scores),
        "positive_insufficient_count": sum(1 for item in positive_eval if item["insufficient_evidence"]),
        "negative_insufficient_count": sum(1 for item in negative_eval if item["insufficient_evidence"]),
        "high_score_insufficient_count": len(high_score_insufficient),
        "positive_score_quantiles": quantile_summary(positive_scores),
        "negative_score_quantiles": quantile_summary(negative_scores),
        "positive_p_value_quantiles": quantile_summary(positive_p_values),
        "negative_p_value_quantiles": quantile_summary(negative_p_values),
        "detected_positive_ids": detected_positive_ids,
        "false_positive_ids": false_positive_ids,
        "false_negative_ids": false_negative_ids,
        "high_score_insufficient": high_score_insufficient,
        "note": note,
    }


def score_and_decide(
    rows: list[dict[str, Any]],
    *,
    statistic: str,
    calibration_scores: list[float],
    target_fpr: float,
    alpha_context: float,
    label: str,
) -> list[dict[str, Any]]:
    outputs = []
    for row in rows:
        score = compute_statistic(row, statistic, alpha_context=alpha_context)
        insufficient = bool(row.get("insufficient_evidence", False))
        decision = detect_with_conformal(
            score=score,
            calibration_scores=calibration_scores,
            target_fpr=target_fpr,
            insufficient_evidence=insufficient,
        )
        outputs.append(
            {
                "label": label,
                "id": str(row.get("id", "")),
                "score": decision.score,
                "p_conformal": decision.p_conformal,
                "is_detected": decision.is_detected,
                "is_detected_without_gate": decision.is_detected_without_gate,
                "insufficient_evidence": insufficient,
                "proxy_windows": safe_int(row.get("proxy_windows")),
                "scoreable_contexts": safe_int(row.get("scoreable_contexts")),
            }
        )
    return outputs


def context_scores(row: dict[str, Any]) -> list[float]:
    summaries = row.get("context_summaries")
    if not isinstance(summaries, list):
        return []
    return [
        safe_float(summary.get("context_score"))
        for summary in summaries
        if isinstance(summary, dict)
    ]


def context_p_values(row: dict[str, Any]) -> list[float]:
    return [clamp_p(10.0 ** (-score)) for score in context_scores(row)]


def topk_sum(values: list[float], k: int) -> float:
    return sum(sorted(values, reverse=True)[:k])


def topk_mean(values: list[float], k: int) -> float:
    selected = sorted(values, reverse=True)[:k]
    return sum(selected) / len(selected) if selected else 0.0


def berk_jones(p_values: list[float]) -> float:
    sorted_p = sorted(clamp_p(value) for value in p_values)
    n = len(sorted_p)
    if n == 0:
        return 0.0
    best = 0.0
    for index, p_value in enumerate(sorted_p, start=1):
        q_value = index / n
        if q_value <= p_value:
            continue
        left = q_value * math.log(q_value / p_value)
        right = 0.0 if q_value >= 1.0 else (1.0 - q_value) * math.log((1.0 - q_value) / (1.0 - p_value))
        best = max(best, n * (left + right))
    return best


def higher_criticism(p_values: list[float]) -> float:
    sorted_p = sorted(clamp_p(value) for value in p_values)
    n = len(sorted_p)
    if n == 0:
        return 0.0
    best = 0.0
    for index, p_value in enumerate(sorted_p, start=1):
        q_value = index / n
        if q_value <= p_value:
            continue
        denominator = math.sqrt(max(p_value * (1.0 - p_value), P_EPS))
        best = max(best, math.sqrt(n) * (q_value - p_value) / denominator)
    return best


def recovered_hit_oracle(
    *,
    positives: list[dict[str, Any]],
    proxy_recall_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_sample: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "accepted_count": 0,
            "proxy_recovered_count": 0,
            "proxy_recovered_hit_count": 0,
        }
    )
    for row in proxy_recall_rows:
        sample_id = str(row.get("id", ""))
        per_sample[sample_id]["accepted_count"] += 1
        if row.get("matched_proxy_window_id") is not None:
            per_sample[sample_id]["proxy_recovered_count"] += 1
        if row.get("proxy_in_valid_set") is True:
            per_sample[sample_id]["proxy_recovered_hit_count"] += 1

    positive_ids = [str(row.get("id", "")) for row in positives]
    detected_ids = [
        sample_id
        for sample_id in positive_ids
        if per_sample[sample_id]["proxy_recovered_hit_count"] > 0
    ]
    return {
        "oracle_not_final_code_detector": True,
        "positive_count": len(positive_ids),
        "detected_positive_count": len(detected_ids),
        "oracle_tpr": rate(len(detected_ids), len(positive_ids)),
        "detected_positive_ids": detected_ids,
        "per_sample": {
            sample_id: per_sample[sample_id]
            for sample_id in positive_ids
        },
    }


def oracle_result_row(setting: str, oracle: dict[str, Any], positive_count: int) -> dict[str, Any]:
    detected_ids = list(oracle["detected_positive_ids"])
    return {
        "setting": setting,
        "statistic": "recovered_hit_oracle",
        "formula": STATISTIC_DEFINITIONS["recovered_hit_oracle"]["formula"],
        "uses_final_code_only": False,
        "uses_diagnostic_sidecar": True,
        "calibration_method": "oracle upper bound only",
        "production_readiness": "not a detector claim",
        "risk": STATISTIC_DEFINITIONS["recovered_hit_oracle"]["risk"],
        "target_fpr": None,
        "alpha_context": None,
        "threshold_exclusive": None,
        "threshold_rule": "not applicable",
        "conformal_p_value_rule": "not applicable",
        "positive_count": positive_count,
        "negative_count": 0,
        "calibration_negative_count": 0,
        "detected_positive_count": len(detected_ids),
        "false_positive_count": 0,
        "false_negative_count": positive_count - len(detected_ids),
        "tpr": rate(len(detected_ids), positive_count),
        "observed_fpr": None,
        "tpr_without_gate": rate(len(detected_ids), positive_count),
        "observed_fpr_without_gate": None,
        "auroc": None,
        "positive_insufficient_count": None,
        "negative_insufficient_count": None,
        "high_score_insufficient_count": None,
        "positive_score_quantiles": {},
        "negative_score_quantiles": {},
        "positive_p_value_quantiles": {},
        "negative_p_value_quantiles": {},
        "detected_positive_ids": detected_ids,
        "false_positive_ids": [],
        "false_negative_ids": [],
        "high_score_insufficient": [],
        "note": "Oracle upper bound from E1/E2 sidecars; not a final-code detector.",
    }


def build_loss_reclassification(
    *,
    e1_root: Path,
    statistic_rows: list[dict[str, Any]],
    positives: list[dict[str, Any]],
    target_fpr: float,
) -> dict[str, Any]:
    summary = load_json(e1_root / "analysis/e1_e2_summary.json")
    per_sample = summary.get("per_sample_failure_table", [])
    original_detected_ids = {
        str(row.get("id", ""))
        for row in per_sample
        if row.get("is_watermarked") is True
    }
    sample_score_low_ids = {
        str(row.get("id", ""))
        for row in per_sample
        if row.get("main_loss_reason") == "proxy_recovered_hit_but_sample_score_low"
    }
    original_false_negative_ids = {
        str(row.get("id", ""))
        for row in positives
        if str(row.get("id", "")) not in original_detected_ids
    }
    rows = []
    for result in statistic_rows:
        statistic = result["statistic"]
        if statistic == "recovered_hit_oracle":
            continue
        detected = set(result["detected_positive_ids"])
        rows.append(
            {
                "statistic": statistic,
                "tpr": result["tpr"],
                "observed_fpr": result["observed_fpr"],
                "converted_sample_score_low_count": len(sample_score_low_ids & detected),
                "converted_sample_score_low_ids": sorted(sample_score_low_ids & detected),
                "original_false_negatives_recovered_count": len(original_false_negative_ids & detected),
                "original_false_negatives_recovered_ids": sorted(original_false_negative_ids & detected),
                "high_score_insufficient": result["high_score_insufficient"],
                "tpr_without_gate": result["tpr_without_gate"],
                "observed_fpr_without_gate": result["observed_fpr_without_gate"],
                "fpr_controlled_at_target": result["observed_fpr"] is not None and result["observed_fpr"] <= target_fpr,
            }
        )
    oracle = next(
        (row for row in statistic_rows if row["statistic"] == "recovered_hit_oracle"),
        None,
    )
    return {
        "target_fpr": target_fpr,
        "original_detector_detected_ids": sorted(original_detected_ids),
        "original_detector_detected_count": len(original_detected_ids),
        "sample_score_low_ids": sorted(sample_score_low_ids),
        "sample_score_low_count": len(sample_score_low_ids),
        "original_false_negative_ids": sorted(original_false_negative_ids),
        "original_false_negative_count": len(original_false_negative_ids),
        "oracle": oracle,
        "rows": rows,
    }


def write_run_plan(
    output_root: Path,
    e1_root: Path,
    full164_root: Path,
    target_fpr: float,
    alpha_context: float,
    seed: int,
) -> Path:
    path = output_root / "RUN_PLAN.md"
    lines = [
        "# SAWR Sparse Statistic Conformal Run Plan",
        "",
        "## Scope",
        "- This run is diagnostic-only.",
        "- Actual detector input remains final code only.",
        "- Diagnostic sidecars are used only for analysis, loss attribution, and oracle upper-bound explanation.",
        "- The production detector default statistic, threshold, sample score, and input contract are not modified.",
        "",
        "## New Script",
        "- Add and run `scripts/analyze_sawr_sparse_stat_conformal.py` as a standalone diagnostic script.",
        "- The script reads detector detail JSONL files and E1/E2 sidecars, then writes analysis artifacts under this output directory.",
        "",
        "## Inputs",
        f"- E1/E2 probe root: `{e1_root}`",
        f"- Full164 bestgen root: `{full164_root}`",
        "- 20-sample inputs: positive/negative details, proxy_window_diagnostics, proxy_recall_results, exact_replay_results, e1_e2_summary.",
        "- Full164 inputs: negative_calibration_details, negative_test_details, positive_details, all_positive_details, all_negative_details, report JSON files.",
        "",
        "## Statistics",
        "- `current_mean`, `context_max`, `top2_mean`, `top3_mean`, `top2_sum`, `top3_sum`.",
        "- `count_sig_contexts` with predefined `alpha_context=0.05`.",
        "- `tippett_min_p`, `fisher`, `stouffer`, `cauchy`, `berk_jones`, `higher_criticism`.",
        "- `recovered_hit_oracle` for 20-sample upper bound only; it is not a detector claim.",
        "",
        "## Calibration and Evaluation",
        f"- Target FPR: `{target_fpr}`.",
        f"- Context alpha grid for count statistic: `{alpha_context}`.",
        "- Primary conformal p-value rule: `(1 + count(calibration_stat >= sample_stat)) / (1 + n_calibration)`.",
        "- Detection rule: `p_conformal <= target_fpr` and `insufficient_evidence == false`.",
        f"- 20-sample negative details are split task-disjoint with seed `{seed}` because separate calibration/test details were not persisted.",
        "- Full164 uses the existing 82/82 negative calibration/test details and evaluates both split-positive n=82 and all-positive n=164.",
        "- Repeated split stability uses full164 all_negative_details with seeds `0,1,2,3,4,20260616`.",
        "",
        "## Stop/Go",
        "- If a statistic controls held-out full164 FPR at <=5% and materially improves TPR, recommend an experimental detector mode patch plan only.",
        "- If TPR improves but FPR exceeds 5%, recommend larger negative corpus plus strict conformal calibration; do not change production defaults.",
        "- If 20-sample improves but full164 regresses, recommend full164 proxy sidecar and repeated split diagnostics.",
        "- If all statistics remain weak while E1/E2 recovered hit is high, continue sample aggregation/calibration design; do not revert generation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_input_artifact_audit(analysis_dir: Path, e1_root: Path, full164_root: Path) -> Path:
    path = analysis_dir / "input_artifact_audit.md"
    e1_detection = e1_root / "detection/gamma025"
    full_detection = full164_root / "detection/gamma025"
    e1_candidate = first_jsonl(e1_root / "generation/humaneval_sawr_candidate_text_sidecar_20260616_180646.jsonl")
    e1_proxy = first_jsonl(e1_detection / "proxy_window_diagnostics.jsonl")
    e1_positive = first_jsonl(e1_detection / "positive_details.jsonl")
    full_positive = first_jsonl(full_detection / "positive_details.jsonl")
    full_context_summary = first_nonempty_context_summary(
        [
            full_detection / "positive_details.jsonl",
            full_detection / "all_positive_details.jsonl",
            full_detection / "negative_calibration_details.jsonl",
            full_detection / "negative_test_details.jsonl",
        ]
    )
    full_proxy_files = sorted(full164_root.glob("**/*proxy*")) + sorted(full164_root.glob("**/*sidecar*"))

    required_candidate = {
        "normalized_text",
        "candidate_hash",
        "normalized_text_hash",
    }
    required_proxy = {
        "normalized_text",
        "normalized_text_hash",
        "context_score",
        "context_raw",
        "in_valid_set",
        "min_margin",
    }
    required_detail = {
        "id",
        "score",
        "p_value",
        "is_watermarked",
        "insufficient_evidence",
        "context_summaries",
    }
    context_summary_keys = set(full_context_summary)

    rows = [
        {
            "artifact": "E1/E2 generation candidate sidecar",
            "rows": line_count(e1_root / "generation/humaneval_sawr_candidate_text_sidecar_20260616_180646.jsonl"),
            "required_present": required_candidate <= set(e1_candidate),
            "missing": sorted(required_candidate - set(e1_candidate)),
            "verdict": "sufficient for E1 exact replay",
        },
        {
            "artifact": "E1/E2 final proxy sidecar",
            "rows": line_count(e1_detection / "proxy_window_diagnostics.jsonl"),
            "required_present": required_proxy <= set(e1_proxy),
            "missing": sorted(required_proxy - set(e1_proxy)),
            "verdict": "sufficient for proxy-window diagnostics and oracle attribution",
        },
        {
            "artifact": "E1/E2 positive detector details",
            "rows": line_count(e1_detection / "positive_details.jsonl"),
            "required_present": required_detail <= set(e1_positive),
            "missing": sorted(required_detail - set(e1_positive)),
            "verdict": "sufficient for sample-level statistics",
        },
        {
            "artifact": "Full164 detector details",
            "rows": line_count(full_detection / "positive_details.jsonl"),
            "required_present": required_detail <= set(full_positive),
            "missing": sorted(required_detail - set(full_positive)),
            "verdict": "sufficient for context-summary-level statistics only",
        },
    ]
    lines = [
        "# Input Artifact Audit",
        "",
        markdown_table(rows, ["artifact", "rows", "required_present", "missing", "verdict"]),
        "",
        "## 20-Sample E1/E2 Probe",
        "- Generation candidate sidecar includes normalized candidate text and hashes.",
        "- Final proxy window diagnostics include final-code-only proxy windows, context score/raw, LSH hit, margin, and sample decision fields.",
        "- Positive detector details include sample detection result and context summaries.",
        "- This is enough for diagnostic oracle upper bound and loss reclassification.",
        "",
        "## Full164 Bestgen",
        f"- Proxy/sidecar files found under full164 root: `{len(full_proxy_files)}`.",
        "- Full164 has detector details and calibration/test splits, but no proxy_window_diagnostics.",
        "- Therefore full164 cannot support exact proxy-window-level Cauchy/Berk-Jones claims.",
        "- Full164 can support context-summary-level statistics from `context_summaries[].context_score` and `context_summaries[].context_raw`.",
        f"- First context summary keys observed: `{sorted(context_summary_keys)}`.",
        "",
        "## Constructible Statistics From Existing Details",
        "- current mean baseline: yes, via detector `score` or mean context_score.",
        "- context max: yes.",
        "- top-k context score: yes.",
        "- count of context p-values below alpha: yes, using `p = 10 ** (-context_score)` and predefined alpha=0.05.",
        "- Fisher/Stouffer/Cauchy/Berk-Jones/Higher-Criticism: yes at context-summary level with empirical/conformal calibration only.",
        "",
        "## Strictness Notes",
        "- Full164 is the primary held-out FPR evidence because it has separate negative calibration and negative test details.",
        "- 20-sample strict sample-level split is post-hoc from 164 negative details; its context_score transform came from the original E1/E2 calibration artifact.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_statistic_definitions(analysis_dir: Path) -> Path:
    path = analysis_dir / "statistic_definitions.md"
    rows = [
        {
            "statistic": name,
            "formula": definition["formula"],
            "final_code_only": definition["final_code_only"],
            "uses_diagnostic_sidecar": definition["uses_diagnostic_sidecar"],
            "calibration": definition["calibration_method"],
            "production_candidate": definition["production_candidate"],
            "risk": definition["risk"],
        }
        for name, definition in STATISTIC_DEFINITIONS.items()
    ]
    lines = [
        "# Statistic Definitions",
        "",
        "All non-oracle statistics consume only final-code detector details. Diagnostic sidecars are not detector input.",
        "",
        markdown_table(
            rows,
            [
                "statistic",
                "formula",
                "final_code_only",
                "uses_diagnostic_sidecar",
                "calibration",
                "production_candidate",
                "risk",
            ],
        ),
        "",
        "## Conformal Rule",
        "",
        "`p_conformal = (1 + count(negative_calibration_stat >= sample_stat)) / (1 + n_calibration)`",
        "",
        "Primary detection uses `p_conformal <= target_fpr` plus the existing insufficient-evidence gate. Ungated values are reported only as a diagnostic ablation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_results_markdown(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    intro: list[str],
    baseline_rows: list[dict[str, Any]] | None = None,
) -> None:
    table_rows = [
        {
            "setting": row["setting"],
            "statistic": row["statistic"],
            "final_code_only": row["uses_final_code_only"],
            "sidecar": row["uses_diagnostic_sidecar"],
            "TPR": row["tpr"],
            "FPR": row["observed_fpr"],
            "AUROC": row["auroc"],
            "threshold": row["threshold_exclusive"],
            "pos_insuff": row["positive_insufficient_count"],
            "neg_insuff": row["negative_insufficient_count"],
            "det_pos": row["detected_positive_count"],
            "FP": row["false_positive_count"],
            "readiness": row["production_readiness"],
        }
        for row in rows
    ]
    lines = [f"# {title}", ""]
    lines.extend(f"- {item}" for item in intro)
    lines.append("")
    if baseline_rows:
        lines.extend(
            [
                "## Baseline Comparisons",
                "",
                markdown_table(
                    baseline_rows,
                    ["run", "positive_n", "negative_n", "tpr", "fpr", "auroc"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Statistic Table",
            "",
            markdown_table(
                table_rows,
                [
                    "setting",
                    "statistic",
                    "final_code_only",
                    "sidecar",
                    "TPR",
                    "FPR",
                    "AUROC",
                    "threshold",
                    "pos_insuff",
                    "neg_insuff",
                    "det_pos",
                    "FP",
                    "readiness",
                ],
            ),
            "",
            "## High-Score Insufficient Samples",
            "",
        ]
    )
    for row in rows:
        if row.get("high_score_insufficient"):
            lines.append(
                f"- `{row['setting']}/{row['statistic']}`: "
                f"{json.dumps(row['high_score_insufficient'], ensure_ascii=False)}"
            )
    if not any(row.get("high_score_insufficient") for row in rows):
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_loss_reclassification_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    table_rows = [
        {
            "statistic": row["statistic"],
            "TPR": row["tpr"],
            "FPR": row["observed_fpr"],
            "converted_low": row["converted_sample_score_low_count"],
            "orig_fn_recovered": row["original_false_negatives_recovered_count"],
            "TPR_no_gate": row["tpr_without_gate"],
            "FPR_no_gate": row["observed_fpr_without_gate"],
            "FPR_controlled": row["fpr_controlled_at_target"],
        }
        for row in rows
    ]
    lines = [
        "# 20-Sample Loss Reclassification",
        "",
        f"- Original detector detected positives: {payload['original_detector_detected_count']}/20.",
        f"- Original `proxy_recovered_hit_but_sample_score_low` sample count: {payload['sample_score_low_count']}.",
        f"- Original false negatives: {payload['original_false_negative_count']}.",
        "",
        markdown_table(
            table_rows,
            [
                "statistic",
                "TPR",
                "FPR",
                "converted_low",
                "orig_fn_recovered",
                "TPR_no_gate",
                "FPR_no_gate",
                "FPR_controlled",
            ],
        ),
        "",
        "## Converted Sample-Score-Low IDs",
    ]
    for row in rows:
        if row["converted_sample_score_low_ids"]:
            lines.append(f"- `{row['statistic']}`: {', '.join(row['converted_sample_score_low_ids'])}")
    lines.extend(
        [
            "",
            "## Oracle Upper Bound",
            "",
            json.dumps(payload["oracle"], ensure_ascii=False, indent=2),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stability_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Calibration Stability",
        "",
        f"- Seeds: `{payload['seeds']}`.",
        f"- Positive set: `{payload['positive_set']}`.",
        f"- Negative source: `{payload['negative_source']}`.",
        f"- Target FPR: `{payload['target_fpr']}`.",
        "- Tail warning: only 82 calibration negatives per split; FPR tail estimates are high variance.",
        "",
        markdown_table(
            payload["summary"],
            [
                "statistic",
                "runs",
                "fpr_mean",
                "fpr_std",
                "fpr_min",
                "fpr_max",
                "tpr_mean",
                "tpr_std",
                "tpr_min",
                "tpr_max",
                "threshold_mean",
                "threshold_std",
                "has_fpr_gt_005",
                "has_fpr_gt_010",
                "passes_005_all_splits",
                "meaningfully_above_current_all_positive_tpr",
            ],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stop_go_decision(
    analysis_dir: Path,
    *,
    full164_results: dict[str, Any],
    stability_results: dict[str, Any],
    target_fpr: float,
) -> Path:
    path = analysis_dir / "stop_go_decision.md"
    split_rows = [
        row
        for row in full164_results["results"]
        if row["setting"] == "full164_split_positive"
    ]
    all_rows = [
        row
        for row in full164_results["results"]
        if row["setting"] == "full164_all_positive"
    ]
    improved_and_controlled = [
        row
        for row in split_rows
        if row["observed_fpr"] is not None
        and row["observed_fpr"] <= target_fpr
        and row["tpr"] > 0.219512
    ]
    improved_but_uncontrolled = [
        row
        for row in split_rows
        if row["observed_fpr"] is not None
        and row["observed_fpr"] > target_fpr
        and row["tpr"] > 0.219512
    ]
    stability_pass = [
        row
        for row in stability_results["summary"]
        if row["passes_005_all_splits"] and row["meaningfully_above_current_all_positive_tpr"]
    ]
    if improved_and_controlled and stability_pass:
        decision = "go_experimental_detector_mode_patch_plan"
        reason = "At least one split-positive statistic improves TPR while controlling held-out FPR, and repeated split stability also passes."
    elif improved_and_controlled:
        decision = "conditional_go_experimental_mode_after_stability"
        reason = "Held-out split has a candidate, but repeated split stability is not strong enough for production defaults."
    elif improved_but_uncontrolled:
        decision = "no_go_larger_negative_corpus"
        reason = "Some statistics improve TPR but exceed the 5% held-out FPR target."
    else:
        best_all = max(all_rows, key=lambda row: row["tpr"] if row["observed_fpr"] is not None and row["observed_fpr"] <= target_fpr else -1)
        if best_all["observed_fpr"] is not None and best_all["observed_fpr"] <= target_fpr and best_all["tpr"] > 0.268293:
            decision = "conditional_go_all_positive_only_needs_split_confirmation"
            reason = "All-positive trend has a controlled candidate, but split-positive criterion was not met."
        else:
            decision = "no_go_continue_statistic_design"
            reason = "No full164 split-positive statistic both improves TPR and controls held-out FPR at 5%."

    lines = [
        "# Stop/Go Decision",
        "",
        f"- Decision: `{decision}`.",
        f"- Reason: {reason}",
        "- Production default detector change: no.",
        "- Recommended next step: larger negative corpus plus full164 proxy diagnostic sidecar before any default detector change.",
        "",
        "## Candidates Improving Split TPR and Controlling FPR",
        "",
        markdown_table(
            [
                {
                    "statistic": row["statistic"],
                    "TPR": row["tpr"],
                    "FPR": row["observed_fpr"],
                    "AUROC": row["auroc"],
                    "threshold": row["threshold_exclusive"],
                }
                for row in improved_and_controlled
            ],
            ["statistic", "TPR", "FPR", "AUROC", "threshold"],
        ),
        "",
        "## Stability-Passing Candidates",
        "",
        markdown_table(
            [
                {
                    "statistic": row["statistic"],
                    "fpr_max": row["fpr_max"],
                    "tpr_mean": row["tpr_mean"],
                    "threshold_std": row["threshold_std"],
                }
                for row in stability_pass
            ],
            ["statistic", "fpr_max", "tpr_mean", "threshold_std"],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_final_report(
    output_root: Path,
    *,
    e1_root: Path,
    full164_root: Path,
    e1_results: dict[str, Any],
    full164_results: dict[str, Any],
    stability_results: dict[str, Any],
    target_fpr: float,
) -> Path:
    path = output_root / "SAWR_SPARSE_STAT_CONFORMAL_REPORT.md"
    full_split_rows = [
        row for row in full164_results["results"] if row["setting"] == "full164_split_positive"
    ]
    full_all_rows = [
        row for row in full164_results["results"] if row["setting"] == "full164_all_positive"
    ]
    controlled_split = [
        row
        for row in full_split_rows
        if row["observed_fpr"] is not None and row["observed_fpr"] <= target_fpr
    ]
    improved_controlled_split = [
        row for row in controlled_split if row["tpr"] > 0.219512
    ]
    best_split = max(full_split_rows, key=lambda row: (row["observed_fpr"] <= target_fpr if row["observed_fpr"] is not None else False, row["tpr"]))
    best_all = max(full_all_rows, key=lambda row: (row["observed_fpr"] <= target_fpr if row["observed_fpr"] is not None else False, row["tpr"]))
    e1_rows = [
        row for row in e1_results["results"] if row["statistic"] != "recovered_hit_oracle"
    ]
    e1_controlled = [
        row
        for row in e1_rows
        if row["observed_fpr"] is not None and row["observed_fpr"] <= target_fpr
    ]
    e1_oracle = e1_results["oracle"]
    stability_summary = stability_results["summary"]
    stability_pass = [row for row in stability_summary if row["passes_005_all_splits"]]

    comparison_rows = []
    for row in [*e1_rows, *full_split_rows, *full_all_rows]:
        comparison_rows.append(
            {
                "statistic": row["statistic"],
                "final_code": row["uses_final_code_only"],
                "sidecar": row["uses_diagnostic_sidecar"],
                "setting": row["setting"],
                "TPR": row["tpr"],
                "FPR": row["observed_fpr"],
                "AUROC": row["auroc"],
                "threshold": row["threshold_exclusive"],
                "p_rule": "conformal <= 0.05",
                "pos_insuff": row["positive_insufficient_count"],
                "neg_insuff": row["negative_insufficient_count"],
                "readiness": row["production_readiness"],
            }
        )
    stability_table = [
        {
            "statistic": row["statistic"],
            "FPR_mean": row["fpr_mean"],
            "FPR_std": row["fpr_std"],
            "FPR_max": row["fpr_max"],
            "TPR_mean": row["tpr_mean"],
            "TPR_std": row["tpr_std"],
            "TPR_max": row["tpr_max"],
            "threshold_std": row["threshold_std"],
            "passes_5pct": row["passes_005_all_splits"],
        }
        for row in stability_summary
    ]

    if improved_controlled_split:
        executive_answer = (
            "A full164 split-positive statistic improved TPR while keeping held-out FPR <= 5%, "
            "but this remains diagnostic and should only motivate an experimental detector mode."
        )
    else:
        executive_answer = (
            "No full164 split-positive statistic simultaneously improved over the current split TPR "
            "and controlled held-out FPR at <= 5%."
        )

    lines = [
        "# SAWR Sparse Statistic Conformal Report",
        "",
        "## Executive summary",
        "",
        f"- Diagnostic statistic result: {executive_answer}",
        f"- 20-sample E1/E2 oracle recovered-hit upper bound: {e1_oracle['detected_positive_count']}/{e1_oracle['positive_count']} TPR={format_number(e1_oracle['oracle_tpr'])}; this is not an actual detector.",
        f"- 20-sample controlled diagnostic statistics at FPR<=5%: {', '.join(row['statistic'] for row in e1_controlled) if e1_controlled else 'none'}.",
        f"- Best full164 split row under the target gate heuristic: `{best_split['statistic']}` TPR={format_number(best_split['tpr'])}, FPR={format_number(best_split['observed_fpr'])}, AUROC={format_number(best_split['auroc'])}.",
        f"- Best full164 all-positive row under the target gate heuristic: `{best_all['statistic']}` TPR={format_number(best_all['tpr'])}, FPR={format_number(best_all['observed_fpr'])}, AUROC={format_number(best_all['auroc'])}.",
        f"- Repeated split statistics passing FPR<=5% in every split: {', '.join(row['statistic'] for row in stability_pass) if stability_pass else 'none'}.",
        "- Production detector recommendation: do not change the default detector judgment in this round.",
        "- Next step: larger negative corpus and full164 proxy-window diagnostic sidecar before an experimental detector mode patch.",
        "",
        "## Input evidence recap",
        "",
        f"- Full164 report: `{full164_root / 'FULL164_BESTGEN_REPORT.md'}`.",
        "- Full164 bestgen: 164 final rows, 24020 audit rows, split TPR=0.219512/FPR=0.121951/AUROC=0.611392, all-positive TPR=0.268293/FPR=0.121951/AUROC=0.633998.",
        f"- E1/E2 report: `{e1_root / 'E1_E2_REPLAY_PROXY_REPORT.md'}`.",
        "- E1/E2: exact replay 35/35, proxy recall 34/35, final proxy recovered hit 32/35, main loss reason `proxy_recovered_hit_but_sample_score_low`.",
        "- Therefore this round focuses on sample aggregation/statistics and strict conformal calibration, not generation acceptance.",
        "",
        "## Statistic comparison table",
        "",
        markdown_table(
            comparison_rows,
            [
                "statistic",
                "final_code",
                "sidecar",
                "setting",
                "TPR",
                "FPR",
                "AUROC",
                "threshold",
                "p_rule",
                "pos_insuff",
                "neg_insuff",
                "readiness",
            ],
        ),
        "",
        "## Calibration stability table",
        "",
        markdown_table(
            stability_table,
            [
                "statistic",
                "FPR_mean",
                "FPR_std",
                "FPR_max",
                "TPR_mean",
                "TPR_std",
                "TPR_max",
                "threshold_std",
                "passes_5pct",
            ],
        ),
        "",
        "## Loss interpretation",
        "",
        "- `proxy_recovered_hit_but_sample_score_low` is directly tested in `analysis/loss_reclassification_20sample.md`.",
        "- Remaining false negatives are explained by low sample-level sparse statistic after conformal calibration, insufficient evidence gate, or absent/syntax-invalid proxy windows.",
        "- False positives are listed per statistic in `analysis/statistic_results_full164.json` under `false_positive_ids`; their profile is high context-summary tail score under the same final-code-only details.",
        "- The insufficient gate can block strong sparse hits; ungated TPR/FPR is reported as diagnostic ablation only.",
        "- If sparse statistics improve TPR but fail FPR stability, the limiting factor is calibration tail variance rather than generation-side recovery.",
        "",
        "## Production recommendation",
        "",
        "- Do not change production detector defaults in this round.",
        "- A production patch would require a named experimental detector mode, tests for input-contract cleanliness, deterministic statistic computation, conformal p-value calibration, insufficient gate behavior, and backward-compatible report fields.",
        "- Larger negative corpus is needed because 82 calibration negatives make 5% tail decisions depend on only a few samples.",
        "- Full164 proxy diagnostic sidecar is needed before claiming proxy-window-level sparse statistic behavior on the full run.",
        "",
        "## Stop/go decision",
        "",
        (output_root / "analysis/stop_go_decision.md").read_text(encoding="utf-8"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(output_root: Path, artifacts: list[Path], *, e1_root: Path, full164_root: Path) -> None:
    lines = [
        "# SAWR sparse statistic conformal manifest",
        "",
        "Mode: diagnostic-only",
        f"E1/E2 input root: {e1_root}",
        f"Full164 input root: {full164_root}",
        "",
        "Artifacts:",
    ]
    for path in artifacts:
        status = "present" if path.exists() else "missing"
        lines.append(f"- {status}: {path}")
    (output_root / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_context_cache(
    path: Path,
    groups: list[tuple[str, str, list[dict[str, Any]]]],
    *,
    alpha_context: float,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source, label, rows in groups:
            for row in rows:
                scores = context_scores(row)
                p_values = context_p_values(row)
                payload = {
                    "source": source,
                    "label": label,
                    "id": row.get("id"),
                    "insufficient_evidence": row.get("insufficient_evidence"),
                    "scoreable_contexts": row.get("scoreable_contexts"),
                    "proxy_windows": row.get("proxy_windows"),
                    "context_scores": scores,
                    "context_p_values": p_values,
                    "current_mean": compute_statistic(row, "current_mean", alpha_context=alpha_context),
                    "context_max": compute_statistic(row, "context_max", alpha_context=alpha_context),
                    "count_sig_contexts": compute_statistic(row, "count_sig_contexts", alpha_context=alpha_context),
                }
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def split_by_task(
    rows: list[dict[str, Any]],
    *,
    calibration_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_rows[task_id(str(row.get("id", "")))].append(row)
    task_ids = sorted(task_rows)
    random.Random(seed).shuffle(task_ids)
    calibration_task_count = round(len(task_ids) * calibration_ratio)
    calibration_tasks = set(task_ids[:calibration_task_count])
    test_tasks = set(task_ids[calibration_task_count:])
    calibration_rows = [
        row for task in task_ids if task in calibration_tasks for row in task_rows[task]
    ]
    test_rows = [
        row for task in task_ids if task in test_tasks for row in task_rows[task]
    ]
    manifest = {
        "seed": seed,
        "calibration_ratio": calibration_ratio,
        "total_rows": len(rows),
        "total_task_count": len(task_ids),
        "calibration_rows": len(calibration_rows),
        "test_rows": len(test_rows),
        "calibration_task_count": len(calibration_tasks),
        "test_task_count": len(test_tasks),
        "calibration_tasks": sorted(calibration_tasks),
        "test_tasks": sorted(test_tasks),
    }
    return calibration_rows, test_rows, manifest


def threshold_exclusive(calibration_scores: list[float], target_fpr: float) -> float | None:
    if not calibration_scores:
        return None
    max_count_ge = math.floor(target_fpr * (len(calibration_scores) + 1) - 1.0 + 1e-12)
    if max_count_ge < 0:
        return math.inf
    descending = sorted(calibration_scores, reverse=True)
    if max_count_ge >= len(descending):
        return -math.inf
    return descending[max_count_ge]


def auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.0
    rank_sum = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                rank_sum += 1.0
            elif positive == negative:
                rank_sum += 0.5
    return rank_sum / (len(positive_scores) * len(negative_scores))


def quantile_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": quantile(values, 0.0),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.5),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": quantile(values, 1.0),
    }


def quantile(values: list[float], p_value: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * p_value
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def flatten_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        flattened.append(
            {
                "setting": row["setting"],
                "statistic": row["statistic"],
                "uses_final_code_only": row["uses_final_code_only"],
                "uses_diagnostic_sidecar": row["uses_diagnostic_sidecar"],
                "positive_count": row["positive_count"],
                "negative_count": row["negative_count"],
                "calibration_negative_count": row["calibration_negative_count"],
                "tpr": row["tpr"],
                "observed_fpr": row["observed_fpr"],
                "auroc": row["auroc"],
                "threshold_exclusive": row["threshold_exclusive"],
                "positive_insufficient_count": row["positive_insufficient_count"],
                "negative_insufficient_count": row["negative_insufficient_count"],
                "detected_positive_count": row["detected_positive_count"],
                "false_positive_count": row["false_positive_count"],
                "false_negative_count": row["false_negative_count"],
                "high_score_insufficient_count": row["high_score_insufficient_count"],
                "tpr_without_gate": row["tpr_without_gate"],
                "observed_fpr_without_gate": row["observed_fpr_without_gate"],
                "positive_score_p50": row.get("positive_score_quantiles", {}).get("p50"),
                "positive_score_p95": row.get("positive_score_quantiles", {}).get("p95"),
                "negative_score_p95": row.get("negative_score_quantiles", {}).get("p95"),
                "positive_p_value_p50": row.get("positive_p_value_quantiles", {}).get("p50"),
                "negative_p_value_p50": row.get("negative_p_value_quantiles", {}).get("p50"),
                "detected_positive_ids": json.dumps(row["detected_positive_ids"], ensure_ascii=False),
                "false_positive_ids": json.dumps(row["false_positive_ids"], ensure_ascii=False),
                "false_negative_ids": json.dumps(row["false_negative_ids"], ensure_ascii=False),
                "note": row["note"],
            }
        )
    return flattened


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        rows = []
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return format_number(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    if value is None:
        return "n/a"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    numeric = safe_float(value)
    if math.isinf(numeric):
        return "inf" if numeric > 0 else "-inf"
    return f"{numeric:.6f}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def first_jsonl(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    return rows[0] if rows else {}


def first_nonempty_context_summary(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            summaries = row.get("context_summaries")
            if not isinstance(summaries, list):
                continue
            for summary in summaries:
                if isinstance(summary, dict) and summary:
                    return summary
    return {}


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def task_id(sample_id: str) -> str:
    return sample_id.split("#", 1)[0]


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def clamp_p(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return min(max(value, P_EPS), 1.0 - P_EPS)


def safe_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def safe_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
