#!/usr/bin/env python
"""Offline exploration for SAWR sparse final-code detector statistics.

This script intentionally consumes detector detail artifacts instead of changing
the production detector. Audit files are accepted only for label-side diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.diagnostics.sparse_statistics import mark_sparse_statistics_diagnostic


TARGET_FPR = 0.05
EPS = 1e-12
NORMAL = statistics.NormalDist()


@dataclass(frozen=True)
class StatisticSpec:
    name: str
    formula: str
    inputs: str
    calibration_method: str
    production_note: str


STATISTIC_SPECS: dict[str, StatisticSpec] = {
    "current_score": StatisticSpec(
        name="current_score",
        formula="production detector sample score: mean_i(context_score_i)",
        inputs="row.score",
        calibration_method="existing detector threshold or empirical conformal null",
        production_note="baseline only",
    ),
    "tippett_min_p": StatisticSpec(
        name="tippett_min_p",
        formula="max_i -log10(p_i)",
        inputs="context_score_i = -log10(p_i)",
        calibration_method="sample-level empirical/conformal null",
        production_note="good sparse single-context candidate",
    ),
    "fisher": StatisticSpec(
        name="fisher",
        formula="-2 * sum_i log(p_i)",
        inputs="context p_i reconstructed from context_score_i",
        calibration_method="sample-level empirical/conformal null, not chi-square",
        production_note="sensitive to many weak signals; dependence requires empirical null",
    ),
    "weighted_fisher": StatisticSpec(
        name="weighted_fisher",
        formula="sum_i w_bucket(i) * -2 log(p_i)",
        inputs="context p_i and negative-calibrated bucket weights",
        calibration_method="weights from negative null inverse z-variance; outer empirical null",
        production_note="candidate if weights remain stable across negative splits",
    ),
    "stouffer": StatisticSpec(
        name="stouffer",
        formula="sum_i z_i / sqrt(n), z_i = Phi^-1(1-p_i)",
        inputs="context p_i reconstructed from context_score_i",
        calibration_method="sample-level empirical/conformal null",
        production_note="dense weak-signal baseline",
    ),
    "weighted_stouffer": StatisticSpec(
        name="weighted_stouffer",
        formula="sum_i w_i z_i / sqrt(sum_i w_i^2)",
        inputs="context p_i and negative-calibrated bucket weights",
        calibration_method="weights from negative null inverse z-variance; outer empirical null",
        production_note="candidate only if not dominated by n_contexts/code length",
    ),
    "cauchy": StatisticSpec(
        name="cauchy",
        formula="mean_i tan((0.5-p_i) * pi)",
        inputs="context p_i reconstructed from context_score_i",
        calibration_method="sample-level empirical/conformal null",
        production_note="strong sparse/dependent p-value combination candidate",
    ),
    "weighted_cauchy": StatisticSpec(
        name="weighted_cauchy",
        formula="sum_i w_i tan((0.5-p_i) * pi) / sum_i w_i",
        inputs="context p_i and negative-calibrated bucket weights",
        calibration_method="weights from negative null inverse z-variance; outer empirical null",
        production_note="candidate if weight calibration does not overfit",
    ),
    "higher_criticism": StatisticSpec(
        name="higher_criticism",
        formula="max_i sqrt(n)*(i/n-p_(i))/sqrt(p_(i)*(1-p_(i)))",
        inputs="sorted context p_i in (1/(n+1), 0.5]",
        calibration_method="sample-level empirical/conformal null",
        production_note="many-weak sparse-mixture diagnostic",
    ),
    "berk_jones": StatisticSpec(
        name="berk_jones",
        formula="max_i n * KL(i/n || p_(i)) for p_(i) < i/n",
        inputs="sorted context p_i",
        calibration_method="sample-level empirical/conformal null",
        production_note="sparse-mixture diagnostic; empirical null required",
    ),
    "top1_logp": StatisticSpec(
        name="top1_logp",
        formula="sum of largest 1 values of -log10(p_i)",
        inputs="context_score_i",
        calibration_method="sample-level empirical/conformal null",
        production_note="equivalent to Tippett on current context_score scale",
    ),
    "top2_logp": StatisticSpec(
        name="top2_logp",
        formula="sum of largest 2 values of -log10(p_i)",
        inputs="context_score_i",
        calibration_method="k sweep, sample-level empirical/conformal null",
        production_note="candidate if stable; k is reported, not hand-selected",
    ),
    "top3_logp": StatisticSpec(
        name="top3_logp",
        formula="sum of largest 3 values of -log10(p_i)",
        inputs="context_score_i",
        calibration_method="k sweep, sample-level empirical/conformal null",
        production_note="candidate if stable; k is reported, not hand-selected",
    ),
    "count_sig_contexts": StatisticSpec(
        name="count_sig_contexts",
        formula="count_i[p_i <= alpha_context]",
        inputs="context p_i; alpha_context from negative calibration context p5",
        calibration_method="negative-calibrated alpha_context plus sample-level empirical null",
        production_note="interpretable sparse count; needs context dedupe contract",
    ),
    "empirical_lr_maxlogp": StatisticSpec(
        name="empirical_lr_maxlogp",
        formula="log P_dev_pos(bin(max -log10 p_i)) / P_cal_neg(bin(max -log10 p_i))",
        inputs="positive dev split and negative calibration split",
        calibration_method="held-out positive dev for LR; outer negative empirical null",
        production_note="exploratory only with small positive dev",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline SAWR sparse statistic analysis from detector artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recap = subparsers.add_parser("recap", help="Write Phase 1 recap report.")
    recap.add_argument("--summary", required=True)
    recap.add_argument("--output-dir", required=True)
    recap.add_argument("--old-run-root", required=True)
    recap.add_argument("--sweep-root", required=True)
    recap.add_argument("--audit", action="append", default=[], metavar="NAME:PATH")
    recap.set_defaults(func=cmd_recap)

    compare = subparsers.add_parser("compare", help="Compare sparse statistics.")
    compare.add_argument("--positive", action="append", required=True, metavar="NAME:PATH")
    compare.add_argument("--negative-test", required=True)
    compare.add_argument("--negative-calibration")
    compare.add_argument("--calibration-json")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    compare.add_argument("--positive-dev-ratio", type=float, default=0.5)
    compare.add_argument("--seed", type=int, default=20260616)
    compare.add_argument("--note", default="")
    compare.set_defaults(func=cmd_compare)

    stability = subparsers.add_parser(
        "stability",
        help="Run negative-only repeated split stability from negative details.",
    )
    stability.add_argument("--negative-details", required=True)
    stability.add_argument("--output-dir", required=True)
    stability.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    stability.add_argument("--seeds", default="20260616,20260617,20260618,20260619,20260620,20260621,20260622,20260623,20260624,20260625")
    stability.add_argument("--calibration-ratio", type=float, default=0.5)
    stability.set_defaults(func=cmd_stability)

    validate = subparsers.add_parser(
        "validate",
        help="Validate statistics on task-disjoint negative calibration/test and positive test rows.",
    )
    validate.add_argument("--positive-details", required=True)
    validate.add_argument("--negative-details", required=True)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    validate.add_argument("--seed", type=int, default=20260616)
    validate.add_argument("--calibration-ratio", type=float, default=0.5)
    validate.add_argument("--statistics", default="")
    validate.add_argument("--setting-name", default="full164")
    validate.set_defaults(func=cmd_validate)

    proxy = subparsers.add_parser(
        "proxy-diagnose",
        help="Audit/final-code proxy recall diagnosis without using audit as detector input.",
    )
    proxy.add_argument("--positive-details", required=True)
    proxy.add_argument("--audit", required=True)
    proxy.add_argument("--output-dir", required=True)
    proxy.set_defaults(func=cmd_proxy_diagnose)

    args = parser.parse_args()
    args.func(args)
    return 0


def cmd_recap(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_text = Path(args.summary).read_text(encoding="utf-8")
    old_root = Path(args.old_run_root)
    sweep_root = Path(args.sweep_root)

    old_rows = []
    for variant in (
        "semantic_lsh_default_d4_g075",
        "semantic_lsh_d4_g05",
        "semantic_lsh_d8_g025",
    ):
        report_path = old_root / variant / "reports" / "default.json"
        calibration_path = old_root / variant / "calibration" / "default.json"
        positive_path = old_root / variant / "details" / "positive_default.jsonl"
        negative_path = old_root / variant / "details" / "negative_default.jsonl"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        calibration = load_json(calibration_path)
        positives = load_jsonl(positive_path)
        negatives = load_jsonl(negative_path)
        primary = report.get("primary", {})
        old_rows.append(
            {
                "variant": variant,
                "positive_test": len(positives),
                "negative_test": len(negatives),
                "threshold": calibration.get("threshold_5fpr", 0.0),
                "calibration_negatives": calibration.get("sample_negative_count", 0),
                "context_nulls": calibration.get("context_negative_count", 0),
                "tpr": primary.get("tpr_at_5fpr", primary.get("tpr_at_target_fpr", 0.0)),
                "fpr": primary.get("observed_fpr", 0.0),
                "auroc": primary.get("auroc", 0.0),
                "positive_insufficient": primary.get("positive_insufficient_samples", 0),
                "negative_insufficient": primary.get("negative_insufficient_samples", 0),
            }
        )

    sweep_rows = []
    for group in (
        "t02_p2_g025_baseline",
        "t03_p2_g025",
        "t03_p4_g025",
        "t02_p4_g025",
        "t03_p2_g05",
    ):
        report_path = sweep_root / group / "report.json"
        positive_path = sweep_root / group / "positive_details.jsonl"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        positives = load_jsonl(positive_path)
        gamma_dir = "gamma05" if group.endswith("g05") else "gamma025"
        calibration = load_json(sweep_root / gamma_dir / "calibration.json")
        negatives = load_jsonl(sweep_root / gamma_dir / "negative_test_details.jsonl")
        primary = report.get("primary", {})
        sweep_rows.append(
            {
                "group": group,
                "positive_test": len(positives),
                "negative_test": len(negatives),
                "threshold": calibration.get("threshold_5fpr", 0.0),
                "calibration_negatives": calibration.get("sample_negative_count", 0),
                "context_nulls": calibration.get("context_negative_count", 0),
                "tpr": primary.get("tpr_at_5fpr", primary.get("tpr_at_target_fpr", 0.0)),
                "fpr": primary.get("observed_fpr", 0.0),
                "auroc": primary.get("auroc", 0.0),
                "positive_insufficient": primary.get("positive_insufficient_samples", 0),
                "negative_insufficient": primary.get("negative_insufficient_samples", 0),
                "positive_score_p50": quantile([r.get("score", 0.0) for r in positives], 0.5),
                "negative_score_p95": quantile([r.get("score", 0.0) for r in negatives], 0.95),
            }
        )

    audit_rows = []
    for name, path in parse_named_paths(args.audit):
        audit_rows.append({"name": name, **summarize_audit(Path(path))})

    t03_path = sweep_root / "t03_p4_g025" / "positive_details.jsonl"
    gamma025_neg_path = sweep_root / "gamma025" / "negative_test_details.jsonl"
    t03_rows = load_jsonl(t03_path) if t03_path.exists() else []
    gamma025_neg = load_jsonl(gamma025_neg_path) if gamma025_neg_path.exists() else []

    high_score_insufficient = [
        row
        for row in t03_rows
        if row.get("insufficient_evidence") and row.get("score", 0.0) >= row.get("threshold_5fpr", math.inf)
    ]
    negative_tail = [
        row
        for row in gamma025_neg
        if row.get("score", 0.0) >= row.get("threshold_5fpr", math.inf)
    ]

    report_path = output_dir / "phase1_result_recap.md"
    lines = [
        "# Phase 1 Result Recap",
        "",
        "## 旧 164 主实验结果",
        markdown_table(
            old_rows,
            [
                "variant",
                "positive_test",
                "negative_test",
                "calibration_negatives",
                "context_nulls",
                "threshold",
                "tpr",
                "fpr",
                "auroc",
                "positive_insufficient",
                "negative_insufficient",
            ],
        ),
        "",
        "## 新 20-sample sweep 结果",
        markdown_table(
            sweep_rows,
            [
                "group",
                "positive_test",
                "negative_test",
                "calibration_negatives",
                "context_nulls",
                "threshold",
                "tpr",
                "fpr",
                "auroc",
                "positive_insufficient",
                "negative_insufficient",
                "positive_score_p50",
                "negative_score_p95",
            ],
        ),
        "",
        "## Negative Calibration Stability 初步判断",
        "- sweep20 gamma=0.25/0.5 calibration 均只有 82 个 negative sample，context null 为 223；旧 164 主实验 calibration 只有 66 个 negative sample，context null 为 150。",
        "- 5% FPR 对 66-82 个 calibration sample 等价于只由最尾部约 3-4 个 negative 决定阈值，阈值方差必然偏大；这需要 Phase 3 repeated split 验证。",
        "- sweep20 gamma=0.25 的 negative test observed FPR 为 10/82=0.122，高于目标 5%；gamma=0.5 为 6/82=0.073，仍高于目标。",
        "",
        "## insufficient_evidence 影响",
        f"- t03_p4_g025 中高于阈值但因 insufficient_evidence 被 gate 掉的 positive 样本数：{len(high_score_insufficient)}。",
        "- 典型字段路径：`positive_details.jsonl: score / threshold_5fpr / insufficient_evidence / scoreable_contexts / proxy_windows / context_summaries`。",
        "",
        "## Generation Audit Hit 与 Final-Code Detection 一致性",
        markdown_table(
            audit_rows,
            [
                "name",
                "rows",
                "accepted_rows",
                "accepted_samples",
                "unique_accepted_hashes",
                "compound_accepted_rows",
                "has_candidate_text",
            ],
        ),
        "",
        "## 当前 Detector 失败原因",
        "1. **Sparse hit 被 mean 聚合稀释**：`wfcllm/sawr/detect/pipeline.py:_score_sample` 对 context scores 取均值；在 `positive_details.jsonl: context_summaries[].context_score` 中 single strong context 会被多个 zero/weak context 稀释。",
        "2. **Tail calibration 样本太少**：`calibration.json: sample_negative_count` 只有 66 或 82，5% 阈值由极少数 negative tail 决定；`negative_details.jsonl: score` 的 max/p95 与 positive 分布重叠。",
        "3. **insufficient gate 与 sparse evidence 冲突**：`positive_details.jsonl: insufficient_evidence` 会把 `proxy_windows=1` 但 `score>=threshold_5fpr` 的样本判为未检出。",
        "4. **generation audit hit 不等于 final-code proxy hit**：`audit jsonl: accepted_generation_time_window / candidate_hash / position_id` 与 detector `context_summaries` 没有 exact text/hash 连接，final-code proxy recall 需要单独诊断。",
        "5. **context dependence 未建模**：`context_summaries` 来自同一函数/相邻窗口，Fisher/Stouffer 的 analytic null 不可靠；必须用 empirical/conformal sample null。",
        "",
        "## 关键 Artifact 路径",
        f"- summary: `{args.summary}`",
        f"- old run root: `{args.old_run_root}`",
        f"- sweep root: `{args.sweep_root}`",
        "- fields: `report.json: primary/score_distributions`, `calibration.json: threshold_5fpr/sample_scores/context_nulls`, `details.jsonl: score/p_value/is_watermarked/insufficient_evidence/context_summaries`, `audit.jsonl: accepted_generation_time_window/candidate_hash/reason`。",
        "",
        "## 原始 summary.txt",
        "```text",
        summary_text.strip(),
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_manifest(output_dir, "analyze-results", report_path, "implementation", "Phase 1 SAWR result recap")


def cmd_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_fpr = args.target_fpr
    named_positives = [(name, load_jsonl(Path(path))) for name, path in parse_named_paths(args.positive)]
    negative_test = load_jsonl(Path(args.negative_test))
    negative_calibration = load_jsonl(Path(args.negative_calibration)) if args.negative_calibration else []
    calibration_json = load_json(Path(args.calibration_json)) if args.calibration_json else {}

    calibration_source = "negative_calibration_details"
    calibration_rows = negative_calibration
    leakage = False
    if not calibration_rows:
        calibration_rows = negative_test
        calibration_source = "negative_test_leave_one_out_exploratory"
        leakage = True

    context_alpha = context_alpha_from_rows(calibration_rows)
    weights = build_bucket_weights(calibration_rows)
    lr_models = {
        name: fit_empirical_lr(rows, calibration_rows, args.seed, args.positive_dev_ratio)
        for name, rows in named_positives
    }

    cal_stats = {
        stat_name: [compute_stat(row, stat_name, weights, context_alpha, None) for row in calibration_rows]
        for stat_name in STATISTIC_SPECS
    }

    comparison_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []

    for group_name, positives in named_positives:
        lr_model = lr_models[group_name]
        positive_eval_ids = lr_model["eval_ids"]
        for stat_name, spec in STATISTIC_SPECS.items():
            stat_cal_values = cal_stats[stat_name]
            threshold = empirical_threshold(stat_cal_values, target_fpr)
            positive_scores = []
            negative_scores = []
            positive_rows_for_metric = []
            negative_rows_for_metric = []

            for row in positives:
                if stat_name == "empirical_lr_maxlogp" and row.get("id") not in positive_eval_ids:
                    continue
                score = compute_stat(row, stat_name, weights, context_alpha, lr_model)
                positive_scores.append(score)
                positive_rows_for_metric.append(row)
                per_sample_rows.append(
                    per_sample_payload(group_name, "positive", row, stat_name, score, threshold, stat_cal_values)
                )

            for row in negative_test:
                score = compute_stat(row, stat_name, weights, context_alpha, lr_model)
                negative_scores.append(score)
                negative_rows_for_metric.append(row)
                per_sample_rows.append(
                    per_sample_payload(group_name, "negative_test", row, stat_name, score, threshold, stat_cal_values)
                )

            metrics = metric_summary(
                positive_rows_for_metric,
                negative_rows_for_metric,
                positive_scores,
                negative_scores,
                threshold,
            )
            comparison_rows.append(
                {
                    "group": group_name,
                    "statistic": stat_name,
                    "formula": spec.formula,
                    "inputs": spec.inputs,
                    "calibration_method": spec.calibration_method,
                    "calibration_source": calibration_source,
                    "calibration_leakage": leakage,
                    "threshold": threshold,
                    "target_fpr": target_fpr,
                    "positive_count": len(positive_scores),
                    "negative_count": len(negative_scores),
                    "observed_fpr_gated": metrics["observed_fpr_gated"],
                    "observed_fpr_ungated": metrics["observed_fpr_ungated"],
                    "tpr_gated": metrics["tpr_gated"],
                    "tpr_ungated": metrics["tpr_ungated"],
                    "tpr_sufficient_only": metrics["tpr_sufficient_only"],
                    "auroc": auroc(positive_scores, negative_scores),
                    "auprc": auprc(positive_scores, negative_scores),
                    "positive_score_p50": quantile(positive_scores, 0.5),
                    "positive_score_p75": quantile(positive_scores, 0.75),
                    "negative_score_p95": quantile(negative_scores, 0.95),
                    "negative_score_max": max(negative_scores) if negative_scores else 0.0,
                    "robustness": robustness_summary(positive_rows_for_metric + negative_rows_for_metric, positive_scores + negative_scores),
                    "production_suitability": production_suitability(stat_name, metrics, leakage),
                    "failure_mode": failure_mode(stat_name, metrics, leakage),
                    "production_note": spec.production_note,
                    "context_alpha": context_alpha,
                    "calibration_negative_count": len(calibration_rows),
                    "calibration_json_sample_negative_count": calibration_json.get("sample_negative_count"),
                    "calibration_json_threshold_5fpr": calibration_json.get("threshold_5fpr"),
                    "note": args.note,
                }
            )

    write_json(output_dir / "statistic_comparison.json", comparison_rows)
    write_csv(output_dir / "statistic_comparison.csv", comparison_rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", per_sample_rows)
    write_json(
        output_dir / "calibration_stability.json",
        {
            "mode": "phase2_single_split",
            "calibration_source": calibration_source,
            "calibration_leakage": leakage,
            "negative_calibration_count": len(calibration_rows),
            "negative_test_count": len(negative_test),
            "context_alpha": context_alpha,
            "bucket_weight_count": len(weights),
            "calibration_json": {
                "sample_negative_count": calibration_json.get("sample_negative_count"),
                "context_negative_count": calibration_json.get("context_negative_count"),
                "threshold_5fpr": calibration_json.get("threshold_5fpr"),
            },
        },
    )
    write_compare_report(output_dir / "sparse_detector_report.md", comparison_rows, leakage, args)
    append_manifest(output_dir, "analyze-results", output_dir / "statistic_comparison.json", "implementation", "Sparse statistic comparison JSON")
    append_manifest(output_dir, "analyze-results", output_dir / "sparse_detector_report.md", "implementation", "Sparse statistic comparison report")


def cmd_stability(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.negative_details))
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    all_records: list[dict[str, Any]] = []
    summary: dict[str, list[float]] = defaultdict(list)

    for seed in seeds:
        cal_rows, test_rows = split_by_task(rows, args.calibration_ratio, seed)
        context_alpha = context_alpha_from_rows(cal_rows)
        weights = build_bucket_weights(cal_rows)
        cal_stats = {
            stat_name: [compute_stat(row, stat_name, weights, context_alpha, None) for row in cal_rows]
            for stat_name in STATISTIC_SPECS
            if stat_name != "empirical_lr_maxlogp"
        }
        for stat_name, cal_values in cal_stats.items():
            threshold = empirical_threshold(cal_values, args.target_fpr)
            test_values = [compute_stat(row, stat_name, weights, context_alpha, None) for row in test_rows]
            fp = sum(1 for row, score in zip(test_rows, test_values, strict=True) if (not row.get("insufficient_evidence", False)) and score >= threshold)
            fp_ungated = sum(1 for score in test_values if score >= threshold)
            fpr = fp / len(test_rows) if test_rows else 0.0
            fpr_ungated = fp_ungated / len(test_rows) if test_rows else 0.0
            record = {
                "seed": seed,
                "statistic": stat_name,
                "calibration_count": len(cal_rows),
                "test_count": len(test_rows),
                "threshold": threshold,
                "observed_fpr_gated": fpr,
                "observed_fpr_ungated": fpr_ungated,
                "false_positives_gated": fp,
                "false_positives_ungated": fp_ungated,
                "test_score_p50": quantile(test_values, 0.5),
                "test_score_p95": quantile(test_values, 0.95),
                "test_score_max": max(test_values) if test_values else 0.0,
                "context_alpha": context_alpha,
            }
            all_records.append(record)
            summary[f"{stat_name}.threshold"].append(threshold)
            summary[f"{stat_name}.fpr_gated"].append(fpr)
            summary[f"{stat_name}.fpr_ungated"].append(fpr_ungated)

    summary_rows = []
    for stat_name in sorted({record["statistic"] for record in all_records}):
        thresholds = summary[f"{stat_name}.threshold"]
        fprs = summary[f"{stat_name}.fpr_gated"]
        fprs_ungated = summary[f"{stat_name}.fpr_ungated"]
        summary_rows.append(
            {
                "statistic": stat_name,
                "runs": len(thresholds),
                "threshold_mean": mean(thresholds),
                "threshold_std": stdev(thresholds),
                "fpr_gated_mean": mean(fprs),
                "fpr_gated_std": stdev(fprs),
                "fpr_gated_max": max(fprs) if fprs else 0.0,
                "fpr_ungated_mean": mean(fprs_ungated),
                "fpr_ungated_std": stdev(fprs_ungated),
                "fpr_ungated_max": max(fprs_ungated) if fprs_ungated else 0.0,
            }
        )

    write_json(output_dir / "split_records.json", all_records)
    write_csv(output_dir / "split_records.csv", all_records)
    write_json(output_dir / "summary.json", summary_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_stability_report(output_dir / "calibration_stability_report.md", summary_rows, args, len(rows))
    append_manifest(output_dir, "analyze-results", output_dir / "summary.json", "implementation", "Negative split calibration stability summary")


def cmd_validate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positives = load_jsonl(Path(args.positive_details))
    negatives = load_jsonl(Path(args.negative_details))
    negative_cal, negative_test = split_by_task(negatives, args.calibration_ratio, args.seed)
    test_tasks = {task_id(str(row.get("id", ""))) for row in negative_test}
    calibration_tasks = {task_id(str(row.get("id", ""))) for row in negative_cal}
    positive_test = [
        row for row in positives if task_id(str(row.get("id", ""))) in test_tasks
    ]
    positive_dev = [
        row for row in positives if task_id(str(row.get("id", ""))) in calibration_tasks
    ]
    selected = [
        item.strip()
        for item in args.statistics.split(",")
        if item.strip()
    ] or [
        "current_score",
        "tippett_min_p",
        "fisher",
        "weighted_fisher",
        "stouffer",
        "weighted_stouffer",
        "cauchy",
        "weighted_cauchy",
        "higher_criticism",
        "berk_jones",
        "top1_logp",
        "top2_logp",
        "top3_logp",
        "count_sig_contexts",
    ]
    context_alpha = context_alpha_from_rows(negative_cal)
    weights = build_bucket_weights(negative_cal)
    lr_model = fit_empirical_lr(positive_dev, negative_cal, args.seed, 1.0)

    comparison_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for stat_name in selected:
        if stat_name not in STATISTIC_SPECS:
            raise ValueError(f"unknown statistic: {stat_name}")
        cal_values = [
            compute_stat(row, stat_name, weights, context_alpha, lr_model)
            for row in negative_cal
        ]
        threshold = empirical_threshold(cal_values, args.target_fpr)
        positive_scores = [
            compute_stat(row, stat_name, weights, context_alpha, lr_model)
            for row in positive_test
        ]
        negative_scores = [
            compute_stat(row, stat_name, weights, context_alpha, lr_model)
            for row in negative_test
        ]
        metrics = metric_summary(
            positive_test,
            negative_test,
            positive_scores,
            negative_scores,
            threshold,
        )
        tp = sum(
            1
            for row, score in zip(positive_test, positive_scores, strict=True)
            if (not row.get("insufficient_evidence", False)) and score >= threshold
        )
        fp = sum(
            1
            for row, score in zip(negative_test, negative_scores, strict=True)
            if (not row.get("insufficient_evidence", False)) and score >= threshold
        )
        comparison_rows.append(
            {
                "setting": args.setting_name,
                "statistic": stat_name,
                "formula": STATISTIC_SPECS[stat_name].formula,
                "calibration_method": "task-disjoint negative empirical/conformal split",
                "negative_calibration_count": len(negative_cal),
                "negative_test_count": len(negative_test),
                "positive_dev_count": len(positive_dev) if stat_name == "empirical_lr_maxlogp" else 0,
                "positive_test_count": len(positive_test),
                "threshold": threshold,
                "target_fpr": args.target_fpr,
                "observed_fpr_gated": metrics["observed_fpr_gated"],
                "observed_fpr_ungated": metrics["observed_fpr_ungated"],
                "tpr_gated": metrics["tpr_gated"],
                "tpr_ungated": metrics["tpr_ungated"],
                "tpr_sufficient_only": metrics["tpr_sufficient_only"],
                "auroc": auroc(positive_scores, negative_scores),
                "auprc": auprc(positive_scores, negative_scores),
                "tp_gated": tp,
                "fp_gated": fp,
                "tpr_wilson_95": wilson_ci(tp, len(positive_test)),
                "fpr_wilson_95": wilson_ci(fp, len(negative_test)),
                "positive_score_p50": quantile(positive_scores, 0.5),
                "positive_score_p75": quantile(positive_scores, 0.75),
                "negative_score_p95": quantile(negative_scores, 0.95),
                "negative_score_max": max(negative_scores) if negative_scores else 0.0,
                "robustness": robustness_summary(positive_test + negative_test, positive_scores + negative_scores),
                "production_suitability": production_suitability(stat_name, metrics, False),
                "failure_mode": failure_mode(stat_name, metrics, False),
                "context_alpha": context_alpha,
                "seed": args.seed,
            }
        )
        for label, rows, scores in (
            ("positive_test", positive_test, positive_scores),
            ("negative_test", negative_test, negative_scores),
        ):
            for row, score in zip(rows, scores, strict=True):
                per_sample_rows.append(
                    per_sample_payload(
                        args.setting_name,
                        label,
                        row,
                        stat_name,
                        score,
                        threshold,
                        cal_values,
                    )
                )

    write_json(output_dir / "validation_summary.json", comparison_rows)
    write_csv(output_dir / "validation_summary.csv", comparison_rows)
    write_jsonl(output_dir / "validation_per_sample.jsonl", per_sample_rows)
    write_validation_report(output_dir / "validation_report.md", comparison_rows, args)
    append_manifest(output_dir, "analyze-results", output_dir / "validation_summary.json", "implementation", "Task-disjoint sparse statistic validation")


def cmd_proxy_diagnose(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positives = load_jsonl(Path(args.positive_details))
    audit_rows = load_jsonl(Path(args.audit))
    accepted = [row for row in audit_rows if row.get("event") == "accepted_generation_time_window"]
    accepted_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        accepted_by_id[str(row.get("id", ""))].append(row)

    details_by_id = {str(row.get("id", "")): row for row in positives}
    diagnosis_rows = []
    for sample_id, detail in sorted(details_by_id.items()):
        audit_hits = accepted_by_id.get(sample_id, [])
        context_ids = {ctx.get("context_id") for ctx in detail.get("context_summaries", [])}
        audit_positions = {row.get("position_id") for row in audit_hits}
        recovered_positions = sorted(position for position in audit_positions if position in context_ids)
        lost_positions = sorted(position for position in audit_positions if position not in context_ids)
        diagnosis_rows.append(
            {
                "id": sample_id,
                "detected": bool(detail.get("is_watermarked", False)),
                "insufficient_evidence": bool(detail.get("insufficient_evidence", False)),
                "score": float(detail.get("score", 0.0)),
                "proxy_windows": int(detail.get("proxy_windows", 0)),
                "scoreable_contexts": int(detail.get("scoreable_contexts", 0)),
                "accepted_unique_hashes": len({row.get("candidate_hash") for row in audit_hits if row.get("candidate_hash")}),
                "accepted_rows": len(audit_hits),
                "accepted_positions": len(audit_positions),
                "recovered_positions": len(recovered_positions),
                "lost_positions": len(lost_positions),
                "lost_position_examples": lost_positions[:5],
                "accepted_candidate_types": dict(Counter(str(row.get("candidate_type")) for row in audit_hits)),
                "accepted_min_margin_min": min_margin_summary(audit_hits, "min"),
                "accepted_min_margin_max": min_margin_summary(audit_hits, "max"),
            }
        )

    fields = sorted(set().union(*(row.keys() for row in audit_rows))) if audit_rows else []
    has_text = any(any(key in row for key in ("text", "candidate_text", "normalized_text", "window_text")) for row in audit_rows)
    summary = {
        "positive_details": args.positive_details,
        "audit": args.audit,
        "audit_rows": len(audit_rows),
        "accepted_rows": len(accepted),
        "accepted_samples": len(accepted_by_id),
        "unique_accepted_hashes": len({row.get("candidate_hash") for row in accepted if row.get("candidate_hash")}),
        "audit_fields": fields,
        "has_candidate_text": has_text,
        "exact_replay_status": "blocked_missing_candidate_text" if not has_text else "candidate_text_available",
        "note": "Audit is diagnosis-only and is not used as detector input.",
    }
    write_json(output_dir / "proxy_recall_summary.json", summary)
    write_json(output_dir / "proxy_recall_by_sample.json", diagnosis_rows)
    write_csv(output_dir / "proxy_recall_by_sample.csv", diagnosis_rows)
    write_proxy_report(output_dir / "proxy_recall_report.md", summary, diagnosis_rows)
    append_manifest(output_dir, "analyze-results", output_dir / "proxy_recall_report.md", "implementation", "Audit/proxy recall diagnosis")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    if isinstance(payload, dict):
        payload = mark_sparse_statistics_diagnostic(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar_for_csv(row.get(field)) for field in fields})


def scalar_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def parse_named_paths(items: list[str]) -> list[tuple[str, str]]:
    result = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"expected NAME:PATH, got {item!r}")
        name, path = item.split(":", 1)
        if not name or not path:
            raise ValueError(f"expected NAME:PATH, got {item!r}")
        result.append((name, path))
    return result


def context_p_values(row: dict[str, Any]) -> list[float]:
    p_values = []
    for context in row.get("context_summaries", []):
        score = float(context.get("context_score", 0.0))
        if score <= 0:
            p_values.append(1.0)
        else:
            p_values.append(min(1.0, max(EPS, 10 ** (-score))))
    return p_values


def context_scores(row: dict[str, Any]) -> list[float]:
    return [max(0.0, float(context.get("context_score", 0.0))) for context in row.get("context_summaries", [])]


def bucket_key(context: dict[str, Any]) -> str:
    return "|".join(
        [
            str(context.get("structure_type", "")),
            str(context.get("parent_node_type", "")),
            str(context.get("calibration_bucket_level", "")),
        ]
    )


def context_alpha_from_rows(rows: list[dict[str, Any]]) -> float:
    values = [p for row in rows for p in context_p_values(row)]
    if not values:
        return TARGET_FPR
    return max(EPS, quantile(values, TARGET_FPR))


def build_bucket_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_bucket: dict[str, list[float]] = defaultdict(list)
    global_z: list[float] = []
    for row in rows:
        for context in row.get("context_summaries", []):
            score = max(0.0, float(context.get("context_score", 0.0)))
            p_value = min(1.0 - EPS, max(EPS, 10 ** (-score)))
            z_value = NORMAL.inv_cdf(1.0 - p_value)
            by_bucket[bucket_key(context)].append(z_value)
            global_z.append(z_value)
    global_std = stdev(global_z) or 1.0
    raw_weights = {}
    for key, values in by_bucket.items():
        std = stdev(values) if len(values) >= 5 else global_std
        raw_weights[key] = 1.0 / max(std, EPS)
    if not raw_weights:
        return {}
    average = sum(raw_weights.values()) / len(raw_weights)
    return {key: value / average for key, value in raw_weights.items()}


def row_context_weights(row: dict[str, Any], weights: dict[str, float]) -> list[float]:
    result = []
    for context in row.get("context_summaries", []):
        result.append(weights.get(bucket_key(context), 1.0))
    return result


def compute_stat(
    row: dict[str, Any],
    stat_name: str,
    weights: dict[str, float],
    context_alpha: float,
    lr_model: dict[str, Any] | None,
) -> float:
    p_values = context_p_values(row)
    logp = context_scores(row)
    if stat_name == "current_score":
        return float(row.get("score", 0.0))
    if not p_values:
        return 0.0
    if stat_name == "tippett_min_p":
        return max(logp)
    if stat_name == "fisher":
        return sum(-2.0 * math.log(max(p_value, EPS)) for p_value in p_values)
    if stat_name == "weighted_fisher":
        ws = row_context_weights(row, weights)
        return sum(w * -2.0 * math.log(max(p, EPS)) for w, p in zip(ws, p_values, strict=True))
    if stat_name == "stouffer":
        zs = [NORMAL.inv_cdf(1.0 - min(1.0 - EPS, max(EPS, p))) for p in p_values]
        return sum(zs) / math.sqrt(len(zs))
    if stat_name == "weighted_stouffer":
        ws = row_context_weights(row, weights)
        zs = [NORMAL.inv_cdf(1.0 - min(1.0 - EPS, max(EPS, p))) for p in p_values]
        denom = math.sqrt(sum(w * w for w in ws))
        return sum(w * z for w, z in zip(ws, zs, strict=True)) / denom if denom else 0.0
    if stat_name == "cauchy":
        return sum(cauchy_transform(p) for p in p_values) / len(p_values)
    if stat_name == "weighted_cauchy":
        ws = row_context_weights(row, weights)
        denom = sum(ws)
        return sum(w * cauchy_transform(p) for w, p in zip(ws, p_values, strict=True)) / denom if denom else 0.0
    if stat_name == "higher_criticism":
        return higher_criticism(p_values)
    if stat_name == "berk_jones":
        return berk_jones(p_values)
    if stat_name == "top1_logp":
        return sum(sorted(logp, reverse=True)[:1])
    if stat_name == "top2_logp":
        return sum(sorted(logp, reverse=True)[:2])
    if stat_name == "top3_logp":
        return sum(sorted(logp, reverse=True)[:3])
    if stat_name == "count_sig_contexts":
        return float(sum(1 for p in p_values if p <= context_alpha))
    if stat_name == "empirical_lr_maxlogp":
        return empirical_lr_score(max(logp), lr_model)
    raise KeyError(stat_name)


def cauchy_transform(p_value: float) -> float:
    p = min(1.0 - EPS, max(EPS, p_value))
    return math.tan((0.5 - p) * math.pi)


def higher_criticism(p_values: list[float]) -> float:
    ordered = sorted(min(1.0 - EPS, max(EPS, p)) for p in p_values)
    n = len(ordered)
    best = 0.0
    for index, p_value in enumerate(ordered, start=1):
        if p_value <= 1.0 / (n + 1) or p_value > 0.5:
            continue
        numerator = index / n - p_value
        if numerator <= 0:
            continue
        denominator = math.sqrt(max(EPS, p_value * (1.0 - p_value)))
        best = max(best, math.sqrt(n) * numerator / denominator)
    return best


def berk_jones(p_values: list[float]) -> float:
    ordered = sorted(min(1.0 - EPS, max(EPS, p)) for p in p_values)
    n = len(ordered)
    best = 0.0
    for index, p_value in enumerate(ordered, start=1):
        x = index / n
        if p_value >= x or p_value > 0.5:
            continue
        kl = x * math.log(max(EPS, x / p_value)) + (1 - x) * math.log(max(EPS, (1 - x) / (1 - p_value)))
        best = max(best, n * kl)
    return best


def fit_empirical_lr(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    seed: int,
    dev_ratio: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    positive_ids = sorted({str(row.get("id", "")) for row in positives})
    rng.shuffle(positive_ids)
    dev_count = max(1, round(len(positive_ids) * dev_ratio)) if positive_ids else 0
    dev_ids = set(positive_ids[:dev_count])
    eval_ids = set(positive_ids[dev_count:]) or set(positive_ids)
    pos_dev = [row for row in positives if row.get("id") in dev_ids]
    pos_values = [max(context_scores(row), default=0.0) for row in pos_dev]
    neg_values = [max(context_scores(row), default=0.0) for row in negatives]
    all_values = sorted(pos_values + neg_values)
    if len(all_values) < 4:
        return {"bins": [0.0, 1.0], "log_lr": [0.0], "eval_ids": eval_ids, "dev_ids": dev_ids}
    quantiles = sorted(set(quantile(all_values, q) for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
    if len(quantiles) < 2:
        quantiles = [min(all_values), max(all_values) + EPS]
    bins = quantiles
    pos_counts = [0 for _ in range(len(bins) - 1)]
    neg_counts = [0 for _ in range(len(bins) - 1)]
    for value in pos_values:
        pos_counts[bin_index(value, bins)] += 1
    for value in neg_values:
        neg_counts[bin_index(value, bins)] += 1
    total_pos = sum(pos_counts)
    total_neg = sum(neg_counts)
    bucket_count = len(pos_counts)
    log_lr = []
    for pos_count, neg_count in zip(pos_counts, neg_counts, strict=True):
        pos_density = (pos_count + 1) / (total_pos + bucket_count)
        neg_density = (neg_count + 1) / (total_neg + bucket_count)
        log_lr.append(math.log(pos_density / neg_density))
    return {
        "bins": bins,
        "log_lr": log_lr,
        "eval_ids": eval_ids,
        "dev_ids": dev_ids,
        "positive_dev_count": len(pos_dev),
        "negative_calibration_count": len(negatives),
    }


def empirical_lr_score(value: float, model: dict[str, Any] | None) -> float:
    if not model:
        return 0.0
    bins = model.get("bins", [0.0, 1.0])
    log_lr = model.get("log_lr", [0.0])
    return float(log_lr[bin_index(value, bins)])


def bin_index(value: float, bins: list[float]) -> int:
    if len(bins) < 2:
        return 0
    for index in range(len(bins) - 1):
        left = bins[index]
        right = bins[index + 1]
        if index == len(bins) - 2:
            if left <= value <= right:
                return index
        elif left <= value < right:
            return index
    if value < bins[0]:
        return 0
    return len(bins) - 2


def empirical_threshold(values: list[float], target_fpr: float) -> float:
    if not values:
        return math.inf
    # Strict split-conformal upper-tail threshold. A sample is detected when
    # (1 + #calibration_scores >= score) / (n + 1) <= target_fpr.
    # With score >= threshold in downstream code, set the threshold just above
    # the boundary order statistic to avoid tie-driven FPR inflation.
    ordered_desc = sorted(values, reverse=True)
    max_exceedances = math.floor(target_fpr * (len(ordered_desc) + 1) - 1)
    if max_exceedances < 0:
        return math.nextafter(ordered_desc[0], math.inf)
    boundary_index = min(max_exceedances, len(ordered_desc) - 1)
    return math.nextafter(ordered_desc[boundary_index], math.inf)


def conformal_p_value(score: float, calibration_values: list[float]) -> float:
    if not calibration_values:
        return 1.0
    return (1 + sum(1 for value in calibration_values if value >= score)) / (len(calibration_values) + 1)


def per_sample_payload(
    group: str,
    label: str,
    row: dict[str, Any],
    stat_name: str,
    score: float,
    threshold: float,
    calibration_values: list[float],
) -> dict[str, Any]:
    return {
        "group": group,
        "label": label,
        "id": row.get("id", ""),
        "statistic": stat_name,
        "statistic_score": score,
        "threshold": threshold,
        "conformal_p_value": conformal_p_value(score, calibration_values),
        "detected_gated": (not row.get("insufficient_evidence", False)) and score >= threshold,
        "detected_ungated": score >= threshold,
        "current_score": row.get("score", 0.0),
        "current_p_value": row.get("p_value", 1.0),
        "current_is_watermarked": row.get("is_watermarked", False),
        "insufficient_evidence": row.get("insufficient_evidence", False),
        "scoreable_contexts": row.get("scoreable_contexts", 0),
        "proxy_windows": row.get("proxy_windows", 0),
        "direct_statements": row.get("direct_statements", 0),
        "code_chars": row.get("code_chars", 0),
    }


def metric_summary(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    positive_scores: list[float],
    negative_scores: list[float],
    threshold: float,
) -> dict[str, float]:
    positive_gated = [
        (not row.get("insufficient_evidence", False)) and score >= threshold
        for row, score in zip(positives, positive_scores, strict=True)
    ]
    negative_gated = [
        (not row.get("insufficient_evidence", False)) and score >= threshold
        for row, score in zip(negatives, negative_scores, strict=True)
    ]
    positive_ungated = [score >= threshold for score in positive_scores]
    negative_ungated = [score >= threshold for score in negative_scores]
    sufficient_positive = [
        score >= threshold
        for row, score in zip(positives, positive_scores, strict=True)
        if not row.get("insufficient_evidence", False)
    ]
    return {
        "tpr_gated": rate(sum(positive_gated), len(positive_gated)),
        "tpr_ungated": rate(sum(positive_ungated), len(positive_ungated)),
        "observed_fpr_gated": rate(sum(negative_gated), len(negative_gated)),
        "observed_fpr_ungated": rate(sum(negative_ungated), len(negative_ungated)),
        "tpr_sufficient_only": rate(sum(sufficient_positive), len(sufficient_positive)),
    }


def auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.0
    total = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                total += 1.0
            elif positive == negative:
                total += 0.5
    return total / (len(positive_scores) * len(negative_scores))


def auprc(positive_scores: list[float], negative_scores: list[float]) -> float:
    labels_scores = [(1, score) for score in positive_scores] + [(0, score) for score in negative_scores]
    if not positive_scores or not labels_scores:
        return 0.0
    labels_scores.sort(key=lambda item: item[1], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for label, _score in labels_scores:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / len(positive_scores)
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def robustness_summary(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, float]:
    return {
        "corr_score_proxy_windows": pearson(scores, [float(row.get("proxy_windows", 0)) for row in rows]),
        "corr_score_scoreable_contexts": pearson(scores, [float(row.get("scoreable_contexts", 0)) for row in rows]),
        "corr_score_code_chars": pearson(scores, [float(row.get("code_chars", 0)) for row in rows]),
    }


def production_suitability(stat_name: str, metrics: dict[str, float], leakage: bool) -> str:
    if leakage:
        return "exploratory_only_until_held_out_calibration_details_exist"
    if metrics["observed_fpr_gated"] <= TARGET_FPR and metrics["tpr_gated"] > 0:
        if metrics["tpr_gated"] <= TARGET_FPR:
            return "fpr_controlled_low_power_diagnostic"
        if stat_name in {"tippett_min_p", "cauchy", "top2_logp", "count_sig_contexts"}:
            return "promising_experimental_mode"
        return "possible_diagnostic"
    return "not_production_ready"


def failure_mode(stat_name: str, metrics: dict[str, float], leakage: bool) -> str:
    if leakage:
        return "sample threshold calibrated on evaluation negatives; needs held-out rerun"
    if metrics["observed_fpr_gated"] > TARGET_FPR:
        return "negative null tail exceeds target FPR"
    if metrics["tpr_gated"] <= 0.05:
        return "FPR controlled but positive signal too weak or proxy recall too low"
    if stat_name in {"fisher", "stouffer", "weighted_fisher", "weighted_stouffer"}:
        return "dependent contexts may inflate dense-combination statistic"
    return "no major failure in this split"


def split_by_task(rows: list[dict[str, Any]], calibration_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[task_id(str(row.get("id", "")))].append(row)
    tasks = sorted(by_task)
    random.Random(seed).shuffle(tasks)
    calibration_count = round(len(tasks) * calibration_ratio)
    calibration_tasks = set(tasks[:calibration_count])
    calibration_rows = [row for task in tasks if task in calibration_tasks for row in by_task[task]]
    test_rows = [row for task in tasks if task not in calibration_tasks for row in by_task[task]]
    return calibration_rows, test_rows


def task_id(sample_id: str) -> str:
    return sample_id.split("#", 1)[0]


def summarize_audit(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    accepted = [row for row in rows if row.get("event") == "accepted_generation_time_window"]
    return {
        "rows": len(rows),
        "accepted_rows": len(accepted),
        "accepted_samples": len({row.get("id") for row in accepted}),
        "unique_accepted_hashes": len({row.get("candidate_hash") for row in accepted if row.get("candidate_hash")}),
        "compound_accepted_rows": sum(1 for row in accepted if str(row.get("candidate_type", "")).startswith("compound")),
        "has_candidate_text": any(any(key in row for key in ("text", "candidate_text", "normalized_text", "window_text")) for row in rows),
    }


def min_margin_summary(rows: list[dict[str, Any]], mode: str) -> float:
    margins = []
    for row in rows:
        margin = parse_reason_float(str(row.get("reason", "")), "min_margin")
        if margin is not None:
            margins.append(margin)
    if not margins:
        return 0.0
    return min(margins) if mode == "min" else max(margins)


def parse_reason_float(reason: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=([0-9eE+\-.]+)", reason)
    if not match:
        return None
    return float(match.group(1))


def write_compare_report(path: Path, rows: list[dict[str, Any]], leakage: bool, args: argparse.Namespace) -> None:
    best_rows = sorted(rows, key=lambda row: (row["observed_fpr_gated"] <= args.target_fpr, row["tpr_gated"], row["auroc"]), reverse=True)
    lines = [
        "# Sparse Detector Statistic Comparison",
        "",
        f"- negative calibration source: `{rows[0]['calibration_source'] if rows else 'none'}`",
        f"- calibration leakage/exploratory flag: `{leakage}`",
        f"- target FPR: `{args.target_fpr}`",
        f"- note: {args.note}",
        "",
        "## Top Rows",
        markdown_table(
            best_rows[:20],
            [
                "group",
                "statistic",
                "observed_fpr_gated",
                "tpr_gated",
                "tpr_sufficient_only",
                "auroc",
                "auprc",
                "threshold",
                "production_suitability",
                "failure_mode",
            ],
        ),
        "",
        "## All Statistics",
        markdown_table(
            rows,
            [
                "group",
                "statistic",
                "calibration_source",
                "observed_fpr_gated",
                "observed_fpr_ungated",
                "tpr_gated",
                "tpr_ungated",
                "tpr_sufficient_only",
                "auroc",
                "negative_score_p95",
                "positive_score_p50",
                "production_suitability",
            ],
        ),
        "",
        "## Caveats",
        "- context-level p-values are reconstructed as `10 ** (-context_score)` from detector details; no audit/retry/logits/sampling trace is used as detector input.",
        "- if calibration source is `negative_test_leave_one_out_exploratory`, the comparison is for ranking candidate statistics only and is not a production FPR claim.",
        "- analytic Fisher/Stouffer nulls are intentionally not used because proxy windows are dependent.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stability_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace, negative_count: int) -> None:
    stable = sorted(rows, key=lambda row: (row["fpr_gated_mean"], row["fpr_gated_std"], row["threshold_std"]))
    lines = [
        "# Negative-Only Calibration Stability",
        "",
        f"- negative details rows: `{negative_count}`",
        f"- seeds: `{args.seeds}`",
        f"- calibration ratio: `{args.calibration_ratio}`",
        f"- target FPR: `{args.target_fpr}`",
        "",
        markdown_table(
            stable,
            [
                "statistic",
                "runs",
                "threshold_mean",
                "threshold_std",
                "fpr_gated_mean",
                "fpr_gated_std",
                "fpr_gated_max",
                "fpr_ungated_mean",
                "fpr_ungated_std",
                "fpr_ungated_max",
            ],
        ),
        "",
        "## Interpretation Prompts",
        "- If mean or max FPR is above target, the statistic null tail is unstable under current negative corpus size.",
        "- If threshold_std is large relative to threshold_mean, larger negative calibration or conformal calibration is needed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["observed_fpr_gated"] <= args.target_fpr,
            row["tpr_gated"],
            row["auroc"],
        ),
        reverse=True,
    )
    lines = [
        "# Full Positive Setting Validation",
        "",
        f"- setting: `{args.setting_name}`",
        f"- seed: `{args.seed}`",
        f"- calibration ratio: `{args.calibration_ratio}`",
        f"- target FPR: `{args.target_fpr}`",
        "- Thresholds and weights are fit only on negative calibration tasks.",
        "- Positive rows are evaluated only on tasks paired with held-out negative test tasks.",
        "",
        markdown_table(
            ordered,
            [
                "statistic",
                "negative_calibration_count",
                "negative_test_count",
                "positive_test_count",
                "threshold",
                "observed_fpr_gated",
                "tpr_gated",
                "tpr_sufficient_only",
                "auroc",
                "auprc",
                "tp_gated",
                "fp_gated",
                "tpr_wilson_95",
                "fpr_wilson_95",
                "production_suitability",
                "failure_mode",
            ],
        ),
        "",
        "## Interpretation",
        "- A statistic should not enter production unless `observed_fpr_gated <= target_fpr` on held-out negatives and TPR improves over the current baseline.",
        "- If AUROC improves but TPR at controlled FPR stays low, the statistic may rank positives better but the deployable signal remains too sparse or proxy recall remains too low.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_proxy_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lost_total = sum(int(row["lost_positions"]) for row in rows)
    recovered_total = sum(int(row["recovered_positions"]) for row in rows)
    lines = [
        "# Audit / Proxy Recall Diagnosis",
        "",
        "## Summary",
        markdown_table([summary], ["audit_rows", "accepted_rows", "accepted_samples", "unique_accepted_hashes", "has_candidate_text", "exact_replay_status"]),
        "",
        "## Position-Level Proxy Recall",
        f"- recovered accepted positions: `{recovered_total}`",
        f"- lost accepted positions: `{lost_total}`",
        "- This is a best-effort position-id comparison; details artifacts do not include candidate hashes for final-code proxy windows.",
        "",
        markdown_table(
            rows,
            [
                "id",
                "detected",
                "insufficient_evidence",
                "score",
                "proxy_windows",
                "scoreable_contexts",
                "accepted_unique_hashes",
                "accepted_positions",
                "recovered_positions",
                "lost_positions",
                "accepted_min_margin_min",
                "accepted_min_margin_max",
            ],
        ),
        "",
        "## Exact Replay Status",
        "- Audit rows lack exact candidate text fields (`text`, `candidate_text`, `normalized_text`, `window_text`).",
        "- Therefore exact-window replay is blocked unless another block ledger/final artifact can map `candidate_hash` back to normalized candidate text.",
        "- Audit remains diagnosis-only and is not used as final-code detector input.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_manifest(output_dir: Path, skill: str, file_path: Path, stage: str, description: str) -> None:
    manifest = output_dir / "MANIFEST.md"
    if not manifest.exists():
        manifest.write_text(
            "# Research Output Manifest\n\n"
            "> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.\n\n"
            "| Timestamp | Skill | File | Stage | Description |\n"
            "|-----------|-------|------|-------|-------------|\n",
            encoding="utf-8",
        )
    timestamp = "2026-06-16"
    rel = file_path.name if file_path.parent == output_dir else str(file_path)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(f"| {timestamp} | /{skill} | {rel} | {stage} | {description} |\n")


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_md(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def format_md(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return "`" + json.dumps(value, ensure_ascii=False, sort_keys=True)[:160].replace("|", "\\|") + "`"
    return str(value).replace("|", "\\|")


def quantile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_deltas = [x - x_mean for x in xs]
    y_deltas = [y - y_mean for y in ys]
    x_var = sum(value * value for value in x_deltas)
    y_var = sum(value * value for value in y_deltas)
    if x_var == 0 or y_var == 0:
        return 0.0
    return sum(x * y for x, y in zip(x_deltas, y_deltas, strict=True)) / math.sqrt(x_var * y_var)


def wilson_ci(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    z2 = z * z
    proportion = k / n
    denominator = 1 + z2 / n
    center = proportion + z2 / (2 * n)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) / n) + (z2 / (4 * n * n))
    )
    return [
        max(0.0, (center - margin) / denominator),
        min(1.0, (center + margin) / denominator),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
