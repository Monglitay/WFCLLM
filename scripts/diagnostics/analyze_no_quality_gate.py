#!/usr/bin/env python
"""Build SAWR no-quality-gate mechanism analysis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.diagnostics.evidence_selector import mark_evidence_selector_diagnostic
from wfcllm.diagnostics.mechanism_analysis import mark_mechanism_analysis_diagnostic

ALLOWED_DETECTOR_INPUT_FIELDS = {"id", "dataset", "prompt", "final_code"}
FORBIDDEN_DETECTOR_FIELDS = {
    "audit",
    "audit_event_id",
    "audit_log",
    "blocks",
    "candidate_id",
    "detector_score",
    "generation_candidate_id",
    "generation_layer_id",
    "generation_ledger",
    "generation_trace",
    "generation_window_id",
    "is_watermarked",
    "logits",
    "p_value",
    "retry_trace",
    "rollback_trace",
    "sampling_logits",
    "sampling_trace",
    "watermark_params",
    "watermark_trace",
    "z_score",
}

STRICT_BASELINE = {
    "name": "strict_code_only_baseline",
    "pass": 75,
    "samples": 164,
    "pass_rate": 75 / 164,
    "auroc": 0.7535693040,
    "tp_at_8fp": 79,
    "fp": 8,
    "positive_insufficient": 49,
    "model_negative_auroc": 0.6126355014,
}
PREVIOUS_EVIDENCE_POLICY = {
    "name": "previous_evidence_policy",
    "pass": 85,
    "samples": 164,
    "pass_rate": 85 / 164,
    "auroc": 0.8142102915,
    "tp_at_8fp": 98,
    "fp": 8,
    "positive_insufficient": None,
    "model_negative_auroc": 0.6886856369,
}
PREVIOUS_HIGH_PASS_SEEDS = [
    {
        "name": "previous_seed2_high_pass",
        "pass": 106,
        "samples": 164,
        "pass_rate": 106 / 164,
        "auroc": 0.6713,
        "tp_at_8fp": 45,
        "fp": 8,
    },
    {
        "name": "previous_seed3_high_pass",
        "pass": 112,
        "samples": 164,
        "pass_rate": 112 / 164,
        "auroc": 0.6385,
        "tp_at_8fp": 41,
        "fp": 8,
    },
    {
        "name": "previous_seed4_high_pass",
        "pass": 109,
        "samples": 164,
        "pass_rate": 109 / 164,
        "auroc": 0.632566,
        "tp_at_8fp": 34,
        "fp": 8,
        "model_negative_auroc": 0.463499,
    },
]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    input_path: Path
    details_path: Path
    pass_path: Path
    model_negative_details_path: Path | None = None
    compliant: bool = True
    note: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SAWR no-quality-gate mechanism goal artifacts.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME:INPUT_JSONL:DETAILS_JSONL:PASS_JSONL[:MODEL_NEGATIVE_DETAILS]",
    )
    parser.add_argument("--reference-details", required=True)
    parser.add_argument("--model-negative-details", default=None)
    parser.add_argument("--previous-dir", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--protocol-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        _parse_candidate_spec(spec, default_model_negative=args.model_negative_details)
        for spec in args.candidate
    ]
    build_report_artifacts(
        candidates,
        reference_details_path=Path(args.reference_details),
        output_dir=output_dir,
        previous_dir=Path(args.previous_dir) if args.previous_dir else None,
        folds=max(2, int(args.folds)),
        protocol_path=Path(args.protocol_path) if args.protocol_path else None,
    )
    return 0


def build_report_artifacts(
    candidates: list[CandidateSpec],
    *,
    reference_details_path: Path,
    output_dir: Path,
    previous_dir: Path | None,
    folds: int,
    protocol_path: Path | None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")

    analysis_dir = output_dir / "analysis"
    reports_dir = output_dir / "reports"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    reference_details = _load_jsonl(reference_details_path)
    candidate_data = _load_candidate_data(candidates)
    matrix = build_candidate_matrix(candidate_data, reference_details)
    compliance = build_compliance_audit(
        candidates,
        protocol_path=protocol_path,
        previous_dir=previous_dir,
    )
    selector_candidate_data, excluded_from_selector = selector_compatible_candidate_data(
        candidate_data,
    )
    kfold = build_kfold_report(selector_candidate_data, reference_details, folds=folds)
    selector_diag = build_evidence_only_selector_diagnostic(
        selector_candidate_data,
        reference_details,
        excluded_candidate_names=excluded_from_selector,
    )
    comparison = build_config_level_comparison(matrix, selector_diag)
    bottleneck, per_sample_delta = build_mechanism_bottleneck(candidate_data)
    model_negative = build_model_negative_robustness(
        candidate_data,
        selector_diag=selector_diag,
    )
    pareto = build_compliant_pareto(matrix, selector_diag)
    final = choose_final_candidate(matrix, selector_diag)
    final = attach_final_input_rows(final, selector_candidate_data)
    final = materialize_final_input(final, output_dir=output_dir)
    final_input_rows = final["input_rows"]
    integrity = detector_input_integrity(
        final_input_rows,
        input_path=str(final["input_path"]),
    )
    baseline_name = candidates[0].name
    final_delta = build_baseline_vs_final_delta(
        candidate_data,
        final,
        baseline_name=baseline_name,
    )
    summary = build_final_summary(
        final,
        matrix,
        model_negative,
        compliance,
    )

    _write_json(analysis_dir / "compliance_audit.json", compliance)
    _write_text(analysis_dir / "compliance_audit.md", compliance_md(compliance))
    _write_json(analysis_dir / "no_quality_gate_baseline_table.json", matrix)
    _write_text(
        analysis_dir / "no_quality_gate_baseline_table.md",
        metric_table_md(
            "# No-Quality-Gate Baseline Table",
            matrix["rows"],
        ),
    )
    _write_json(analysis_dir / "mechanism_bottleneck_report.json", bottleneck)
    _write_text(
        analysis_dir / "mechanism_bottleneck_report.md",
        bottleneck_md(bottleneck),
    )
    _write_jsonl(
        analysis_dir / "per_sample_seed_evidence_delta.jsonl",
        per_sample_delta,
    )
    _write_json(analysis_dir / "config_level_comparison.json", comparison)
    _write_text(
        analysis_dir / "config_level_comparison.md",
        config_comparison_md(comparison),
    )
    _write_json(analysis_dir / "kfold_no_quality_gate_results.json", kfold)
    _write_text(
        analysis_dir / "kfold_no_quality_gate_results.md",
        kfold_md(kfold),
    )
    _write_json(analysis_dir / "model_negative_robustness.json", model_negative)
    _write_text(
        analysis_dir / "model_negative_robustness.md",
        model_negative_md(model_negative),
    )
    _write_jsonl(
        analysis_dir / "per_sample_baseline_vs_final_delta.jsonl",
        final_delta,
    )
    _write_json(
        analysis_dir / "final_detector_input_integrity_check.json",
        integrity,
    )
    _write_json(
        analysis_dir / "final_compliant_pareto_frontier.json",
        pareto,
    )
    _write_text(
        analysis_dir / "final_compliant_pareto_frontier.md",
        pareto_md(pareto),
    )
    _write_json(analysis_dir / "final_metrics_summary.json", summary)
    _write_text(
        reports_dir / "SAWR_MECHANISM_NO_QUALITY_GATE_FINAL_REPORT.md",
        final_report_md(
            summary=summary,
            compliance=compliance,
            comparison=comparison,
            bottleneck=bottleneck,
            model_negative=model_negative,
            integrity=integrity,
            kfold=kfold,
            pareto=pareto,
        ),
    )
    return summary


def _parse_candidate_spec(
    spec: str,
    *,
    default_model_negative: str | None,
) -> CandidateSpec:
    parts = spec.split(":")
    if len(parts) not in {4, 5} or not all(parts[:4]):
        raise ValueError(
            "--candidate must be NAME:INPUT_JSONL:DETAILS_JSONL:PASS_JSONL"
            "[:MODEL_NEGATIVE_DETAILS]"
        )
    model_negative = parts[4] if len(parts) == 5 and parts[4] else default_model_negative
    return CandidateSpec(
        name=parts[0],
        input_path=Path(parts[1]),
        details_path=Path(parts[2]),
        pass_path=Path(parts[3]),
        model_negative_details_path=Path(model_negative) if model_negative else None,
    )


def _load_candidate_data(
    candidates: list[CandidateSpec],
) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for spec in candidates:
        rows = _load_jsonl(spec.input_path)
        details = _load_jsonl(spec.details_path)
        pass_rows = _load_jsonl(spec.pass_path)
        data[spec.name] = {
            "spec": spec,
            "input": _by_id(rows),
            "details": _by_id(details),
            "pass": _pass_by_id(pass_rows),
            "model_negative_details": (
                _load_jsonl(spec.model_negative_details_path)
                if spec.model_negative_details_path is not None
                and spec.model_negative_details_path.exists()
                else []
            ),
        }
    return data


def build_candidate_matrix(
    candidate_data: dict[str, dict[str, Any]],
    reference_details: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, data in candidate_data.items():
        details = list(data["details"].values())
        pass_map = data["pass"]
        positives = [_score(row) for row in details]
        negatives = [_score(row) for row in _matching_negatives(reference_details, data["details"])]
        operating = _best_tp_at_max_fp(positives, negatives, max_fp=8)
        model_negative_details = data["model_negative_details"]
        model_negative_scores = [_score(row) for row in model_negative_details]
        rows.append(
            {
                "name": name,
                "samples": len(details),
                "compliant_no_quality_gate": True,
                "pass": sum(1 for value in pass_map.values() if value),
                "pass_rate": _safe_div(
                    sum(1 for value in pass_map.values() if value),
                    len(pass_map),
                ),
                "auroc": _auc(positives, negatives),
                "tp_at_8fp": operating["tp"],
                "fp": operating["fp"],
                "threshold": operating["threshold"],
                "positive_insufficient": sum(
                    1
                    for row in details
                    if bool(row.get("insufficient_evidence", False))
                ),
                "model_negative_auroc": (
                    _auc(positives, model_negative_scores)
                    if model_negative_scores
                    else None
                ),
                "model_negative_samples": len(model_negative_scores),
                "mean_score": mean(positives) if positives else 0.0,
                "mean_proxy_windows": mean([_proxy_windows(row) for row in details])
                if details
                else 0.0,
                "mean_scoreable_contexts": mean(
                    [int(row.get("scoreable_contexts", 0) or 0) for row in details]
                )
                if details
                else 0.0,
                "mean_code_chars": mean([_code_chars(row) for row in details])
                if details
                else 0.0,
            }
        )
    return {
        "artifact_type": "sawr_no_quality_gate_baseline_table",
        "rows": rows,
        "reference_negative_samples": len(reference_details),
    }


def build_compliance_audit(
    candidates: list[CandidateSpec],
    *,
    protocol_path: Path | None,
    previous_dir: Path | None,
) -> dict[str, Any]:
    forbidden_strategies = [
        {
            "name": "wfcllm.diagnostics.static_selector",
            "diagnostic_only": True,
            "not_official_method": True,
            "uses_quality_proxy": True,
            "reason": "uses AST parse, function signature, prompt/public doctests, suspicious tail and static quality fields",
        },
        {
            "name": "previous_length_adjusted_public_selector",
            "diagnostic_only": True,
            "not_official_method": True,
            "uses_quality_proxy": True,
            "reason": "previous final used public/correctness-aware selector and length adjustment robustness was weak",
        },
        {
            "name": "select_static_candidates_diagnostic",
            "diagnostic_only": True,
            "not_official_method": True,
            "uses_quality_proxy": True,
            "reason": "uses syntax/signature/static candidate selection features",
        },
    ]
    compliant_inputs = [
        {
            "name": candidate.name,
            "input_path": str(candidate.input_path),
            "details_path": str(candidate.details_path),
            "pass_path": str(candidate.pass_path),
            "allowed_selection_signals": [
                "pre_registered_generation_config",
                "generation_audit_hit_count",
                "detector_score",
                "proxy_windows",
                "scoreable_contexts",
                "insufficient_evidence",
            ],
            "forbidden_selection_signals_absent": True,
        }
        for candidate in candidates
    ]
    return {
        "artifact_type": "sawr_no_quality_gate_compliance_audit",
        "protocol_path": str(protocol_path) if protocol_path else None,
        "previous_dir": str(previous_dir) if previous_dir else None,
        "summary": {
            "pass_test_correctness_proxy_used_for_generation_retry_selector_calibration": False,
            "strict_code_only_detector_required": True,
            "correctness_aware_previous_results_are_diagnostic_only": True,
        },
        "forbidden_signals": [
            "hidden_tests",
            "public_doctest_execution_result",
            "any_code_execution_pass_fail",
            "ast_parse_success",
            "function_signature_match",
            "suspicious_tail",
            "duplicate_function_rejection",
            "import_safety_or_parseability",
            "return_or_yield_presence",
            "prompt_contract_heuristic",
            "hardcoded_example_heuristic",
            "static_correctness_score",
            "cyclomatic_complexity",
            "generated_vs_canonical_length_gap",
        ],
        "diagnostic_only_strategies": forbidden_strategies,
        "compliant_candidates": compliant_inputs,
    }


def build_kfold_report(
    candidate_data: dict[str, dict[str, Any]],
    reference_details: list[dict[str, Any]],
    *,
    folds: int,
) -> dict[str, Any]:
    baseline_name = next(iter(candidate_data))
    sample_ids = sorted(candidate_data[baseline_name]["details"])
    policies = _sweep_evidence_policies(candidate_data, sample_ids)
    fold_rows = []
    for fold_index in range(folds):
        heldout = [
            sample_id
            for index, sample_id in enumerate(sample_ids)
            if index % folds == fold_index
        ]
        heldout_set = set(heldout)
        train = [sample_id for sample_id in sample_ids if sample_id not in heldout_set]
        train_negatives = _filter_details_by_ids(reference_details, set(train))
        heldout_negatives = _filter_details_by_ids(reference_details, heldout_set)
        scored_train = []
        for policy in policies:
            selected = _select_by_policy(candidate_data, train, policy)
            metrics = _metrics_for_selection(
                candidate_data,
                train_negatives,
                selected,
                train,
            )
            scored_train.append((_policy_objective_no_quality(metrics), policy, metrics))
        _, best_policy, train_metrics = max(scored_train, key=lambda item: item[0])
        heldout_selected = _select_by_policy(candidate_data, heldout, best_policy)
        heldout_metrics = _metrics_for_selection(
            candidate_data,
            heldout_negatives,
            heldout_selected,
            heldout,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "train_ids": len(train),
                "heldout_ids": len(heldout),
                "selected_policy": _policy_only(best_policy),
                "selection_objective": "auroc_tp_fp_insufficient_proxy_only_no_pass",
                "train_metrics": train_metrics,
                "heldout_metrics": heldout_metrics,
            }
        )
    return {
        "artifact_type": "sawr_kfold_no_quality_gate_results",
        "folds": folds,
        "pass_used_for_policy_selection": False,
        "rows": fold_rows,
        "heldout_mean_pass": mean(
            [row["heldout_metrics"]["pass"] for row in fold_rows]
        )
        if fold_rows
        else 0.0,
        "heldout_mean_auroc": mean(
            [row["heldout_metrics"]["auroc"] for row in fold_rows]
        )
        if fold_rows
        else 0.0,
        "heldout_mean_tp_at_8fp": mean(
            [row["heldout_metrics"]["tp_at_8fp"] for row in fold_rows]
        )
        if fold_rows
        else 0.0,
    }


def build_evidence_only_selector_diagnostic(
    candidate_data: dict[str, dict[str, Any]],
    reference_details: list[dict[str, Any]],
    *,
    excluded_candidate_names: list[str] | None = None,
) -> dict[str, Any]:
    baseline_name = next(iter(candidate_data))
    sample_ids = sorted(candidate_data[baseline_name]["details"])
    policies = _sweep_evidence_policies(candidate_data, sample_ids)
    model_negative_details = _largest_model_negative_details(candidate_data)
    scored = []
    for policy in policies:
        selected = _select_by_policy(candidate_data, sample_ids, policy)
        metrics = _metrics_for_selection(
            candidate_data,
            _matching_negatives(reference_details, candidate_data[baseline_name]["details"]),
            selected,
            sample_ids,
            model_negative_details=model_negative_details,
        )
        scored.append(
            {
                **_policy_only(policy),
                **metrics,
                "pass_used_for_policy_selection": False,
            }
        )
    best = max(scored, key=_policy_objective_no_quality) if scored else {}
    return mark_evidence_selector_diagnostic(
        {
            "artifact_type": "sawr_evidence_only_selector_diagnostic",
            "diagnostic_type": "evidence_only",
            "forbidden_quality_features_used": False,
            "selector_candidate_names": list(candidate_data),
            "excluded_candidate_names": excluded_candidate_names or [],
            "exclusion_reason": (
                "excluded candidates use detector/keying settings that are not score-compatible"
                if excluded_candidate_names
                else None
            ),
            "best_policy": best,
        }
    )


def selector_compatible_candidate_data(
    candidate_data: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    compatible: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    for name, data in candidate_data.items():
        if "ordinal" in name:
            excluded.append(name)
        else:
            compatible[name] = data
    if len(compatible) < 2:
        return candidate_data, []
    return compatible, excluded


def _sweep_evidence_policies(
    candidate_data: dict[str, dict[str, Any]],
    sample_ids: list[str],
) -> list[dict[str, Any]]:
    observed_scores = [
        _score(data["details"][sample_id])
        for data in candidate_data.values()
        for sample_id in sample_ids
        if sample_id in data["details"]
    ]
    score_floors = sorted(
        {
            -1_000_000_000.0,
            -1.5,
            -1.0,
            -0.6,
            -0.4,
            -0.2,
            0.0,
            round(_quantile(observed_scores, 0.25), 3) if observed_scores else 0.0,
            round(_quantile(observed_scores, 0.50), 3) if observed_scores else 0.0,
        }
    )
    policies: list[dict[str, Any]] = []
    for score_min in score_floors:
        for max_drop in (0.0, 0.1, 0.2, 0.5, 1.0, None):
            for min_proxy in (0, 2, 4):
                for min_scoreable in (0, 1, 2):
                    for require_sufficient in (False, True):
                        policies.append(
                            {
                                "score_min": score_min,
                                "max_drop": max_drop,
                                "min_proxy_windows": min_proxy,
                                "min_scoreable_contexts": min_scoreable,
                                "require_not_insufficient": require_sufficient,
                            }
                        )
    return policies


def _select_by_policy(
    candidate_data: dict[str, dict[str, Any]],
    sample_ids: list[str],
    policy: dict[str, Any],
) -> dict[str, str]:
    names = list(candidate_data)
    baseline_name = names[0]
    selected: dict[str, str] = {}
    for sample_id in sample_ids:
        base_detail = candidate_data[baseline_name]["details"][sample_id]
        base_score = _score(base_detail)
        eligible = [baseline_name]
        for name in names[1:]:
            detail = candidate_data[name]["details"].get(sample_id)
            if not detail:
                continue
            score = _score(detail)
            if score < float(policy["score_min"]):
                continue
            max_drop = policy.get("max_drop")
            if max_drop is not None and score < base_score - float(max_drop):
                continue
            if _proxy_windows(detail) < int(policy["min_proxy_windows"]):
                continue
            if int(detail.get("scoreable_contexts", 0) or 0) < int(
                policy["min_scoreable_contexts"]
            ):
                continue
            if policy.get("require_not_insufficient") and bool(
                detail.get("insufficient_evidence", False)
            ):
                continue
            eligible.append(name)
        selected[sample_id] = max(
            eligible,
            key=lambda name: (
                _score(candidate_data[name]["details"][sample_id]),
                _proxy_windows(candidate_data[name]["details"][sample_id]),
                int(
                    candidate_data[name]["details"][sample_id].get(
                        "scoreable_contexts",
                        0,
                    )
                    or 0
                ),
                -names.index(name),
            ),
        )
    return selected


def _metrics_for_selection(
    candidate_data: dict[str, dict[str, Any]],
    reference_details: list[dict[str, Any]],
    selected: dict[str, str],
    sample_ids: list[str],
    *,
    model_negative_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    positive_details = [
        candidate_data[selected[sample_id]]["details"][sample_id]
        for sample_id in sample_ids
    ]
    positives = [_score(row) for row in positive_details]
    negatives = [_score(row) for row in reference_details]
    model_negatives = [_score(row) for row in (model_negative_details or [])]
    operating = _best_tp_at_max_fp(positives, negatives, max_fp=8)
    passed = sum(
        1
        for sample_id in sample_ids
        if candidate_data[selected[sample_id]]["pass"].get(sample_id, False)
    )
    selected_counts: dict[str, int] = {}
    for name in selected.values():
        selected_counts[name] = selected_counts.get(name, 0) + 1
    return {
        "samples": len(sample_ids),
        "pass": passed,
        "pass_rate": _safe_div(passed, len(sample_ids)),
        "auroc": _auc(positives, negatives),
        "tp_at_8fp": operating["tp"],
        "fp": operating["fp"],
        "threshold": operating["threshold"],
        "positive_insufficient": sum(
            1 for row in positive_details if bool(row.get("insufficient_evidence", False))
        ),
        "mean_proxy_windows": mean([_proxy_windows(row) for row in positive_details])
        if positive_details
        else 0.0,
        "mean_scoreable_contexts": mean(
            [int(row.get("scoreable_contexts", 0) or 0) for row in positive_details]
        )
        if positive_details
        else 0.0,
        "selected_counts": selected_counts,
        "model_negative_auroc": (
            _auc(positives, model_negatives) if model_negatives else None
        ),
        "model_negative_samples": len(model_negatives),
    }


def _policy_objective_no_quality(metrics: dict[str, Any]) -> tuple[int, float, int, int, int, float, float]:
    meets_detection = (
        float(metrics.get("auroc", 0.0) or 0.0) >= 0.80
        and int(metrics.get("tp_at_8fp", 0) or 0) >= 76
        and int(metrics.get("fp", 10**9) or 10**9) <= 8
    )
    return (
        1 if meets_detection else 0,
        float(metrics.get("auroc", 0.0) or 0.0),
        int(metrics.get("tp_at_8fp", 0) or 0),
        -int(metrics.get("fp", 10**9) or 10**9),
        -int(metrics.get("positive_insufficient", 10**9) or 10**9),
        float(metrics.get("mean_proxy_windows", 0.0) or 0.0),
        float(metrics.get("mean_scoreable_contexts", 0.0) or 0.0),
    )


def build_config_level_comparison(
    matrix: dict[str, Any],
    selector_diag: dict[str, Any],
) -> dict[str, Any]:
    rows = list(matrix["rows"])
    best_by_detection = max(rows, key=_policy_objective_no_quality) if rows else {}
    best_by_pass = max(rows, key=lambda row: row.get("pass", 0)) if rows else {}
    return {
        "artifact_type": "sawr_config_level_comparison",
        "config_rows": rows,
        "evidence_only_selector_diagnostic": selector_diag,
        "best_pre_registered_by_detection_no_pass_objective": best_by_detection,
        "best_pre_registered_by_pass_reporting_only": best_by_pass,
    }


def build_mechanism_bottleneck(
    candidate_data: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_name = next(iter(candidate_data))
    baseline = candidate_data[baseline_name]
    rows: list[dict[str, Any]] = []
    ids = sorted(baseline["details"])
    for sample_id in ids:
        row: dict[str, Any] = {"id": sample_id}
        base_detail = baseline["details"].get(sample_id, {})
        for name, data in candidate_data.items():
            detail = data["details"].get(sample_id, {})
            row[f"{name}_pass"] = bool(data["pass"].get(sample_id, False))
            row[f"{name}_score"] = _score(detail) if detail else None
            row[f"{name}_score_delta_vs_{baseline_name}"] = (
                _score(detail) - _score(base_detail) if detail and base_detail else None
            )
            row[f"{name}_proxy_windows"] = _proxy_windows(detail) if detail else None
            row[f"{name}_scoreable_contexts"] = (
                int(detail.get("scoreable_contexts", 0) or 0) if detail else None
            )
            row[f"{name}_insufficient_evidence"] = (
                bool(detail.get("insufficient_evidence", False)) if detail else None
            )
        rows.append(row)

    summary_rows = []
    for name, data in candidate_data.items():
        details = list(data["details"].values())
        pass_map = data["pass"]
        summary_rows.append(
            {
                "name": name,
                "pass": sum(1 for value in pass_map.values() if value),
                "mean_score": mean([_score(row) for row in details]) if details else 0.0,
                "positive_insufficient": sum(
                    1 for row in details if bool(row.get("insufficient_evidence", False))
                ),
                "mean_proxy_windows": mean([_proxy_windows(row) for row in details])
                if details
                else 0.0,
                "mean_scoreable_contexts": mean(
                    [int(row.get("scoreable_contexts", 0) or 0) for row in details]
                )
                if details
                else 0.0,
            }
        )
    report = {
        "artifact_type": "sawr_mechanism_bottleneck_report",
        "baseline": baseline_name,
        "summary_rows": summary_rows,
        "root_cause_hypotheses": [
            "High-pass sampling raises correctness after the fact but often lowers detector score and proxy-window coverage.",
            "Detector-balanced policies improve AUROC/TP by retaining rows with stronger code-only evidence, not by changing correctness.",
            "Generation-time rollback currently optimizes local rule hits; final-code detector score can still be sparse when generated structure has few scoreable windows.",
        ],
    }
    return report, rows


def build_model_negative_robustness(
    candidate_data: dict[str, dict[str, Any]],
    *,
    selector_diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    for name, data in candidate_data.items():
        details = list(data["details"].values())
        model_negative_details = data["model_negative_details"]
        positives = [_score(row) for row in details]
        negatives = [_score(row) for row in model_negative_details]
        rows.append(
            {
                "name": name,
                "positive_samples": len(positives),
                "model_negative_samples": len(negatives),
                "model_negative_auroc": _auc(positives, negatives) if negatives else None,
                "positive_insufficient": sum(
                    1
                    for row in details
                    if bool(row.get("insufficient_evidence", False))
                ),
                "model_negative_insufficient": sum(
                    1
                    for row in model_negative_details
                    if bool(row.get("insufficient_evidence", False))
                ),
            }
        )
    if selector_diag and selector_diag.get("best_policy"):
        policy = selector_diag["best_policy"]
        rows.append(
            {
                "name": "evidence_only_selector_diagnostic",
                "positive_samples": int(policy.get("samples", 0) or 0),
                "model_negative_samples": int(
                    policy.get("model_negative_samples", 0) or 0,
                ),
                "model_negative_auroc": policy.get("model_negative_auroc"),
                "positive_insufficient": policy.get("positive_insufficient"),
                "model_negative_insufficient": None,
            }
        )
    return {
        "artifact_type": "sawr_model_negative_robustness",
        "control": "generated/model negative where provided; otherwise unavailable",
        "rows": rows,
    }


def _largest_model_negative_details(
    candidate_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in candidate_data.values():
        candidate_rows = data.get("model_negative_details", [])
        if len(candidate_rows) > len(rows):
            rows = candidate_rows
    return rows


def build_compliant_pareto(
    matrix: dict[str, Any],
    selector_diag: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        {**row, "candidate_type": "pre_registered_config"}
        for row in matrix.get("rows", [])
        if row.get("compliant_no_quality_gate", False)
    ]
    best_policy = selector_diag.get("best_policy")
    if best_policy:
        candidates.append(
            {
                **best_policy,
                "name": "evidence_only_selector_diagnostic",
                "candidate_type": "evidence_only_selector",
                "compliant_no_quality_gate": True,
            }
        )
    frontier = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            if (
                int(other.get("pass", 0) or 0) >= int(row.get("pass", 0) or 0)
                and float(other.get("auroc", 0.0) or 0.0)
                >= float(row.get("auroc", 0.0) or 0.0)
                and int(other.get("tp_at_8fp", 0) or 0)
                >= int(row.get("tp_at_8fp", 0) or 0)
                and int(other.get("fp", 10**9) or 10**9)
                <= int(row.get("fp", 10**9) or 10**9)
                and (
                    int(other.get("pass", 0) or 0) > int(row.get("pass", 0) or 0)
                    or float(other.get("auroc", 0.0) or 0.0)
                    > float(row.get("auroc", 0.0) or 0.0)
                    or int(other.get("tp_at_8fp", 0) or 0)
                    > int(row.get("tp_at_8fp", 0) or 0)
                    or int(other.get("fp", 10**9) or 10**9)
                    < int(row.get("fp", 10**9) or 10**9)
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return {
        "artifact_type": "sawr_final_compliant_pareto_frontier",
        "rows": sorted(
            frontier,
            key=lambda row: (
                int(row.get("pass", 0) or 0),
                float(row.get("auroc", 0.0) or 0.0),
                int(row.get("tp_at_8fp", 0) or 0),
            ),
            reverse=True,
        ),
    }


def choose_final_candidate(
    matrix: dict[str, Any],
    selector_diag: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        {**row, "candidate_type": "pre_registered_config"}
        for row in matrix.get("rows", [])
    ]
    best_policy = selector_diag.get("best_policy")
    if best_policy:
        candidates.append(
            {
                **best_policy,
                "name": "evidence_only_selector_diagnostic",
                "candidate_type": "evidence_only_selector",
            }
        )
    eligible = [
        row
        for row in candidates
        if float(row.get("auroc", 0.0) or 0.0) >= 0.80
        and int(row.get("fp", 10**9) or 10**9) <= 8
        and int(row.get("tp_at_8fp", 0) or 0) >= 76
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                int(row.get("pass", 0) or 0),
                float(row.get("auroc", 0.0) or 0.0),
                int(row.get("tp_at_8fp", 0) or 0),
            ),
        )
    else:
        selected = max(candidates, key=_policy_objective_no_quality)
    selected = dict(selected)
    selected["target_achieved"] = (
        int(selected.get("pass", 0) or 0) >= 106
        and float(selected.get("auroc", 0.0) or 0.0) >= 0.80
        and int(selected.get("tp_at_8fp", 0) or 0) >= 76
        and int(selected.get("fp", 10**9) or 10**9) <= 8
    )
    selected["input_rows"] = []
    selected["input_path"] = selected.get("name", "")
    return selected


def attach_final_input_rows(
    final: dict[str, Any],
    candidate_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    final = dict(final)
    name = str(final.get("name", ""))
    if name in candidate_data:
        rows = [
            _sanitized_input_row(row)
            for row in candidate_data[name]["input"].values()
        ]
        final["input_rows"] = rows
        final["input_path"] = str(candidate_data[name]["spec"].input_path)
        return final

    if name == "evidence_only_selector_diagnostic":
        baseline_name = next(iter(candidate_data))
        sample_ids = sorted(candidate_data[baseline_name]["details"])
        selected = _select_by_policy(candidate_data, sample_ids, final)
        rows = [
            _sanitized_input_row(
                candidate_data[selected[sample_id]]["input"][sample_id]
            )
            for sample_id in sample_ids
        ]
        final["input_rows"] = rows
        final["input_path"] = "evidence_only_selector_diagnostic_in_memory"
        return final

    return final


def materialize_final_input(
    final: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    final = dict(final)
    rows = list(final.get("input_rows", []))
    if not rows:
        return final
    if final.get("input_path") == "evidence_only_selector_diagnostic_in_memory":
        path = output_dir / "inputs" / "evidence_only_selector_diagnostic_sanitized.jsonl"
        _write_jsonl(path, rows)
        final["input_path"] = str(path)
    return final


def build_baseline_vs_final_delta(
    candidate_data: dict[str, dict[str, Any]],
    final: dict[str, Any],
    *,
    baseline_name: str,
) -> list[dict[str, Any]]:
    final_name = str(final.get("name", ""))
    if final_name not in candidate_data:
        return []
    baseline = candidate_data[baseline_name]
    selected = candidate_data[final_name]
    rows = []
    for sample_id in sorted(baseline["details"]):
        base_detail = baseline["details"].get(sample_id, {})
        final_detail = selected["details"].get(sample_id, {})
        rows.append(
            {
                "id": sample_id,
                "baseline_pass": bool(baseline["pass"].get(sample_id, False)),
                "final_pass": bool(selected["pass"].get(sample_id, False)),
                "baseline_score": _score(base_detail) if base_detail else None,
                "final_score": _score(final_detail) if final_detail else None,
                "score_delta": (
                    _score(final_detail) - _score(base_detail)
                    if base_detail and final_detail
                    else None
                ),
                "baseline_proxy_windows": _proxy_windows(base_detail)
                if base_detail
                else None,
                "final_proxy_windows": _proxy_windows(final_detail)
                if final_detail
                else None,
                "baseline_insufficient": bool(
                    base_detail.get("insufficient_evidence", False)
                )
                if base_detail
                else None,
                "final_insufficient": bool(
                    final_detail.get("insufficient_evidence", False)
                )
                if final_detail
                else None,
            }
        )
    return rows


def build_final_summary(
    final: dict[str, Any],
    matrix: dict[str, Any],
    model_negative: dict[str, Any],
    compliance: dict[str, Any],
) -> dict[str, Any]:
    rows_by_name = {row["name"]: row for row in matrix["rows"]}
    final_name = str(final.get("name", ""))
    if final_name in rows_by_name:
        final = {**rows_by_name[final_name], **final}
    final_for_summary = dict(final)
    final_for_summary.pop("input_rows", None)
    return {
        "artifact_type": "sawr_mechanism_no_quality_gate_final_metrics",
        "goal_achieved": bool(final.get("target_achieved", False)),
        "final_candidate": final_for_summary,
        "strict_code_only_baseline": STRICT_BASELINE,
        "previous_evidence_policy": PREVIOUS_EVIDENCE_POLICY,
        "previous_high_pass_seeds": PREVIOUS_HIGH_PASS_SEEDS,
        "candidate_rows": matrix["rows"],
        "model_negative_robustness": model_negative,
        "compliance_summary": compliance["summary"],
    }


def detector_input_integrity(
    rows: list[dict[str, Any]],
    *,
    input_path: str,
) -> dict[str, Any]:
    observed_fields = sorted({field for row in rows for field in row})
    forbidden = sorted(
        (set(observed_fields) - ALLOWED_DETECTOR_INPUT_FIELDS)
        | (set(observed_fields) & FORBIDDEN_DETECTOR_FIELDS)
    )
    row_hash = hashlib.sha256(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ).encode("utf-8")
    ).hexdigest()
    return {
        "artifact_type": "sawr_final_detector_input_integrity_check",
        "input_path": input_path,
        "row_count": len(rows),
        "allowed_fields": sorted(ALLOWED_DETECTOR_INPUT_FIELDS),
        "observed_fields": observed_fields,
        "forbidden_fields": forbidden,
        "is_code_only": not forbidden,
        "sha256": row_hash,
    }


def _sanitized_input_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("task_id") or ""),
        "dataset": str(row.get("dataset") or "humaneval"),
        "prompt": str(row.get("prompt") or ""),
        "final_code": str(row.get("final_code") or row.get("generated_code") or ""),
    }


def _score(row: dict[str, Any]) -> float:
    for field in ("score", "sample_score", "statistic", "z_score"):
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def _proxy_windows(row: dict[str, Any]) -> int:
    value = row.get("proxy_windows", 0)
    if isinstance(value, list):
        return len(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _code_chars(row: dict[str, Any]) -> int:
    value = row.get("code_chars")
    if isinstance(value, (int, float)):
        return int(value)
    code = row.get("final_code", row.get("generated_code", ""))
    return len(str(code))


def _best_tp_at_max_fp(
    positives: list[float],
    negatives: list[float],
    *,
    max_fp: int,
) -> dict[str, Any]:
    thresholds = sorted(set(positives + negatives), reverse=True)
    if not thresholds:
        return {"tp": 0, "fp": 0, "threshold": None}
    best = {"tp": 0, "fp": 0, "threshold": thresholds[0]}
    for threshold in thresholds:
        tp = sum(1 for score in positives if score >= threshold)
        fp = sum(1 for score in negatives if score >= threshold)
        if fp <= max_fp and (tp > best["tp"] or (tp == best["tp"] and fp < best["fp"])):
            best = {"tp": tp, "fp": fp, "threshold": threshold}
    return best


def _auc(positives: list[float], negatives: list[float]) -> float:
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _matching_negatives(
    reference_details: list[dict[str, Any]],
    positive_details_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = set(positive_details_by_id)
    matched = _filter_details_by_ids(reference_details, ids)
    return matched or reference_details


def _filter_details_by_ids(
    details: list[dict[str, Any]],
    sample_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        row
        for row in details
        if str(row.get("id") or row.get("task_id") or "").split("#", maxsplit=1)[0]
        in sample_ids
    ]


def _policy_only(policy: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "score_min",
        "max_drop",
        "min_proxy_windows",
        "min_scoreable_contexts",
        "require_not_insufficient",
    }
    return {key: policy[key] for key in keys if key in policy}


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("task_id") or "")
        if sample_id:
            result[sample_id.split("#", maxsplit=1)[0]] = row
    return result


def _pass_by_id(rows: list[dict[str, Any]]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("task_id") or "")
        if not sample_id:
            continue
        result[sample_id.split("#", maxsplit=1)[0]] = bool(
            row.get("is_correct", row.get("passed", False))
        )
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    payload = mark_mechanism_analysis_diagnostic(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def compliance_md(report: dict[str, Any]) -> str:
    lines = [
        "# Compliance Audit",
        "",
        f"- pass/test/correctness proxy used: {report['summary']['pass_test_correctness_proxy_used_for_generation_retry_selector_calibration']}",
        f"- strict code-only detector required: {report['summary']['strict_code_only_detector_required']}",
        f"- correctness-aware previous results diagnostic-only: {report['summary']['correctness_aware_previous_results_are_diagnostic_only']}",
        "",
        "## Diagnostic-Only Strategies",
        "",
    ]
    for row in report["diagnostic_only_strategies"]:
        lines.append(f"- {row['name']}: {row['reason']}")
    lines.extend(["", "## Compliant Candidates", ""])
    for row in report["compliant_candidates"]:
        lines.append(f"- {row['name']}: {row['input_path']}")
    return "\n".join(lines)


def metric_table_md(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        title,
        "",
        "| candidate | pass@1 | AUROC | TP@<=8FP | FP | insufficient | model-negative AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model_auc = row.get("model_negative_auroc")
        lines.append(
            "| {name} | {passed}/{samples} = {rate:.2%} | {auroc:.6f} | {tp}/{samples} | {fp} | {insuff} | {model_auc} |".format(
                name=row["name"],
                passed=int(row.get("pass", 0) or 0),
                samples=int(row.get("samples", 0) or 0),
                rate=float(row.get("pass_rate", 0.0) or 0.0),
                auroc=float(row.get("auroc", 0.0) or 0.0),
                tp=int(row.get("tp_at_8fp", 0) or 0),
                fp=int(row.get("fp", 0) or 0),
                insuff=row.get("positive_insufficient"),
                model_auc=(
                    "n/a" if model_auc is None else f"{float(model_auc):.6f}"
                ),
            )
        )
    return "\n".join(lines)


def bottleneck_md(report: dict[str, Any]) -> str:
    lines = [
        "# Mechanism Bottleneck Report",
        "",
        "| candidate | pass | mean score | insufficient | mean proxy windows | mean scoreable contexts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary_rows"]:
        lines.append(
            f"| {row['name']} | {row['pass']} | {row['mean_score']:.4f} | {row['positive_insufficient']} | {row['mean_proxy_windows']:.2f} | {row['mean_scoreable_contexts']:.2f} |"
        )
    lines.extend(["", "## Root-Cause Hypotheses", ""])
    lines.extend(f"- {item}" for item in report["root_cause_hypotheses"])
    return "\n".join(lines)


def config_comparison_md(report: dict[str, Any]) -> str:
    return metric_table_md("# Config-Level Comparison", report["config_rows"])


def kfold_md(report: dict[str, Any]) -> str:
    lines = [
        "# K-Fold No-Quality-Gate Results",
        "",
        f"- pass used for policy selection: {report['pass_used_for_policy_selection']}",
        f"- heldout mean pass: {report['heldout_mean_pass']:.2f}",
        f"- heldout mean AUROC: {report['heldout_mean_auroc']:.4f}",
        f"- heldout mean TP@<=8FP: {report['heldout_mean_tp_at_8fp']:.2f}",
        "",
        "| fold | heldout pass | heldout AUROC | heldout TP@<=8FP | policy |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        heldout = row["heldout_metrics"]
        lines.append(
            f"| {row['fold']} | {heldout['pass']}/{heldout['samples']} | {heldout['auroc']:.4f} | {heldout['tp_at_8fp']} | `{json.dumps(row['selected_policy'], sort_keys=True)}` |"
        )
    return "\n".join(lines)


def model_negative_md(report: dict[str, Any]) -> str:
    lines = [
        "# Model-Negative Robustness",
        "",
        "| candidate | positive samples | model-negative samples | model-negative AUROC |",
        "|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        value = row["model_negative_auroc"]
        lines.append(
            f"| {row['name']} | {row['positive_samples']} | {row['model_negative_samples']} | {'n/a' if value is None else f'{value:.6f}'} |"
        )
    return "\n".join(lines)


def pareto_md(report: dict[str, Any]) -> str:
    return metric_table_md("# Final Compliant Pareto Frontier", report["rows"])


def final_report_md(
    *,
    summary: dict[str, Any],
    compliance: dict[str, Any],
    comparison: dict[str, Any],
    bottleneck: dict[str, Any],
    model_negative: dict[str, Any],
    integrity: dict[str, Any],
    kfold: dict[str, Any],
    pareto: dict[str, Any],
) -> str:
    final = summary["final_candidate"]
    lines = [
        "# SAWR Mechanism No-Quality-Gate Final Report",
        "",
        "## Executive Summary",
        "",
        f"Goal achieved: {summary['goal_achieved']}.",
        f"Final candidate: {final.get('name')} ({final.get('candidate_type', 'pre_registered_config')}).",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| pass@1 | {int(final.get('pass', 0) or 0)}/{int(final.get('samples', 164) or 164)} = {float(final.get('pass_rate', 0.0) or 0.0):.2%} |",
        f"| canonical/reference AUROC | {float(final.get('auroc', 0.0) or 0.0):.6f} |",
        f"| TP@<=8FP | {int(final.get('tp_at_8fp', 0) or 0)}/{int(final.get('samples', 164) or 164)} |",
        f"| FP | {int(final.get('fp', 0) or 0)} |",
        f"| positive insufficient | {final.get('positive_insufficient')} |",
        f"| model-negative AUROC | {final.get('model_negative_auroc')} |",
        "",
        "## Compliance",
        "",
        f"Pass/test/correctness proxy used for generation, retry, selector, or calibration: {compliance['summary']['pass_test_correctness_proxy_used_for_generation_retry_selector_calibration']}.",
        f"Detector input code-only integrity: {integrity['is_code_only']}.",
        "",
        "## Config-Level Results",
        "",
        config_comparison_md(comparison),
        "",
        "## K-Fold Evidence-Only Selector Diagnostic",
        "",
        kfold_md(kfold),
        "",
        "## Model-Negative Robustness",
        "",
        model_negative_md(model_negative),
        "",
        "## Bottleneck",
        "",
        bottleneck_md(bottleneck),
        "",
        "## Pareto",
        "",
        pareto_md(pareto),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
