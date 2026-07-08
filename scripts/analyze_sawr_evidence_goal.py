#!/usr/bin/env python
"""Analyze SAWR evidence-constrained candidate and robustness experiments."""

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
ALLOWED_DETECTOR_INPUT_FIELDS = {"id", "dataset", "prompt", "final_code"}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    input_path: Path
    details_path: Path
    pass_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SAWR evidence-constrained goal analysis artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    robustness = subparsers.add_parser(
        "robustness",
        help="Summarize reference-negative vs model-negative detector robustness.",
    )
    robustness.add_argument("--input-dir", required=True)
    robustness.add_argument("--output-json", required=True)
    robustness.add_argument("--output-md", required=True)
    robustness.set_defaults(handler=_cmd_robustness)

    policy = subparsers.add_parser(
        "policy-report",
        help="Sweep evidence-constrained multi-candidate selector policies.",
    )
    policy.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME:INPUT_JSONL:DETAILS_JSONL:PASS_JSONL",
    )
    policy.add_argument("--reference-details", required=True)
    policy.add_argument("--model-negative-details", default=None)
    policy.add_argument("--output-dir", required=True)
    policy.add_argument("--folds", type=int, default=5)
    policy.set_defaults(handler=_cmd_policy_report)

    return parser


def _cmd_robustness(args: argparse.Namespace) -> int:
    summary = build_model_negative_robustness(Path(args.input_dir))
    _write_json(Path(args.output_json), summary)
    _write_md(Path(args.output_md), _robustness_md(summary))
    return 0


def _cmd_policy_report(args: argparse.Namespace) -> int:
    summary = build_evidence_policy_report(
        [_parse_candidate_spec(spec) for spec in args.candidate],
        reference_details_path=Path(args.reference_details),
        model_negative_details_path=(
            Path(args.model_negative_details)
            if args.model_negative_details is not None
            else None
        ),
        output_dir=Path(args.output_dir),
        folds=args.folds,
        final_policy=None,
    )
    print(json.dumps(summary["final"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_model_negative_robustness(input_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        reference_report_path = family_dir / "reference_report.json"
        model_report_path = family_dir / "model_negative_report.json"
        if not reference_report_path.exists() or not model_report_path.exists():
            continue

        reference_report = _load_json(reference_report_path)
        model_report = _load_json(model_report_path)
        reference_primary = _primary(reference_report)
        model_primary = _primary(model_report)
        reference_auroc = float(reference_primary.get("auroc", 0.0) or 0.0)
        model_auroc = float(model_primary.get("auroc", 0.0) or 0.0)
        drop = round(reference_auroc - model_auroc, 12)
        row = {
            "family": family_dir.name,
            "reference_auroc": reference_auroc,
            "model_negative_auroc": model_auroc,
            "model_negative_drop": drop,
            "reference_positive_samples": int(
                reference_primary.get("positive_samples", 0) or 0
            ),
            "reference_negative_samples": int(
                reference_primary.get("negative_samples", 0) or 0
            ),
            "model_negative_samples": int(
                model_primary.get("negative_samples", 0) or 0
            ),
            "reference_positive_insufficient": int(
                reference_primary.get("positive_insufficient_samples", 0) or 0
            ),
            "model_negative_insufficient": int(
                model_primary.get("negative_insufficient_samples", 0) or 0
            ),
            "mean_code_chars": {
                "positive": _mean_code_chars(family_dir / "positive_details.jsonl"),
                "reference_negative": _mean_code_chars(
                    family_dir / "reference_negative_details.jsonl"
                ),
                "model_negative": _mean_code_chars(
                    family_dir / "model_negative_details.jsonl"
                ),
            },
        }
        row["diagnostic_only"] = bool(
            row["model_negative_auroc"] < 0.75 or row["model_negative_drop"] >= 0.05
        )
        rows.append(row)

    return {
        "artifact_type": "sawr_model_negative_robustness",
        "rows": rows,
        "diagnostic_only_families": [
            row["family"] for row in rows if row["diagnostic_only"]
        ],
    }


def build_evidence_policy_report(
    candidates: list[CandidateSpec],
    *,
    reference_details_path: Path,
    model_negative_details_path: Path | None,
    output_dir: Path,
    folds: int,
    final_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if len(candidates) < 1:
        raise ValueError("at least one candidate is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_data = {
        spec.name: {
            "input": _by_id(_load_jsonl(spec.input_path)),
            "details": _by_id(_load_jsonl(spec.details_path)),
            "pass": _pass_by_id(_load_jsonl(spec.pass_path)),
        }
        for spec in candidates
    }
    reference_details = _load_jsonl(reference_details_path)
    model_negative_details = (
        _load_jsonl(model_negative_details_path)
        if model_negative_details_path is not None
        else []
    )
    baseline_name = candidates[0].name
    sample_ids = sorted(candidate_data[baseline_name]["details"])

    candidate_matrix = _candidate_matrix(
        candidate_data,
        reference_details,
        model_negative_details,
    )
    policies = _sweep_policies(candidate_data, sample_ids)
    scored_policies = [
        {
            **policy,
            **_metrics_for_selection(
                candidate_data,
                reference_details,
                model_negative_details,
                _select_by_policy(candidate_data, sample_ids, policy),
                sample_ids,
            ),
        }
        for policy in policies
    ]
    folds_report = _cross_validate_policies(
        candidate_data,
        reference_details,
        model_negative_details,
        policies,
        sample_ids=sample_ids,
        folds=max(2, int(folds)),
    )
    selected_policy = final_policy or _best_policy(scored_policies)
    selected = _select_by_policy(candidate_data, sample_ids, selected_policy)
    final_metrics = _metrics_for_selection(
        candidate_data,
        reference_details,
        model_negative_details,
        selected,
        sample_ids,
    )
    selected_rows = [
        _sanitized_row(candidate_data[selected[sample_id]]["input"][sample_id])
        for sample_id in sample_ids
    ]
    final_details = [
        candidate_data[selected[sample_id]]["details"][sample_id]
        for sample_id in sample_ids
    ]
    per_sample_delta = _per_sample_delta(
        candidate_data,
        selected,
        sample_ids,
        baseline_name=baseline_name,
    )
    integrity = _detector_input_integrity(
        selected_rows,
        input_path=str(output_dir / "final_selected_sanitized.jsonl"),
    )
    pareto = _pareto_frontier(scored_policies)
    bottleneck = _bottleneck_report(candidate_data, selected, sample_ids, baseline_name)

    _write_jsonl(output_dir / "final_selected_sanitized.jsonl", selected_rows)
    _write_jsonl(output_dir / "final_positive_details_from_candidates.jsonl", final_details)
    _write_jsonl(output_dir / "per_sample_baseline_vs_final_delta.jsonl", per_sample_delta)
    _write_json(output_dir / "candidate_metric_matrix.json", candidate_matrix)
    _write_json(output_dir / "kfold_or_heldout_selector_results.json", folds_report)
    _write_md(
        output_dir / "kfold_or_heldout_selector_results.md",
        _folds_md(folds_report),
    )
    _write_json(output_dir / "final_pareto_frontier.json", pareto)
    _write_md(output_dir / "final_pareto_frontier.md", _pareto_md(pareto))
    _write_json(output_dir / "selector_or_generation_bottleneck_report.json", bottleneck)
    _write_md(
        output_dir / "selector_or_generation_bottleneck_report.md",
        _bottleneck_md(bottleneck),
    )
    _write_json(output_dir / "final_detector_input_integrity_check.json", integrity)

    summary = {
        "artifact_type": "sawr_evidence_constrained_policy_summary",
        "baseline": next(
            row for row in candidate_matrix["rows"] if row["name"] == baseline_name
        ),
        "final": {
            "name": "evidence_constrained_policy",
            "policy": _policy_only(selected_policy),
            **final_metrics,
        },
    }
    summary["final"]["delta_vs_strict_code_only_baseline"] = _metric_delta(
        summary["baseline"],
        summary["final"],
    )
    _write_json(output_dir / "final_metrics_summary.json", summary)
    return summary


def _parse_candidate_spec(spec: str) -> CandidateSpec:
    parts = spec.split(":", maxsplit=3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(
            "--candidate must be NAME:INPUT_JSONL:DETAILS_JSONL:PASS_JSONL"
        )
    return CandidateSpec(parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3]))


def _candidate_matrix(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    reference_details: list[dict[str, Any]],
    model_negative_details: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for name, data in candidate_data.items():
        ids = sorted(data["details"])
        selected = {sample_id: name for sample_id in ids}
        rows.append(
            {
                "name": name,
                **_metrics_for_selection(
                    candidate_data,
                    reference_details,
                    model_negative_details,
                    selected,
                    ids,
                ),
            }
        )
    return {"artifact_type": "sawr_evidence_candidate_metric_matrix", "rows": rows}


def _sweep_policies(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
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
    policies = []
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
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    policy: dict[str, Any],
) -> dict[str, str]:
    names = list(candidate_data)
    baseline_name = names[0]
    selected: dict[str, str] = {}
    for sample_id in sample_ids:
        baseline_detail = candidate_data[baseline_name]["details"][sample_id]
        baseline_score = _score(baseline_detail)
        eligible = [baseline_name]
        for name in names[1:]:
            detail = candidate_data[name]["details"].get(sample_id)
            if not detail:
                continue
            score = _score(detail)
            if score < float(policy["score_min"]):
                continue
            max_drop = policy.get("max_drop")
            if max_drop is not None and score < baseline_score - float(max_drop):
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
                -names.index(name),
            ),
        )
    return selected


def _metrics_for_selection(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    reference_details: list[dict[str, Any]],
    model_negative_details: list[dict[str, Any]],
    selected: dict[str, str],
    sample_ids: list[str],
) -> dict[str, Any]:
    positive_details = [
        candidate_data[selected[sample_id]]["details"][sample_id]
        for sample_id in sample_ids
    ]
    positive_scores = [_score(row) for row in positive_details]
    reference_scores = [_score(row) for row in reference_details]
    model_negative_scores = [_score(row) for row in model_negative_details]
    reference_operating = _best_tp_at_max_fp(
        positive_scores,
        reference_scores,
        max_fp=8,
    )
    model_operating = _best_tp_at_max_fp(
        positive_scores,
        model_negative_scores,
        max_fp=8,
    )
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
        "auroc": _auc(positive_scores, reference_scores),
        "model_negative_auroc": (
            _auc(positive_scores, model_negative_scores)
            if model_negative_scores
            else None
        ),
        "tp_at_8fp": reference_operating["tp"],
        "fp": reference_operating["fp"],
        "threshold": reference_operating["threshold"],
        "model_negative_tp_at_8fp": (
            model_operating["tp"] if model_negative_scores else None
        ),
        "model_negative_fp": (
            model_operating["fp"] if model_negative_scores else None
        ),
        "positive_insufficient": sum(
            1 for row in positive_details if bool(row.get("insufficient_evidence", False))
        ),
        "selected_counts": selected_counts,
    }


def _cross_validate_policies(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    reference_details: list[dict[str, Any]],
    model_negative_details: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    *,
    sample_ids: list[str],
    folds: int,
) -> dict[str, Any]:
    rows = []
    for fold_index in range(folds):
        heldout = [
            sample_id
            for index, sample_id in enumerate(sample_ids)
            if index % folds == fold_index
        ]
        train = [sample_id for sample_id in sample_ids if sample_id not in set(heldout)]
        train_scored = []
        for policy in policies:
            train_selected = _select_by_policy(candidate_data, train, policy)
            train_metrics = _metrics_for_selection(
                candidate_data,
                reference_details,
                model_negative_details,
                train_selected,
                train,
            )
            train_scored.append((_policy_objective(train_metrics), policy, train_metrics))
        _objective, selected_policy, train_metrics = max(
            train_scored,
            key=lambda item: item[0],
        )
        heldout_selected = _select_by_policy(candidate_data, heldout, selected_policy)
        heldout_metrics = _metrics_for_selection(
            candidate_data,
            reference_details,
            model_negative_details,
            heldout_selected,
            heldout,
        )
        rows.append(
            {
                "fold": fold_index,
                "train_ids": len(train),
                "heldout_ids": len(heldout),
                "selected_policy": _policy_only(selected_policy),
                "train_metrics": train_metrics,
                "heldout_metrics": heldout_metrics,
            }
        )
    return {
        "artifact_type": "sawr_evidence_selector_kfold_results",
        "folds": folds,
        "rows": rows,
        "heldout_mean_pass": mean([row["heldout_metrics"]["pass"] for row in rows])
        if rows
        else 0.0,
        "heldout_mean_auroc": mean([row["heldout_metrics"]["auroc"] for row in rows])
        if rows
        else 0.0,
        "heldout_mean_model_negative_auroc": mean(
            [
                row["heldout_metrics"]["model_negative_auroc"]
                for row in rows
                if row["heldout_metrics"]["model_negative_auroc"] is not None
            ]
        )
        if any(
            row["heldout_metrics"]["model_negative_auroc"] is not None
            for row in rows
        )
        else None,
        "heldout_mean_tp_at_8fp": mean(
            [row["heldout_metrics"]["tp_at_8fp"] for row in rows]
        )
        if rows
        else 0.0,
    }


def _best_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no policies to select")
    return max(rows, key=lambda row: _policy_objective(row))


def _policy_objective(metrics: dict[str, Any]) -> tuple[int, int, float, float, int]:
    model_negative_auroc = metrics.get("model_negative_auroc")
    robust_model = model_negative_auroc is None or float(model_negative_auroc) >= 0.75
    meets_detection = (
        float(metrics["auroc"]) >= 0.80
        and int(metrics["tp_at_8fp"]) >= min(76, int(metrics["samples"]))
        and int(metrics["fp"]) <= 8
        and robust_model
    )
    return (
        1 if meets_detection else 0,
        int(metrics["pass"]),
        float(metrics["auroc"]),
        float(model_negative_auroc) if model_negative_auroc is not None else 0.0,
        int(metrics["tp_at_8fp"]),
    )


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["pass"] >= row["pass"]
                and other["auroc"] >= row["auroc"]
                and other["tp_at_8fp"] >= row["tp_at_8fp"]
                and other["fp"] <= row["fp"]
                and (
                    other["pass"] > row["pass"]
                    or other["auroc"] > row["auroc"]
                    or other["tp_at_8fp"] > row["tp_at_8fp"]
                    or other["fp"] < row["fp"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append({key: row[key] for key in row if key != "selected_counts"})
    return sorted(frontier, key=lambda item: (item["pass"], item["auroc"]), reverse=True)


def _per_sample_delta(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    selected: dict[str, str],
    sample_ids: list[str],
    *,
    baseline_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for sample_id in sample_ids:
        selected_name = selected[sample_id]
        baseline_detail = candidate_data[baseline_name]["details"][sample_id]
        final_detail = candidate_data[selected_name]["details"][sample_id]
        rows.append(
            {
                "id": sample_id,
                "selected_candidate": selected_name,
                "baseline_pass": bool(
                    candidate_data[baseline_name]["pass"].get(sample_id, False)
                ),
                "final_pass": bool(
                    candidate_data[selected_name]["pass"].get(sample_id, False)
                ),
                "baseline_score": _score(baseline_detail),
                "final_score": _score(final_detail),
                "score_delta": _score(final_detail) - _score(baseline_detail),
                "baseline_proxy_windows": _proxy_windows(baseline_detail),
                "final_proxy_windows": _proxy_windows(final_detail),
                "baseline_insufficient": bool(
                    baseline_detail.get("insufficient_evidence", False)
                ),
                "final_insufficient": bool(
                    final_detail.get("insufficient_evidence", False)
                ),
            }
        )
    return rows


def _bottleneck_report(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    selected: dict[str, str],
    sample_ids: list[str],
    baseline_name: str,
) -> dict[str, Any]:
    selected_improves_pass = []
    selected_loses_pass = []
    high_pass_rejected = []
    for sample_id in sample_ids:
        baseline_pass = bool(candidate_data[baseline_name]["pass"].get(sample_id, False))
        selected_name = selected[sample_id]
        final_pass = bool(candidate_data[selected_name]["pass"].get(sample_id, False))
        if final_pass and not baseline_pass:
            selected_improves_pass.append(sample_id)
        if baseline_pass and not final_pass:
            selected_loses_pass.append(sample_id)
        pass_candidates = [
            name
            for name, data in candidate_data.items()
            if bool(data["pass"].get(sample_id, False))
        ]
        if pass_candidates and selected_name not in pass_candidates:
            high_pass_rejected.append(
                {
                    "id": sample_id,
                    "selected": selected_name,
                    "passing_candidates": pass_candidates,
                    "selected_score": _score(
                        candidate_data[selected_name]["details"][sample_id]
                    ),
                }
            )
    return {
        "artifact_type": "sawr_evidence_selector_bottleneck",
        "baseline": baseline_name,
        "selected_pass_gains": len(selected_improves_pass),
        "selected_pass_losses": len(selected_loses_pass),
        "oracle_union_pass": sum(
            1
            for sample_id in sample_ids
            if any(bool(data["pass"].get(sample_id, False)) for data in candidate_data.values())
        ),
        "passing_candidate_rejected_count": len(high_pass_rejected),
        "selected_pass_gain_ids": selected_improves_pass,
        "selected_pass_loss_ids": selected_loses_pass,
        "passing_candidate_rejected_examples": high_pass_rejected[:50],
    }


def _detector_input_integrity(
    rows: list[dict[str, Any]],
    *,
    input_path: str,
) -> dict[str, Any]:
    bad_rows = []
    sha = hashlib.sha256()
    for index, row in enumerate(rows, start=1):
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        sha.update(encoded + b"\n")
        extra = sorted(set(row) - ALLOWED_DETECTOR_INPUT_FIELDS)
        forbidden = sorted(FORBIDDEN_DETECTOR_FIELDS & set(row))
        missing = sorted(ALLOWED_DETECTOR_INPUT_FIELDS - set(row))
        if extra or forbidden or missing:
            bad_rows.append(
                {
                    "line": index,
                    "id": row.get("id"),
                    "extra_fields": extra,
                    "forbidden_fields": forbidden,
                    "missing_fields": missing,
                }
            )
    return {
        "artifact_type": "sawr_detector_input_integrity_check",
        "input": input_path,
        "rows": len(rows),
        "allowed_fields": sorted(ALLOWED_DETECTOR_INPUT_FIELDS),
        "forbidden_fields_checked": sorted(FORBIDDEN_DETECTOR_FIELDS),
        "bad_row_count": len(bad_rows),
        "bad_rows": bad_rows,
        "sha256": sha.hexdigest(),
    }


def _metric_delta(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": int(final["pass"]) - int(baseline["pass"]),
        "pass_rate": float(final["pass_rate"]) - float(baseline["pass_rate"]),
        "auroc": float(final["auroc"]) - float(baseline["auroc"]),
        "tp_at_8fp": int(final["tp_at_8fp"]) - int(baseline["tp_at_8fp"]),
        "fp": int(final["fp"]) - int(baseline["fp"]),
    }


def _policy_only(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "score_min",
            "max_drop",
            "min_proxy_windows",
            "min_scoreable_contexts",
            "require_not_insufficient",
        )
    }


def _primary(report: dict[str, Any]) -> dict[str, Any]:
    primary = report.get("primary", {})
    if not isinstance(primary, dict):
        raise ValueError("detector report primary section must be an object")
    return primary


def _mean_code_chars(path: Path) -> float:
    if not path.exists():
        return 0.0
    values = [
        float(row.get("code_chars", 0.0) or 0.0)
        for row in _load_jsonl(path)
        if row.get("code_chars") is not None
    ]
    return mean(values) if values else 0.0


def _score(row: dict[str, Any]) -> float:
    return float(row.get("score", 0.0) or 0.0)


def _proxy_windows(row: dict[str, Any]) -> int:
    value = row.get("proxy_windows", 0)
    if isinstance(value, (list, tuple)):
        return len(value)
    return int(value or 0)


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


def _best_tp_at_max_fp(
    positive: list[float],
    negative: list[float],
    *,
    max_fp: int,
) -> dict[str, Any]:
    best = {"tp": 0, "fp": 0, "threshold": math.inf}
    for threshold in sorted(set(positive + negative), reverse=True):
        fp = sum(score >= threshold for score in negative)
        if fp > max_fp:
            continue
        tp = sum(score >= threshold for score in positive)
        if tp > best["tp"] or (tp == best["tp"] and fp > best["fp"]):
            best = {"tp": tp, "fp": fp, "threshold": threshold}
    return best


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (
        position - lower
    )


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _sanitized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("task_id") or ""),
        "dataset": str(row.get("dataset") or "humaneval"),
        "prompt": str(row.get("prompt") or ""),
        "final_code": str(row.get("final_code") or row.get("generated_code") or ""),
    }


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("task_id") or "")
        if not sample_id:
            raise ValueError("row is missing id")
        if sample_id in by_id:
            raise ValueError(f"duplicate id: {sample_id}")
        by_id[sample_id] = row
    return by_id


def _pass_by_id(rows: list[dict[str, Any]]) -> dict[str, bool]:
    result = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("task_id") or "")
        if not sample_id:
            raise ValueError("pass row is missing id")
        result[sample_id] = bool(row.get("is_correct", row.get("passed", False)))
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _robustness_md(summary: dict[str, Any]) -> str:
    lines = [
        "# SAWR Model-Negative Robustness",
        "",
        "| family | reference AUROC | model-negative AUROC | drop | diagnostic-only |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {family} | {ref:.4f} | {model:.4f} | {drop:.4f} | {diag} |".format(
                family=row["family"],
                ref=row["reference_auroc"],
                model=row["model_negative_auroc"],
                drop=row["model_negative_drop"],
                diag="yes" if row["diagnostic_only"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "Diagnostic-only means the reference-negative gain does not survive against matched model-generated negatives, or drops by at least 0.05 AUROC.",
        ]
    )
    return "\n".join(lines) + "\n"


def _folds_md(report: dict[str, Any]) -> str:
    lines = [
        "# SAWR Evidence-Constrained K-Fold Report",
        "",
        f"- folds: {report.get('folds')}",
        f"- heldout mean pass: {report.get('heldout_mean_pass', 0.0):.2f}",
        f"- heldout mean AUROC: {report.get('heldout_mean_auroc', 0.0):.4f}",
        f"- heldout mean model-negative AUROC: {report.get('heldout_mean_model_negative_auroc')}",
        f"- heldout mean TP@<=8FP: {report.get('heldout_mean_tp_at_8fp', 0.0):.2f}",
        "",
        "| fold | heldout pass | heldout AUROC | model-neg AUROC | heldout TP@<=8FP | selected policy |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows", []):
        heldout = row["heldout_metrics"]
        model_auroc = heldout.get("model_negative_auroc")
        model_text = "n/a" if model_auroc is None else f"{model_auroc:.4f}"
        lines.append(
            "| {fold} | {passed}/{total} | {auroc:.4f} | {model} | {tp} | `{policy}` |".format(
                fold=row["fold"],
                passed=heldout["pass"],
                total=heldout["samples"],
                auroc=heldout["auroc"],
                model=model_text,
                tp=heldout["tp_at_8fp"],
                policy=json.dumps(row["selected_policy"], sort_keys=True),
            )
        )
    return "\n".join(lines) + "\n"


def _pareto_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SAWR Evidence-Constrained Pareto Frontier",
        "",
        "| pass | AUROC | model-neg AUROC | TP@<=8FP | FP | policy |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        model_auroc = row.get("model_negative_auroc")
        model_text = "n/a" if model_auroc is None else f"{model_auroc:.4f}"
        lines.append(
            "| {passed}/{samples} | {auroc:.4f} | {model} | {tp} | {fp} | `{policy}` |".format(
                passed=row["pass"],
                samples=row["samples"],
                auroc=row["auroc"],
                model=model_text,
                tp=row["tp_at_8fp"],
                fp=row["fp"],
                policy=json.dumps(_policy_only(row), sort_keys=True),
            )
        )
    return "\n".join(lines) + "\n"


def _bottleneck_md(report: dict[str, Any]) -> str:
    lines = [
        "# SAWR Evidence-Constrained Bottleneck Report",
        "",
        f"- baseline: {report.get('baseline')}",
        f"- selected pass gains: {report.get('selected_pass_gains')}",
        f"- selected pass losses: {report.get('selected_pass_losses')}",
        f"- oracle union pass: {report.get('oracle_union_pass')}",
        f"- passing candidate rejected count: {report.get('passing_candidate_rejected_count')}",
        "",
        "Interpretation: pass-improving candidates remain available, but evidence-constrained selection rejects some passing rows when their strict code-only detector evidence is weaker than competing candidates.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
