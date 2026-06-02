"""Stop/go summary aggregation for anchor validation diagnostics."""

from __future__ import annotations

from collections import defaultdict
from statistics import variance
from typing import Any

from wfcllm.evaluation.anchor_validation.metrics import bootstrap_mean_ci
from wfcllm.evaluation.anchor_validation.schema import (
    RegionMetricRow,
    SelectionSimulationRow,
)

_GAMMA_SCOPED_METHODS = {"candidate_centroid_oracle"}
_RANDOM_GAP_EXCLUDED_METHODS = {
    "vanilla",
    "random",
    "seqmark_oracle",
    "candidate_centroid_oracle",
    "context_centroid_oracle",
}


def build_anchor_validation_summary(
    metrics_rows: list[RegionMetricRow],
    selection_rows: list[SelectionSimulationRow],
    context_count: int,
    methods: tuple[str, ...],
    pool_quality: dict[str, Any] | None = None,
    empirical_gamma_rows: list[dict[str, Any]] | None = None,
    method_oracle_agreement: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    oracle_gap = _oracle_gap(metrics_rows)
    anchor_diagnostics = build_anchor_diagnostics(metrics_rows)
    evidence = {
        "paired_entropy_delta": _paired_entropy_delta(metrics_rows, baseline="vanilla"),
        "random_anchor_gap": _random_anchor_gap(metrics_rows),
        "oracle_gap": oracle_gap,
        "seqmark_oracle_gain_ratio": {
            method: values["gain_ratio"]
            for method, values in oracle_gap.items()
        },
        "method_oracle_agreement": method_oracle_agreement or {},
        "gamma_calibration": _gamma_calibration_summary(empirical_gamma_rows or []),
        "valid_hit_balance": _valid_hit_balance(metrics_rows),
        "key_wise_variance": _key_wise_variance(metrics_rows),
        "retry": _retry_summary(selection_rows),
        "low_entropy": _low_entropy_summary(metrics_rows),
        "node_type": _node_type_summary(metrics_rows),
    }
    gates = _layered_gates(evidence, pool_quality or {"context_count": context_count})
    first_stage_passed = all(gate["passed"] for gate in gates.values())
    return {
        "meta": {
            "context_count": context_count,
            "methods": list(methods),
        },
        "summary": {
            "mean_entropy_by_method": _mean_entropy_by_method(metrics_rows),
            "mean_hit_acquisition_by_method": {
                method: values["overall_hit_acquisition"]
                for method, values in evidence["retry"].items()
            },
            "pool_quality": pool_quality or {},
        },
        "anchor_diagnostics": anchor_diagnostics,
        "go_no_go": {
            "first_stage_passed": first_stage_passed,
            "end_to_end_followup_allowed": first_stage_passed,
            "gates": gates,
            "evidence": evidence,
        },
    }


def build_anchor_diagnostics(rows: list[RegionMetricRow]) -> dict[str, Any]:
    return {
        "by_method": _diagnostics_by_method(rows),
        "by_node_type": _node_type_summary(rows),
        "by_candidate_count_bucket": _bucketed_entropy(
            rows,
            lambda row: _candidate_count_bucket(row.candidate_count),
        ),
        "by_block_ordinal_bucket": _bucketed_entropy(
            rows,
            lambda row: _block_ordinal_bucket(row.block_ordinal),
        ),
        "top_contexts": _top_context_deltas(rows),
        "valid_hit_balance_by_gamma": _valid_hit_balance_by_gamma(rows),
    }


def _unkeyed(rows: list[RegionMetricRow]) -> list[RegionMetricRow]:
    return [
        row
        for row in rows
        if row.key_id is None and row.method not in _GAMMA_SCOPED_METHODS
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_entropy_by_method(rows: list[RegionMetricRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _unkeyed(rows):
        grouped[row.method].append(row.normalized_entropy)
    return {method: _mean(values) for method, values in sorted(grouped.items())}


def _diagnostics_by_method(rows: list[RegionMetricRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RegionMetricRow]] = defaultdict(list)
    for row in _unkeyed(rows):
        grouped[row.method].append(row)
    return {
        method: {
            "context_count": float(len({row.context_id for row in method_rows})),
            "mean_entropy": _mean([row.normalized_entropy for row in method_rows]),
            "mean_collapse_ratio": _mean([row.collapse_ratio for row in method_rows]),
            "mean_effective_region_count": _mean(
                [row.effective_region_count for row in method_rows]
            ),
            "mean_hamming_diversity": _mean(
                [row.hamming_diversity for row in method_rows]
            ),
        }
        for method, method_rows in sorted(grouped.items())
    }


def _context_method_means(rows: list[RegionMetricRow]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in _unkeyed(rows):
        grouped[(row.context_id, row.method)].append(row.normalized_entropy)
    return {key: _mean(values) for key, values in grouped.items()}


def _paired_entropy_delta(
    rows: list[RegionMetricRow],
    baseline: str,
) -> dict[str, dict[str, float | list[float]]]:
    by_context = _context_method_means(rows)
    contexts = {row.context_id for row in _unkeyed(rows)}
    methods = {row.method for row in _unkeyed(rows) if row.method != baseline}
    result: dict[str, dict[str, float | list[float]]] = {}
    for method in sorted(methods):
        deltas = [
            by_context[(context_id, method)] - by_context[(context_id, baseline)]
            for context_id in contexts
            if (context_id, method) in by_context and (context_id, baseline) in by_context
        ]
        ci = bootstrap_mean_ci(deltas, iterations=1000, seed=7)
        result[f"{method}_vs_{baseline}"] = {
            "mean": _mean(deltas),
            "median": sorted(deltas)[len(deltas) // 2] if deltas else 0.0,
            "win_rate": _mean([1.0 if delta > 0 else 0.0 for delta in deltas]),
            "bootstrap_ci_95": [ci[0], ci[1]],
        }
    return result


def _random_anchor_gap(rows: list[RegionMetricRow]) -> dict[str, dict[str, float | list[float]]]:
    paired = _paired_entropy_delta(rows, baseline="random")
    return {
        key.replace("_vs_random", "_minus_random"): value
        for key, value in paired.items()
        if key.removesuffix("_vs_random") not in _RANDOM_GAP_EXCLUDED_METHODS
    }


def _oracle_gap(rows: list[RegionMetricRow]) -> dict[str, dict[str, float]]:
    by_context = _context_method_means(rows)
    contexts = {row.context_id for row in _unkeyed(rows)}
    methods = {row.method for row in _unkeyed(rows)} - {"vanilla", "seqmark_oracle"}
    result: dict[str, dict[str, float]] = {}
    for method in sorted(methods):
        gain_values: list[float] = []
        oracle_gain_values: list[float] = []
        ratio_values: list[float] = []
        for context_id in contexts:
            vanilla = by_context.get((context_id, "vanilla"))
            oracle = by_context.get((context_id, "seqmark_oracle"))
            current = by_context.get((context_id, method))
            if vanilla is None or oracle is None or current is None:
                continue
            denom = oracle - vanilla
            gain = current - vanilla
            gain_values.append(gain)
            oracle_gain_values.append(denom)
            if denom > 0:
                ratio_values.append(gain / denom)
        result[method] = {
            "gain_vs_vanilla": _mean(gain_values),
            "oracle_gain": _mean(oracle_gain_values),
            "gain_ratio": _mean(ratio_values),
        }
    return result


def _gamma_calibration_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method", ""))].append(row)
    result: dict[str, dict[str, float]] = {}
    for method, method_rows in sorted(grouped.items()):
        deltas = [abs(float(row["delta"])) for row in method_rows if "delta" in row]
        empirical = [
            float(row["empirical_gamma"])
            for row in method_rows
            if "empirical_gamma" in row
        ]
        target = [
            float(row["target_gamma"])
            for row in method_rows
            if "target_gamma" in row
        ]
        result[method] = {
            "row_count": float(len(method_rows)),
            "mean_target_gamma": _mean(target),
            "mean_empirical_gamma": _mean(empirical),
            "mean_delta": _mean(deltas),
            "max_delta": max(deltas) if deltas else 0.0,
        }
    return result


def _valid_hit_balance(rows: list[RegionMetricRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.key_id is not None and row.gamma_deviation is not None:
            grouped[row.method].append(row.gamma_deviation)
    return {
        method: {
            "mean_delta_gamma": _mean(values),
            "max_delta_gamma": max(values) if values else 0.0,
        }
        for method, values in sorted(grouped.items())
    }


def _key_wise_variance(rows: list[RegionMetricRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.key_id is not None and row.valid_hit_rate is not None:
            grouped[row.method].append(row.valid_hit_rate)
    return {
        method: variance(values) if len(values) > 1 else 0.0
        for method, values in sorted(grouped.items())
    }


def _retry_summary(rows: list[SelectionSimulationRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[SelectionSimulationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
    result: dict[str, dict[str, float]] = {}
    for method, method_rows in sorted(grouped.items()):
        values = {
            "overall_hit_acquisition": _mean([1.0 if row.hit_acquired else 0.0 for row in method_rows]),
            "overall_fallback_rate": _mean([1.0 if row.fallback else 0.0 for row in method_rows]),
            "overall_z_proxy": _mean([row.z_proxy for row in method_rows]),
            "mean_selected_rank": _mean([float(row.selected_rank) for row in method_rows]),
        }
        for budget in (4, 8):
            budget_rows = [row for row in method_rows if row.retry_budget == budget]
            values[f"budget_{budget}_hit_acquisition"] = _mean([1.0 if row.hit_acquired else 0.0 for row in budget_rows])
            values[f"budget_{budget}_fallback_rate"] = _mean([1.0 if row.fallback else 0.0 for row in budget_rows])
            values[f"budget_{budget}_z_proxy"] = _mean([row.z_proxy for row in budget_rows])
            values[f"budget_{budget}_mean_selected_rank"] = _mean([float(row.selected_rank) for row in budget_rows])
        values.update(_quality_proxy_means(method_rows))
        result[method] = values
    return result


def _quality_proxy_means(rows: list[SelectionSimulationRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.quality.items():
            if isinstance(value, bool):
                grouped[f"quality_{key}"].append(1.0 if value else 0.0)
            elif isinstance(value, int | float):
                grouped[f"quality_{key}"].append(float(value))
    return {key: _mean(values) for key, values in sorted(grouped.items())}


def _low_entropy_summary(rows: list[RegionMetricRow]) -> dict[str, float]:
    baseline = [
        row.normalized_entropy
        for row in _unkeyed(rows)
        if row.method == "vanilla"
    ]
    if not baseline:
        return {}
    threshold = sorted(baseline)[max(0, int(0.25 * (len(baseline) - 1)))]
    low_contexts = {
        row.context_id
        for row in _unkeyed(rows)
        if row.method == "vanilla" and row.normalized_entropy <= threshold
    }
    return _mean_entropy_by_method([row for row in rows if row.context_id in low_contexts])


def _node_type_summary(rows: list[RegionMetricRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RegionMetricRow]] = defaultdict(list)
    for row in _unkeyed(rows):
        if row.node_type is not None:
            grouped[row.node_type].append(row)
    return {
        node_type: _mean_entropy_by_method(node_rows)
        for node_type, node_rows in sorted(grouped.items())
    }


def _bucketed_entropy(
    rows: list[RegionMetricRow],
    bucket_fn,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RegionMetricRow]] = defaultdict(list)
    for row in _unkeyed(rows):
        bucket = bucket_fn(row)
        if bucket is not None:
            grouped[bucket].append(row)
    return {
        bucket: _mean_entropy_by_method(bucket_rows)
        for bucket, bucket_rows in sorted(grouped.items())
    }


def _candidate_count_bucket(candidate_count: int) -> str:
    if candidate_count <= 3:
        return "1-3"
    if candidate_count <= 8:
        return "4-8"
    if candidate_count <= 16:
        return "9-16"
    if candidate_count <= 24:
        return "17-24"
    return "25+"


def _block_ordinal_bucket(block_ordinal: int | None) -> str | None:
    if block_ordinal is None:
        return None
    if block_ordinal <= 3:
        return "0-3"
    if block_ordinal <= 7:
        return "4-7"
    if block_ordinal <= 11:
        return "8-11"
    return "12+"


def _top_context_deltas(rows: list[RegionMetricRow]) -> dict[str, list[dict[str, Any]]]:
    by_context = _context_method_means(rows)
    contexts = sorted({row.context_id for row in _unkeyed(rows)})
    methods = sorted({row.method for row in _unkeyed(rows)} - {"vanilla", "seqmark_oracle"})
    oracle_minus_method: list[dict[str, Any]] = []
    method_minus_vanilla: list[dict[str, Any]] = []
    for context_id in contexts:
        vanilla = by_context.get((context_id, "vanilla"))
        oracle = by_context.get((context_id, "seqmark_oracle"))
        for method in methods:
            current = by_context.get((context_id, method))
            if current is None:
                continue
            if oracle is not None:
                oracle_minus_method.append(
                    {
                        "context_id": context_id,
                        "method": method,
                        "delta": oracle - current,
                    }
                )
            if vanilla is not None:
                method_minus_vanilla.append(
                    {
                        "context_id": context_id,
                        "method": method,
                        "delta": current - vanilla,
                    }
                )
    return {
        "seqmark_oracle_minus_method_largest": sorted(
            oracle_minus_method,
            key=lambda item: item["delta"],
            reverse=True,
        )[:10],
        "method_minus_vanilla_largest": sorted(
            method_minus_vanilla,
            key=lambda item: item["delta"],
            reverse=True,
        )[:10],
        "method_minus_vanilla_most_negative": sorted(
            method_minus_vanilla,
            key=lambda item: item["delta"],
        )[:10],
    }


def _valid_hit_balance_by_gamma(
    rows: list[RegionMetricRow],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[RegionMetricRow]] = defaultdict(list)
    for row in rows:
        if row.key_id is None or row.gamma is None:
            continue
        grouped[(row.method, f"{row.gamma:g}")].append(row)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (method, gamma), gamma_rows in sorted(grouped.items()):
        hit_rates = [
            row.valid_hit_rate
            for row in gamma_rows
            if row.valid_hit_rate is not None
        ]
        deltas = [
            row.gamma_deviation
            for row in gamma_rows
            if row.gamma_deviation is not None
        ]
        result[method][gamma] = {
            "row_count": float(len(gamma_rows)),
            "mean_valid_hit_rate": _mean([float(value) for value in hit_rates]),
            "mean_delta_gamma": _mean([float(value) for value in deltas]),
            "max_delta_gamma": max(deltas) if deltas else 0.0,
        }
    return {method: values for method, values in sorted(result.items())}


def _passes_first_stage(evidence: dict[str, Any]) -> bool:
    deterministic = ("slot_context", "slot_context_skeleton")
    paired = evidence["paired_entropy_delta"]
    random_gap = evidence["random_anchor_gap"]
    oracle_ratio = evidence["seqmark_oracle_gain_ratio"]
    balance = evidence["valid_hit_balance"]
    retry = evidence["retry"]
    low_entropy = evidence["low_entropy"]
    node_type = evidence["node_type"]
    for method in deterministic:
        if paired.get(f"{method}_vs_vanilla", {}).get("mean", 0.0) < 0.10:
            continue
        if low_entropy.get(method, 0.0) - low_entropy.get("vanilla", 0.0) < 0.15:
            continue
        node_type_wins = sum(
            1
            for values in node_type.values()
            if values.get(method, 0.0) - values.get("vanilla", 0.0) >= 0.05
        )
        if node_type_wins < 2:
            continue
        if random_gap.get(f"{method}_minus_random", {}).get("mean", 0.0) < 0.05:
            continue
        if oracle_ratio.get(method, 0.0) < 0.50:
            continue
        if balance.get(method, {}).get("max_delta_gamma", 1.0) > 0.05:
            continue
        method_retry_gain = max(
            retry.get(method, {}).get("budget_4_hit_acquisition", 0.0),
            retry.get(method, {}).get("budget_8_hit_acquisition", 0.0),
        )
        vanilla_retry_gain = max(
            retry.get("vanilla", {}).get("budget_4_hit_acquisition", 0.0),
            retry.get("vanilla", {}).get("budget_8_hit_acquisition", 0.0),
        )
        random_retry_gain = max(
            retry.get("random", {}).get("budget_4_hit_acquisition", 0.0),
            retry.get("random", {}).get("budget_8_hit_acquisition", 0.0),
        )
        if method_retry_gain <= vanilla_retry_gain:
            continue
        if method_retry_gain <= random_retry_gain:
            continue
        if retry.get(method, {}).get("overall_z_proxy", -999.0) <= retry.get("vanilla", {}).get("overall_z_proxy", -999.0):
            continue
        if retry.get(method, {}).get("overall_fallback_rate", 1.0) > retry.get("vanilla", {}).get("overall_fallback_rate", 1.0) + 0.05:
            continue
        if not _quality_non_regression(retry, method, baseline="vanilla"):
            continue
        if retry.get(method, {}).get("mean_selected_rank", 999.0) > 8.0:
            continue
        return True
    return False


def _layered_gates(
    evidence: dict[str, Any],
    pool_quality: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "data_quality_gate": _data_quality_gate(pool_quality),
        "anchor_signal_gate": _anchor_signal_gate(evidence),
        "vanilla_improvement_gate": _vanilla_improvement_gate(evidence),
        "balance_gate": _balance_gate(evidence),
        "retry_gate": _retry_gate(evidence),
    }


def _gate(passed: bool, reasons: list[str]) -> dict[str, Any]:
    return {"passed": passed, "failed_reasons": reasons}


def _data_quality_gate(pool_quality: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    context_count = int(pool_quality.get("context_count", 0))
    candidates = pool_quality.get("candidates_per_context", {})
    median_candidates = float(candidates.get("median", 0.0)) if isinstance(candidates, dict) else 0.0
    node_types = pool_quality.get("node_type_distribution", {})
    node_type_count = len(node_types) if isinstance(node_types, dict) else 0
    parse_valid_rate = float(pool_quality.get("parse_valid_rate", 0.0))
    if context_count < 40:
        reasons.append(f"context_count {context_count} < 40")
    if median_candidates < 8:
        reasons.append(f"median candidates per context {median_candidates:g} < 8")
    if node_type_count < 4:
        reasons.append(f"node_type categories {node_type_count} < 4")
    if parse_valid_rate < 0.98:
        reasons.append(f"parse_valid_rate {parse_valid_rate:.4f} < 0.98")
    return _gate(not reasons, reasons)


def _anchor_signal_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    delta = evidence["random_anchor_gap"].get("role_aware_slot_context_minus_random", {})
    ci = delta.get("bootstrap_ci_95", [0.0, 0.0])
    ci_lower = float(ci[0]) if ci else 0.0
    win_rate = float(delta.get("win_rate", 0.0))
    if ci_lower <= 0.0:
        reasons.append(f"role_aware_slot_context_minus_random CI lower {ci_lower:.4f} <= 0")
    if win_rate < 0.60:
        reasons.append(f"role_aware_slot_context_minus_random win_rate {win_rate:.4f} < 0.60")
    return _gate(not reasons, reasons)


def _vanilla_improvement_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    delta = evidence["paired_entropy_delta"].get("role_aware_slot_context_vs_vanilla", {})
    ci = delta.get("bootstrap_ci_95", [0.0, 0.0])
    mean_delta = float(delta.get("mean", 0.0))
    ci_lower = float(ci[0]) if ci else 0.0
    gain_ratio = float(evidence["oracle_gap"].get("role_aware_slot_context", {}).get("gain_ratio", 0.0))
    if mean_delta <= 0.03:
        reasons.append(f"role_aware_slot_context_vs_vanilla mean {mean_delta:.4f} <= 0.03")
    if ci_lower < 0.0:
        reasons.append(f"role_aware_slot_context_vs_vanilla CI lower {ci_lower:.4f} < 0")
    if gain_ratio < 0.15:
        reasons.append(f"gain_ratio {gain_ratio:.4f} < 0.15")
    return _gate(not reasons, reasons)


def _balance_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    balance = evidence["valid_hit_balance"].get("role_aware_slot_context", {})
    max_delta = float(balance.get("max_delta_gamma", 1.0))
    mean_delta = float(balance.get("mean_delta_gamma", 1.0))
    if max_delta > 0.35:
        reasons.append(f"max_delta_gamma {max_delta:.4f} > 0.35")
    if mean_delta > 0.15:
        reasons.append(f"mean_delta_gamma {mean_delta:.4f} > 0.15")
    return _gate(not reasons, reasons)


def _retry_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    retry = evidence["retry"]
    method = retry.get("role_aware_slot_context", {})
    vanilla = retry.get("vanilla", {})
    random = retry.get("random", {})
    method_hit = float(method.get("budget_4_hit_acquisition", 0.0))
    vanilla_hit = float(vanilla.get("budget_4_hit_acquisition", 0.0))
    random_hit = float(random.get("budget_4_hit_acquisition", 0.0))
    fallback = float(method.get("budget_4_fallback_rate", method.get("overall_fallback_rate", 1.0)))
    if method_hit <= vanilla_hit:
        reasons.append(f"budget_4_hit_acquisition {method_hit:.4f} <= vanilla {vanilla_hit:.4f}")
    if method_hit <= random_hit:
        reasons.append(f"budget_4_hit_acquisition {method_hit:.4f} <= random {random_hit:.4f}")
    if fallback > 0.15:
        reasons.append(f"fallback_rate {fallback:.4f} > 0.15")
    return _gate(not reasons, reasons)


def _quality_non_regression(
    retry: dict[str, dict[str, float]],
    method: str,
    baseline: str,
) -> bool:
    for quality_key in ("quality_syntax_valid", "quality_parse_valid"):
        method_value = retry.get(method, {}).get(quality_key)
        baseline_value = retry.get(baseline, {}).get(quality_key)
        if method_value is not None and baseline_value is not None:
            if method_value < baseline_value - 0.01:
                return False
    return True
