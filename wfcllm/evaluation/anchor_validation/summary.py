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


def build_anchor_validation_summary(
    metrics_rows: list[RegionMetricRow],
    selection_rows: list[SelectionSimulationRow],
    context_count: int,
    methods: tuple[str, ...],
) -> dict[str, Any]:
    evidence = {
        "paired_entropy_delta": _paired_entropy_delta(metrics_rows, baseline="vanilla"),
        "random_anchor_gap": _random_anchor_gap(metrics_rows),
        "seqmark_oracle_gain_ratio": _seqmark_oracle_gain_ratio(metrics_rows),
        "valid_hit_balance": _valid_hit_balance(metrics_rows),
        "key_wise_variance": _key_wise_variance(metrics_rows),
        "retry": _retry_summary(selection_rows),
        "low_entropy": _low_entropy_summary(metrics_rows),
        "node_type": _node_type_summary(metrics_rows),
    }
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
        },
        "go_no_go": {
            "first_stage_passed": _passes_first_stage(evidence),
            "end_to_end_followup_allowed": _passes_first_stage(evidence),
            "evidence": evidence,
        },
    }


def _unkeyed(rows: list[RegionMetricRow]) -> list[RegionMetricRow]:
    return [row for row in rows if row.key_id is None]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_entropy_by_method(rows: list[RegionMetricRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _unkeyed(rows):
        grouped[row.method].append(row.normalized_entropy)
    return {method: _mean(values) for method, values in sorted(grouped.items())}


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
        if key.startswith(("slot", "context", "skeleton", "prompt"))
    }


def _seqmark_oracle_gain_ratio(rows: list[RegionMetricRow]) -> dict[str, float]:
    by_context = _context_method_means(rows)
    contexts = {row.context_id for row in _unkeyed(rows)}
    methods = {row.method for row in _unkeyed(rows)} - {"vanilla", "seqmark_oracle"}
    ratios: dict[str, float] = {}
    for method in sorted(methods):
        values: list[float] = []
        for context_id in contexts:
            vanilla = by_context.get((context_id, "vanilla"))
            oracle = by_context.get((context_id, "seqmark_oracle"))
            current = by_context.get((context_id, method))
            if vanilla is None or oracle is None or current is None:
                continue
            denom = oracle - vanilla
            if denom > 0:
                values.append((current - vanilla) / denom)
        ratios[method] = _mean(values)
    return ratios


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
