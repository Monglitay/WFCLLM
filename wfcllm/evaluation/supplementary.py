"""Supplementary Ablation mechanism and latency summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import median
from typing import Any


def build_supplementary_mechanism_report(
    *,
    generation_manifest: Mapping[str, object],
    candidate_rows: Iterable[Mapping[str, object]],
    latency_rows: Iterable[Mapping[str, object]],
    max_rewrites: int,
    baseline_mean_latency_seconds: float | None = None,
) -> dict[str, object]:
    """Build the audited generation mechanism funnel.

    Candidate-zero is retained as the original-program control and is therefore
    not counted as a rewrite attempt.  Duplicate attempts are exact duplicate
    candidate texts within one sample/window trajectory, including a duplicate
    of candidate-zero.
    """

    if type(max_rewrites) is not int or max_rewrites < 1:
        raise ValueError("max_rewrites must be a positive integer")
    sample_count = _nonnegative_int(generation_manifest, "sample_count")

    candidates = [dict(row) for row in candidate_rows]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        sample_id = row.get("id")
        window_index = row.get("window_index")
        candidate_index = row.get("candidate_index")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or type(window_index) is not int
            or window_index < 0
            or type(candidate_index) is not int
            or candidate_index < 0
        ):
            raise ValueError("candidate funnel row identity is invalid")
        grouped[(sample_id, window_index)].append(row)

    for trajectory in grouped.values():
        indices = [int(row["candidate_index"]) for row in trajectory]
        if len(indices) != len(set(indices)) or 0 not in indices:
            raise ValueError(
                "candidate funnel trajectories require unique indices and candidate-zero"
            )
        trajectory.sort(key=lambda row: int(row["candidate_index"]))

    window_count = len(grouped)
    carriers_by_sample: dict[str, int] = defaultdict(int)
    for sample_id, _window_index in grouped:
        carriers_by_sample[sample_id] += 1
    suitable_count = sum(
        trajectory[0].get("evaluation_status") == "evaluated_original"
        for trajectory in grouped.values()
    )
    candidate_zero_accept_count = sum(
        trajectory[0].get("evaluation_status") == "evaluated_original"
        and trajectory[0].get("selected") is True
        for trajectory in grouped.values()
    )
    rewrite_rows = [
        row
        for trajectory in grouped.values()
        for row in trajectory
        if int(row["candidate_index"]) > 0
    ]
    rewrite_count = len(rewrite_rows)

    duplicate_count = 0
    for trajectory in grouped.values():
        seen: set[str] = set()
        for row in trajectory:
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError("candidate funnel text must be a string")
            if int(row["candidate_index"]) > 0 and text in seen:
                duplicate_count += 1
            seen.add(text)

    structurally_valid_count = sum(
        row.get("structure_ok") is True for row in rewrite_rows
    )
    detector_recoverable_count = sum(
        row.get("structure_ok") is True
        and row.get("semantic_preservation_passed") is True
        for row in rewrite_rows
    )
    stable_margin_count = sum(
        row.get("semantic_stable") is True for row in rewrite_rows
    )
    keyed_hit_count = sum(
        row.get("semantic_stable") is True
        and row.get("semantic_hit") is True
        for row in rewrite_rows
    )
    final_accepted_count = sum(
        row.get("selected") is True for row in rewrite_rows
    )
    budget_exhausted_count = 0
    for trajectory in grouped.values():
        evaluated_rewrites = sum(
            int(row["candidate_index"]) > 0
            and row.get("evaluation_status")
            != "generated_not_evaluated_after_accept"
            for row in trajectory
        )
        if (
            trajectory[0].get("selected") is True
            and evaluated_rewrites >= max_rewrites
        ):
            budget_exhausted_count += 1

    latency_values: list[float] = []
    latency_ids: set[str] = set()
    per_sample_latency: list[dict[str, str | float]] = []
    for row in latency_rows:
        sample_id = row.get("id")
        value = row.get("generation_latency_seconds")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in latency_ids
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("generation latency sidecar row is invalid")
        latency_ids.add(sample_id)
        latency_values.append(float(value))
        per_sample_latency.append(
            {
                "id": sample_id,
                "generation_latency_seconds": float(value),
            }
        )
    if len(latency_values) != sample_count:
        raise ValueError("generation latency must cover every sample exactly once")
    if not latency_values:
        raise ValueError("generation latency cannot be empty")
    if set(carriers_by_sample) - latency_ids:
        raise ValueError(
            "candidate funnel contains a sample missing from generation latency"
        )

    total_latency = sum(latency_values)
    mean_latency = total_latency / len(latency_values)
    ordered_latency = sorted(latency_values)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latency)) - 1)
    if baseline_mean_latency_seconds is None:
        relative_multiplier = None
    else:
        baseline = float(baseline_mean_latency_seconds)
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("baseline mean latency must be finite and positive")
        relative_multiplier = mean_latency / baseline

    carrier_value = _funnel_value(window_count, sample_count)
    carrier_value["per_program"] = {
        sample_id: carriers_by_sample.get(sample_id, 0)
        for sample_id in sorted(latency_ids)
    }
    carrier_value["mean_per_program"] = (
        window_count / sample_count if sample_count else 0.0
    )
    return {
        "mechanism_funnel": {
            "carrier": carrier_value,
            "suitable_pass": _funnel_value(suitable_count, window_count),
            "candidate_zero_accept": _funnel_value(
                candidate_zero_accept_count, suitable_count
            ),
            "rewrite_attempt": _funnel_value(rewrite_count, rewrite_count),
            "duplicate": _funnel_value(duplicate_count, rewrite_count),
            "structurally_valid": _funnel_value(
                structurally_valid_count, rewrite_count
            ),
            "detector_recoverable": _funnel_value(
                detector_recoverable_count, rewrite_count
            ),
            "stable_margin": _funnel_value(stable_margin_count, rewrite_count),
            "keyed_hit": _funnel_value(keyed_hit_count, rewrite_count),
            "final_accepted": _funnel_value(
                final_accepted_count, rewrite_count
            ),
            "budget_exhausted": _funnel_value(
                budget_exhausted_count, suitable_count
            ),
        },
        "latency": {
            "definition": "base_generation_start_to_final_program_delivery_monotonic/v1",
            "sample_count": len(latency_values),
            "per_sample": sorted(
                per_sample_latency, key=lambda row: str(row["id"])
            ),
            "total_seconds": total_latency,
            "mean_seconds": mean_latency,
            "median_seconds": float(median(latency_values)),
            "p95_seconds": ordered_latency[p95_index],
            "baseline_relative_mean_multiplier": relative_multiplier,
        },
    }


def _nonnegative_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _funnel_value(count: int, denominator: int) -> dict[str, int | float]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else 0.0,
    }


__all__ = ["build_supplementary_mechanism_report"]
