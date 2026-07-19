"""Group-level admission checks for semantic-gate data collection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any

FEASIBILITY_CONTRACT_VERSION = "gate-data-feasibility/v1"
_PILOT_MIN = 100
_PILOT_MAX = 300
_FULL_MIN = 300
_FULL_MAX = 300
FEASIBILITY_THRESHOLD_ITEMS: tuple[tuple[str, int | float], ...] = (
    ("pilot_independent_group_min", 100),
    ("pilot_independent_group_max", 300),
    ("full_independent_group_min", 300),
    ("full_independent_group_max", 300),
    ("pilot_suitable_positive_min", 10),
    ("pilot_suitable_negative_min", 25),
    ("full_suitable_positive_min", 30),
    ("full_suitable_negative_min", 75),
    ("window_length_group_min", 50),
    ("major_statement_family_count_min", 4),
    ("major_statement_family_group_min", 25),
    ("r3_minus_r1_bootstrap_lower_95_exclusive_min", 0.0),
    ("holdout_key_absolute_decline_max", 0.10),
    ("validation_test_suitable_positive_min", 3),
    ("validation_test_suitable_negative_min", 8),
)


@dataclass(frozen=True)
class FeasibilityGroup:
    """One independent window start; derived context/key/budget rows are absent."""

    group_id: str
    suitable_target: bool
    window_lengths: tuple[int, ...]
    statement_family: str
    r1_success_rate: float
    r3_success_rate: float
    holdout_success_rate: float
    split: str
    repository_id: str
    task_id: str
    generation_model_id: str
    structural_invalid_rate: float
    numeric_instability_rate: float
    first_hit_candidate_position: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        if type(self.suitable_target) is not bool:
            raise ValueError("suitable_target must be a bool")
        if (
            not isinstance(self.window_lengths, tuple)
            or not self.window_lengths
            or any(type(value) is not int or value not in {1, 2, 3} for value in self.window_lengths)
            or len(set(self.window_lengths)) != len(self.window_lengths)
        ):
            raise ValueError("window_lengths must be unique W1/W2/W3 integers")
        if not isinstance(self.statement_family, str) or not self.statement_family:
            raise ValueError("statement_family must be a non-empty string")
        for name in (
            "r1_success_rate",
            "r3_success_rate",
            "holdout_success_rate",
            "structural_invalid_rate",
            "numeric_instability_rate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite rate in [0, 1]")
        for name in ("repository_id", "task_id", "generation_model_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.first_hit_candidate_position is not None and (
            type(self.first_hit_candidate_position) is not int
            or not 0 <= self.first_hit_candidate_position <= 3
        ):
            raise ValueError("first_hit_candidate_position must be None or 0 through 3")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")


@dataclass(frozen=True)
class AdmissionResult:
    passed: bool
    observed: int | float
    requirement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDataFeasibilitySummary:
    contract_version: str
    scale: str
    passed: bool
    independent_group_count: int
    admissions: Mapping[str, AdmissionResult]
    deterministic_label_stability_curve: Mapping[str, Mapping[str, int | float]]
    thresholds: Mapping[str, int | float]
    suitable_groups: Mapping[str, int | float]
    coverage: Mapping[str, Any]
    reliable_success: Mapping[str, Any]
    rewrite_health: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "scale": self.scale,
            "passed": self.passed,
            "independent_group_count": self.independent_group_count,
            "admissions": {name: result.to_dict() for name, result in sorted(self.admissions.items())},
            "thresholds": dict(sorted(self.thresholds.items())),
            "suitable_groups": dict(self.suitable_groups),
            "coverage": dict(self.coverage),
            "reliable_success": dict(self.reliable_success),
            "rewrite_health": dict(self.rewrite_health),
            "deterministic_label_stability_curve": {
                size: dict(values)
                for size, values in self.deterministic_label_stability_curve.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"


def evaluate_gate_data_feasibility(
    groups: Sequence[FeasibilityGroup], *, scale: str
) -> GateDataFeasibilitySummary:
    """Apply the frozen v1 admission line to independent groups only."""

    if isinstance(groups, (str, bytes, bytearray)) or not isinstance(groups, Sequence):
        raise ValueError("groups must be a sequence of FeasibilityGroup values")
    snapshot = tuple(groups)
    if not snapshot or any(not isinstance(group, FeasibilityGroup) for group in snapshot):
        raise ValueError("groups must contain FeasibilityGroup values")
    snapshot = tuple(sorted(snapshot, key=lambda group: group.group_id))
    ids = tuple(group.group_id for group in snapshot)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate independent group_id values are forbidden")
    if scale not in {"pilot", "full"}:
        raise ValueError("scale must be pilot or full")

    count = len(snapshot)
    minimum, maximum = (_PILOT_MIN, _PILOT_MAX) if scale == "pilot" else (_FULL_MIN, _FULL_MAX)
    positives = sum(group.suitable_target for group in snapshot)
    negatives = count - positives
    windows = {length: sum(length in group.window_lengths for group in snapshot) for length in (1, 2, 3)}
    families: dict[str, int] = {}
    for group in snapshot:
        families[group.statement_family] = families.get(group.statement_family, 0) + 1
    family_min = int(dict(FEASIBILITY_THRESHOLD_ITEMS)["major_statement_family_group_min"])
    qualifying_families = sum(value >= family_min for value in families.values())
    differences = tuple(group.r3_success_rate - group.r1_success_rate for group in snapshot)
    lower, gain_upper = _bootstrap_mean_ci95(differences, ids)
    training_success = sum(group.r3_success_rate for group in snapshot) / count
    holdout_success = sum(group.holdout_success_rate for group in snapshot) / count
    # The contract limits transfer *degradation*. A better holdout rate is not
    # itself a failure and therefore contributes zero decline.
    gap = max(0.0, training_success - holdout_success)

    thresholds = dict(FEASIBILITY_THRESHOLD_ITEMS)
    positive_min = int(thresholds[f"{scale}_suitable_positive_min"])
    negative_min = int(thresholds[f"{scale}_suitable_negative_min"])
    admissions = {
        f"{scale}_group_count": AdmissionResult(minimum <= count <= maximum, count, f"{minimum} <= independent groups <= {maximum}"),
        "suitable_positive_groups": AdmissionResult(positives >= positive_min, positives, f">= {positive_min}"),
        "suitable_negative_groups": AdmissionResult(negatives >= negative_min, negatives, f">= {negative_min}"),
        **{
            f"window_length_w{length}_groups": AdmissionResult(windows[length] >= 50, windows[length], ">= 50")
            for length in (1, 2, 3)
        },
        "major_statement_families": AdmissionResult(qualifying_families >= 4, qualifying_families, ">= 4 families with >= 25 independent groups each"),
        "r3_minus_r1_bootstrap_lower_95": AdmissionResult(lower > 0.0, lower, "> 0"),
        "holdout_key_absolute_gap": AdmissionResult(gap <= 0.10, gap, "<= 0.10"),
    }
    ordered = sorted(snapshot, key=lambda group: (hashlib.sha256(group.group_id.encode("utf-8")).hexdigest(), group.group_id))
    selected_positive = sum(group.suitable_target for group in ordered)
    curve: dict[str, dict[str, int | float]] = {
        "full": {
            "group_count": count,
            "suitable_positive_count": selected_positive,
            "suitable_positive_rate": selected_positive / count,
        }
    }
    reliable_success = {}
    for budget_name, values in (
        ("r1", tuple(group.r1_success_rate for group in snapshot)),
        ("r3", tuple(group.r3_success_rate for group in snapshot)),
    ):
        ci_low, ci_high = _bootstrap_mean_ci95(values, ids)
        reliable_success[budget_name] = {
            "mean": sum(values) / count,
            "bootstrap_ci95": [ci_low, ci_high],
        }
    first_hits = [
        group.first_hit_candidate_position
        for group in snapshot
        if group.first_hit_candidate_position is not None
    ]
    first_hit_distribution = {
        str(position): sum(value == position for value in first_hits)
        for position in range(4)
    }
    return GateDataFeasibilitySummary(
        contract_version=FEASIBILITY_CONTRACT_VERSION,
        scale=scale,
        passed=all(result.passed for result in admissions.values()),
        independent_group_count=count,
        admissions=admissions,
        deterministic_label_stability_curve=curve,
        thresholds=dict(FEASIBILITY_THRESHOLD_ITEMS),
        suitable_groups={
            "positive_count": positives,
            "negative_count": negatives,
            "positive_rate": positives / count,
            "negative_rate": negatives / count,
        },
        coverage={
            "window_length_group_counts": {str(key): value for key, value in windows.items()},
            "statement_family_group_counts": dict(sorted(families.items())),
            "repository_count": len({group.repository_id for group in snapshot}),
            "task_count": len({group.task_id for group in snapshot}),
            "generation_model_count": len({group.generation_model_id for group in snapshot}),
            "repository_ids": sorted({group.repository_id for group in snapshot}),
            "task_ids": sorted({group.task_id for group in snapshot}),
            "generation_model_ids": sorted({group.generation_model_id for group in snapshot}),
        },
        reliable_success={
            **reliable_success,
            "r3_minus_r1": {
                "mean": sum(differences) / count,
                "bootstrap_ci95": [lower, gain_upper],
            },
            "holdout_key_mean": holdout_success,
            "training_key_r3_mean": training_success,
            "holdout_absolute_decline": gap,
        },
        rewrite_health={
            "structural_invalid_rate": sum(group.structural_invalid_rate for group in snapshot) / count,
            "numeric_instability_rate": sum(group.numeric_instability_rate for group in snapshot) / count,
            "first_hit_candidate_position_distribution": first_hit_distribution,
            "first_hit_observed_count": len(first_hits),
            "first_hit_missing_count": count - len(first_hits),
            "first_hit_candidate_position_mean": None if not first_hits else sum(first_hits) / len(first_hits),
        },
    )


def _bootstrap_mean_ci95(values: tuple[float, ...], group_ids: tuple[str, ...]) -> tuple[float, float]:
    ordered_pairs = tuple(sorted(zip(group_ids, values), key=lambda item: (item[0], item[1])))
    ordered_ids = tuple(group_id for group_id, _ in ordered_pairs)
    ordered_values = tuple(value for _, value in ordered_pairs)
    seed_bytes = hashlib.sha256("\0".join(ordered_ids).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    count = len(ordered_values)
    means = []
    # Fixed v1 bootstrap count and seed make the contract reproducible.
    for _ in range(1_000):
        means.append(sum(ordered_values[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return means[24], means[974]
