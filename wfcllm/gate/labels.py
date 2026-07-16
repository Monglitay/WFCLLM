"""Causal, no-quality-proxy supervision from one ordered candidate trajectory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from wfcllm.gate.schema import CandidateObservation
from wfcllm.method.contracts import reject_quality_proxy_fields

_TRAINING_KEY_COUNT = 32
_REWRITE_BUDGETS = (1, 3, 6)
_CANONICAL_KEY_IDS = tuple(
    f"train-key-{key_index:03d}"
    for key_index in range(_TRAINING_KEY_COUNT)
)
_KEY_EVIDENCE_FIELDS = frozenset({"hit", "stable", "margin"})
_RATE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LabelThresholds:
    """Frozen first-version thresholds for reliable gate-data labels.

    ``configured_margin`` is applied in addition to the key-specific ``hit``
    and ``stable`` facts recorded for each candidate.
    """

    r3_hit_rate: float = 0.60
    structural_valid_rate: float = 2 / 3
    unstable_rate: float = 0.10
    configured_margin: float = 0.0

    def __post_init__(self) -> None:
        for name in ("r3_hit_rate", "structural_valid_rate", "unstable_rate"):
            _validate_rate(name, getattr(self, name))
        _validate_non_negative_finite(
            "configured_margin", self.configured_margin
        )


@dataclass(frozen=True)
class BudgetOutcome:
    """Reliable-hit and rewrite-health summary for one trajectory prefix."""

    budget: int
    candidate_indices: tuple[int, ...]
    rewrite_count: int
    success_rate: float
    structural_valid_rate: float
    unstable_rate: float
    first_hit_by_key_id: Mapping[str, int | None]

    def __post_init__(self) -> None:
        if type(self.budget) is not int or self.budget not in _REWRITE_BUDGETS:
            raise ValueError("budget must be one of 1, 3, or 6")
        expected_indices = tuple(range(self.budget + 1))
        if (
            not isinstance(self.candidate_indices, tuple)
            or any(type(index) is not int for index in self.candidate_indices)
            or self.candidate_indices != expected_indices
        ):
            raise ValueError("candidate_indices must be the budget prefix")
        if type(self.rewrite_count) is not int or self.rewrite_count != self.budget:
            raise ValueError("rewrite_count must equal budget")
        for name in ("success_rate", "structural_valid_rate", "unstable_rate"):
            _validate_rate(name, getattr(self, name))
        if not isinstance(self.first_hit_by_key_id, Mapping):
            raise ValueError("first_hit_by_key_id must be a mapping")
        if set(self.first_hit_by_key_id) != set(_CANONICAL_KEY_IDS):
            raise ValueError(
                "first_hit_by_key_id must contain exactly the 32 training key IDs"
            )
        first_hits = {
            key_id: self.first_hit_by_key_id[key_id]
            for key_id in _CANONICAL_KEY_IDS
        }
        for candidate_index in first_hits.values():
            if candidate_index is not None and (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index not in expected_indices
            ):
                raise ValueError("first-hit indices must belong to the budget prefix")
        expected_success_rate = (
            sum(value is not None for value in first_hits.values())
            / _TRAINING_KEY_COUNT
        )
        if not math.isclose(
            self.success_rate,
            expected_success_rate,
            rel_tol=0.0,
            abs_tol=_RATE_TOLERANCE,
        ):
            raise ValueError(
                "success_rate must equal reliable first hits divided by 32"
            )
        _validate_discrete_rate(
            "structural_valid_rate",
            self.structural_valid_rate,
            denominator=self.budget,
        )
        _validate_discrete_rate(
            "unstable_rate",
            self.unstable_rate,
            denominator=self.budget,
        )
        if self.unstable_rate > self.structural_valid_rate + _RATE_TOLERANCE:
            raise ValueError("unstable_rate must not exceed structural_valid_rate")
        object.__setattr__(
            self,
            "first_hit_by_key_id",
            MappingProxyType(first_hits),
        )

    def to_dict(self) -> dict[str, Any]:
        row = {
            "budget": self.budget,
            "candidate_indices": list(self.candidate_indices),
            "rewrite_count": self.rewrite_count,
            "success_rate": self.success_rate,
            "structural_valid_rate": self.structural_valid_rate,
            "unstable_rate": self.unstable_rate,
            "first_hit_by_key_id": dict(self.first_hit_by_key_id),
        }
        reject_quality_proxy_fields(row)
        return row


@dataclass(frozen=True)
class GateLabels:
    """Two supervision targets plus auditable same-trajectory budget outcomes."""

    close_target: bool
    suitable_target: bool
    budgets: Mapping[int, BudgetOutcome]

    def __post_init__(self) -> None:
        if not isinstance(self.close_target, bool):
            raise ValueError("close_target must be a bool")
        if not isinstance(self.suitable_target, bool):
            raise ValueError("suitable_target must be a bool")
        if not isinstance(self.budgets, Mapping):
            raise ValueError("budgets must be a mapping")
        if (
            any(type(budget) is not int for budget in self.budgets)
            or set(self.budgets) != set(_REWRITE_BUDGETS)
        ):
            raise ValueError("budgets must contain exactly outcomes for 1, 3, and 6")
        budgets = {budget: self.budgets[budget] for budget in _REWRITE_BUDGETS}
        for budget, outcome in budgets.items():
            if not isinstance(outcome, BudgetOutcome) or outcome.budget != budget:
                raise ValueError("each budget must map to its BudgetOutcome")
        if self.suitable_target and not self.close_target:
            raise ValueError("suitable_target requires close_target")
        _validate_first_hit_prefixes(budgets)
        _validate_rewrite_metric_prefixes(budgets)
        object.__setattr__(self, "budgets", MappingProxyType(budgets))

    def to_dict(self) -> dict[str, Any]:
        row = {
            "close_target": self.close_target,
            "suitable_target": self.suitable_target,
            "budgets": {
                str(budget): outcome.to_dict()
                for budget, outcome in self.budgets.items()
            },
        }
        reject_quality_proxy_fields(row)
        return row


def build_gate_labels(
    candidates: tuple[CandidateObservation, ...],
    *,
    training_key_count: int,
    thresholds: LabelThresholds | None = None,
    hard_boundary: bool = False,
) -> GateLabels:
    """Build causal labels from candidate 0 and ordered rewrites 1 through 6.

    All budgets are prefixes of ``candidates``. Candidate 0 participates in
    reliable semantic success, but only candidates 1..R contribute to rewrite
    structural-validity and instability rates, both with R as denominator. A
    structurally unavailable rewrite is not classified as numerical
    instability. A reliable hit requires candidate-level precision and batch
    stability plus key-level ``hit``, ``stable``, and configured margin.

    ``hard_boundary`` is a keyword-only fact about the currently generated
    window. No future statement or later-than-R3 candidate influences either
    supervision target.
    """

    if type(training_key_count) is not int or training_key_count != _TRAINING_KEY_COUNT:
        raise ValueError("training_key_count must remain fixed at 32")
    if not isinstance(hard_boundary, bool):
        raise ValueError("hard_boundary must be a bool")
    if thresholds is None:
        thresholds = LabelThresholds()
    elif not isinstance(thresholds, LabelThresholds):
        raise ValueError("thresholds must be LabelThresholds")
    _validate_trajectory(candidates)

    key_ids = _CANONICAL_KEY_IDS
    candidate_zero = candidates[0]
    if (
        not _is_structurally_usable(candidate_zero)
        or set(candidate_zero.lsh_by_key_id) != set(key_ids)
    ):
        raise ValueError(
            "candidate 0 must be structurally valid with complete evidence"
        )
    if not candidate_zero.code.strip():
        raise ValueError("candidate 0 code must be nonblank")
    for candidate in candidates:
        structurally_usable = _is_structurally_usable(candidate)
        if not candidate.lsh_by_key_id and not structurally_usable:
            continue
        if set(candidate.lsh_by_key_id) != set(key_ids):
            raise ValueError(
                "each candidate must contain exactly all training key IDs"
            )
        for key_id in key_ids:
            _validate_key_evidence(candidate.lsh_by_key_id[key_id], key_id=key_id)

    reliable_hits = tuple(
        {
            key_id: (
                False
                if not candidate.lsh_by_key_id
                else _is_reliable_hit(
                    candidate,
                    candidate.lsh_by_key_id[key_id],
                    configured_margin=thresholds.configured_margin,
                )
            )
            for key_id in key_ids
        }
        for candidate in candidates
    )

    outcomes: dict[int, BudgetOutcome] = {}
    for budget in _REWRITE_BUDGETS:
        prefix = candidates[: budget + 1]
        rewrite_prefix = prefix[1:]
        first_hits: dict[str, int | None] = {}
        for key_id in key_ids:
            first_hits[key_id] = next(
                (
                    candidate_index
                    for candidate_index in range(budget + 1)
                    if reliable_hits[candidate_index][key_id]
                ),
                None,
            )

        structural_count = sum(
            _is_structurally_usable(candidate)
            for candidate in rewrite_prefix
        )
        unstable_count = sum(
            _is_structurally_usable(candidate)
            and not _is_candidate_stable(candidate)
            for candidate in rewrite_prefix
        )
        outcomes[budget] = BudgetOutcome(
            budget=budget,
            candidate_indices=tuple(candidate.candidate_index for candidate in prefix),
            rewrite_count=len(rewrite_prefix),
            success_rate=sum(value is not None for value in first_hits.values())
            / training_key_count,
            structural_valid_rate=structural_count / budget,
            unstable_rate=unstable_count / budget,
            first_hit_by_key_id=first_hits,
        )

    r3 = outcomes[3]
    suitable_target = (
        r3.success_rate >= thresholds.r3_hit_rate
        and r3.structural_valid_rate >= thresholds.structural_valid_rate
        and r3.unstable_rate <= thresholds.unstable_rate
    )
    current_unit_count = candidates[0].unit_count
    if current_unit_count not in {1, 2, 3}:
        raise ValueError("candidate 0 unit_count must be one of 1, 2, or 3")
    close_target = hard_boundary or current_unit_count == 3 or suitable_target
    return GateLabels(
        close_target=close_target,
        suitable_target=suitable_target,
        budgets=outcomes,
    )


def _validate_trajectory(candidates: object) -> None:
    if not isinstance(candidates, tuple):
        raise ValueError("candidates must be a tuple")
    if any(not isinstance(candidate, CandidateObservation) for candidate in candidates):
        raise ValueError("candidates must contain only CandidateObservation instances")
    if tuple(candidate.candidate_index for candidate in candidates) != tuple(range(7)):
        raise ValueError(
            "candidate trajectory must contain ordered indices 0 through 6"
        )


def _validate_key_evidence(evidence: object, *, key_id: str) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"evidence for {key_id} must be a mapping")
    evidence_row = dict(evidence)
    reject_quality_proxy_fields(evidence_row)
    if set(evidence_row) != _KEY_EVIDENCE_FIELDS:
        raise ValueError(
            f"evidence for {key_id} must contain exactly hit, stable, and margin"
        )
    if not isinstance(evidence["hit"], bool):
        raise ValueError(f"hit evidence for {key_id} must be a bool")
    if not isinstance(evidence["stable"], bool):
        raise ValueError(f"stable evidence for {key_id} must be a bool")
    _validate_non_negative_finite(
        f"margin evidence for {key_id}", evidence["margin"]
    )


def _is_structurally_usable(candidate: CandidateObservation) -> bool:
    return (
        candidate.parse_status == "ok"
        and candidate.same_parent_scope
        and candidate.unit_count in {1, 2, 3}
    )


def _is_candidate_stable(candidate: CandidateObservation) -> bool:
    return (
        candidate.stable_across_precision_modes
        and candidate.stable_across_batch_modes
    )


def _is_reliable_hit(
    candidate: CandidateObservation,
    evidence: Mapping[str, bool | float],
    *,
    configured_margin: float,
) -> bool:
    return (
        _is_structurally_usable(candidate)
        and _is_candidate_stable(candidate)
        and evidence["hit"] is True
        and evidence["stable"] is True
        and float(evidence["margin"]) >= configured_margin
    )


def _validate_rate(name: str, value: object) -> None:
    if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite rate in [0, 1]")


def _validate_non_negative_finite(name: str, value: object) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite non-negative number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_discrete_rate(name: str, value: float, *, denominator: int) -> None:
    scaled = value * denominator
    if not math.isclose(
        scaled,
        round(scaled),
        rel_tol=0.0,
        abs_tol=_RATE_TOLERANCE,
    ):
        raise ValueError(
            f"{name} must be an integer candidate count divided by budget"
        )


def _validate_first_hit_prefixes(
    budgets: Mapping[int, BudgetOutcome],
) -> None:
    for key_id in _CANONICAL_KEY_IDS:
        previous_budget = 1
        previous_hit = budgets[previous_budget].first_hit_by_key_id[key_id]
        for budget in (3, 6):
            current_hit = budgets[budget].first_hit_by_key_id[key_id]
            if previous_hit is not None:
                if current_hit != previous_hit:
                    raise ValueError(
                        "cross-budget first-hit indices must preserve prefix hits"
                    )
            elif current_hit is not None and current_hit <= previous_budget:
                raise ValueError(
                    "new cross-budget first-hit index must be in the added prefix"
                )
            previous_budget = budget
            previous_hit = current_hit


def _validate_rewrite_metric_prefixes(
    budgets: Mapping[int, BudgetOutcome],
) -> None:
    for previous_budget, budget in zip(_REWRITE_BUDGETS, _REWRITE_BUDGETS[1:]):
        previous = budgets[previous_budget]
        current = budgets[budget]
        previous_structural = round(
            previous.structural_valid_rate * previous_budget
        )
        current_structural = round(current.structural_valid_rate * budget)
        previous_unstable = round(previous.unstable_rate * previous_budget)
        current_unstable = round(current.unstable_rate * budget)
        structural_delta = current_structural - previous_structural
        unstable_delta = current_unstable - previous_unstable
        budget_delta = budget - previous_budget
        if not 0 <= structural_delta <= budget_delta:
            raise ValueError(
                "structural prefix count growth exceeds the added rewrites"
            )
        if not 0 <= unstable_delta <= structural_delta:
            raise ValueError(
                "unstable prefix count growth must be within structural growth"
            )
