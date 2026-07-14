from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

from wfcllm.dynamic_semantic.channel import (
    AggregateEvidence,
    UnitEvidence,
    aggregate_unit_evidence,
)
from wfcllm.generation.quality_v2 import (
    StaticQualityAssessment,
    assess_static_quality,
)
from wfcllm.generation.selection_v2 import RetryAttempt


@dataclass(frozen=True)
class _ScoredDynamicAttempt:
    attempt: RetryAttempt
    quality: StaticQualityAssessment
    aggregate: AggregateEvidence
    erasures: int


@dataclass(frozen=True)
class DynamicAttemptSelection:
    result: Any
    attempt_index: int
    aggregate: AggregateEvidence
    quality: StaticQualityAssessment
    no_embedding: bool


def select_dynamic_attempt(
    *,
    prompt: str,
    attempts: tuple[RetryAttempt, ...],
    evidence_by_attempt: Mapping[int, tuple[UnitEvidence, ...]],
    erasures_by_attempt: Mapping[int, int],
    minimum_independent_units: int,
) -> DynamicAttemptSelection:
    if not attempts:
        raise ValueError("attempts must not be empty")
    attempt_indices = [attempt.attempt_index for attempt in attempts]
    if len(set(attempt_indices)) != len(attempt_indices):
        raise ValueError("attempt indices must be unique")
    scored = tuple(
        _ScoredDynamicAttempt(
            attempt=attempt,
            quality=assess_static_quality(
                prompt=prompt,
                final_code=str(attempt.result.final_code),
            ),
            aggregate=aggregate_unit_evidence(
                evidence_by_attempt.get(attempt.attempt_index, ()),
                minimum_independent_units=minimum_independent_units,
            ),
            erasures=int(erasures_by_attempt.get(attempt.attempt_index, 0)),
        )
        for attempt in attempts
    )
    eligible = tuple(
        item
        for item in scored
        if item.quality.eligible and item.aggregate.eligible
    )
    no_embedding = not eligible
    if eligible:
        selected = max(eligible, key=_semantic_ranking_key)
    else:
        selected = max(scored, key=_static_fallback_key)
    return DynamicAttemptSelection(
        result=selected.attempt.result,
        attempt_index=selected.attempt.attempt_index,
        aggregate=selected.aggregate,
        quality=selected.quality,
        no_embedding=no_embedding,
    )


def _semantic_ranking_key(
    item: _ScoredDynamicAttempt,
) -> tuple[int, Fraction, int, int, int]:
    return (
        item.quality.quality_tier,
        Fraction(item.aggregate.numerator, item.aggregate.denominator),
        item.aggregate.independent_units,
        -item.erasures,
        -item.attempt.attempt_index,
    )


def _static_fallback_key(item: _ScoredDynamicAttempt) -> tuple[int, int, int]:
    return (
        item.quality.quality_tier,
        -int(item.attempt.result.fallback_count),
        -item.attempt.attempt_index,
    )
