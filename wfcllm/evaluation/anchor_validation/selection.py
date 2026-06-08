"""Offline retry-budget selection simulation."""

from __future__ import annotations

import math

from wfcllm.evaluation.anchor_validation.schema import (
    CandidateBlock,
    SelectionSimulationRow,
)


def simulate_retry_selection(
    context_id: str,
    method: str,
    key_id: str,
    gamma: float,
    retry_budget: int,
    candidates: tuple[CandidateBlock, ...],
    signatures_by_candidate_id: dict[str, tuple[int, ...]],
    valid_set: frozenset[tuple[int, ...]],
) -> SelectionSimulationRow:
    ordered = sorted(candidates, key=lambda item: item.rank)
    budgeted = ordered[: max(1, retry_budget)]
    selected = ordered[0]
    hit_acquired = False
    for candidate in budgeted:
        signature = signatures_by_candidate_id[candidate.candidate_id]
        if signature in valid_set:
            selected = candidate
            hit_acquired = True
            break
    fallback = not hit_acquired
    z_proxy = _z_proxy(1 if hit_acquired else 0, gamma)
    return SelectionSimulationRow(
        context_id=context_id,
        method=method,
        key_id=key_id,
        gamma=gamma,
        retry_budget=retry_budget,
        selected_candidate_id=selected.candidate_id,
        selected_rank=selected.rank,
        hit_acquired=hit_acquired,
        fallback=fallback,
        z_proxy=z_proxy,
        quality={
            **dict(selected.quality),
            "syntax_valid": selected.syntax_valid,
            "parse_valid": selected.parse_valid,
        },
    )


def _z_proxy(observed_hit: int, gamma: float) -> float:
    variance = gamma * (1.0 - gamma)
    if variance <= 0.0:
        return 0.0
    return (observed_hit - gamma) / math.sqrt(variance)
