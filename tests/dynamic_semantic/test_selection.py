from __future__ import annotations

from types import SimpleNamespace

from wfcllm.dynamic_semantic.channel import UnitEvidence
from wfcllm.dynamic_semantic.selection import select_dynamic_attempt
from wfcllm.generation.selection_v2 import RetryAttempt


PROMPT = 'def f(x):\n    """Return a value."""\n'


def _attempt(index: int, body: str, *, fallbacks: int = 0) -> RetryAttempt:
    result = SimpleNamespace(
        final_code=PROMPT + body,
        fallback_count=fallbacks,
        accepted_hit_count=0,
        closed_without_hit_count=0,
        candidate_count=0,
    )
    return RetryAttempt(attempt_index=index, seed=100 + index, result=result)


def test_selection_prefers_semantic_statistic_within_valid_quality() -> None:
    attempts = (
        _attempt(0, "    return x\n"),
        _attempt(1, "    return x + 1\n"),
    )
    evidence = {
        0: (UnitEvidence.synthetic("a", matches=3),),
        1: (UnitEvidence.synthetic("b", matches=7),),
    }

    selected = select_dynamic_attempt(
        prompt=PROMPT,
        attempts=attempts,
        evidence_by_attempt=evidence,
        erasures_by_attempt={0: 0, 1: 0},
        minimum_independent_units=1,
    )

    assert selected.attempt_index == 1
    assert selected.no_embedding is False
    assert selected.aggregate.numerator == 7


def test_invalid_high_evidence_candidate_cannot_beat_valid_candidate() -> None:
    attempts = (
        _attempt(0, "    return x\n"),
        _attempt(1, "    return (\n"),
    )
    evidence = {
        0: (UnitEvidence.synthetic("a", matches=4),),
        1: (UnitEvidence.synthetic("b", matches=7),),
    }

    selected = select_dynamic_attempt(
        prompt=PROMPT,
        attempts=attempts,
        evidence_by_attempt=evidence,
        erasures_by_attempt={0: 0, 1: 0},
        minimum_independent_units=1,
    )

    assert selected.attempt_index == 0


def test_selection_tie_breaks_by_units_erasures_then_attempt_index() -> None:
    attempts = tuple(_attempt(index, f"    return x + {index}\n") for index in range(3))
    evidence = {
        0: (UnitEvidence.synthetic("a", matches=7),),
        1: (
            UnitEvidence.synthetic("b", matches=7),
            UnitEvidence.synthetic("c", matches=7),
        ),
        2: (
            UnitEvidence.synthetic("d", matches=7),
            UnitEvidence.synthetic("e", matches=7),
        ),
    }

    selected = select_dynamic_attempt(
        prompt=PROMPT,
        attempts=attempts,
        evidence_by_attempt=evidence,
        erasures_by_attempt={0: 0, 1: 2, 2: 1},
        minimum_independent_units=1,
    )

    assert selected.attempt_index == 2


def test_no_eligible_semantic_candidate_uses_static_fallback() -> None:
    attempts = (
        _attempt(0, "    return x\n", fallbacks=2),
        _attempt(1, "    return x + 1\n", fallbacks=0),
    )

    selected = select_dynamic_attempt(
        prompt=PROMPT,
        attempts=attempts,
        evidence_by_attempt={0: (), 1: ()},
        erasures_by_attempt={0: 0, 1: 0},
        minimum_independent_units=3,
    )

    assert selected.attempt_index == 1
    assert selected.no_embedding is True
    assert selected.aggregate.independent_units == 0
