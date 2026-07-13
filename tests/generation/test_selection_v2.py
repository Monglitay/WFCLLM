from __future__ import annotations

from types import SimpleNamespace

import pytest

from wfcllm.generation.selection_v2 import (
    V2_RETRY_LEDGER_SCHEMA_VERSION,
    RetryAttempt,
    V2RetryAttemptSelector,
)


PROMPT = "def target(x):\n    \"\"\"Return x.\"\"\"\n"


def _result(final_code: str, *, accepted: int = 0, fallback: int = 0):
    return SimpleNamespace(
        final_code=final_code,
        accepted_hit_count=accepted,
        closed_without_hit_count=0,
        fallback_count=fallback,
        candidate_count=1,
        audit_events=[],
    )


def _score(value: float, unit_count: int = 1):
    return SimpleNamespace(
        raw_score=value,
        unit_count=unit_count,
        duplicate_count=0,
        total_bits=16 * unit_count,
        matched_bits=round(16 * unit_count * value),
        unit_evidence=(),
    )


class _FakeScorer:
    def __init__(self, scores: dict[str, object]) -> None:
        self._scores = scores

    def score_code(self, code: str):
        return self._scores[code]


def test_selector_never_prefers_invalid_candidate_for_score() -> None:
    invalid = PROMPT + "    return (\n"
    valid = PROMPT + "    return x\n"
    selector = V2RetryAttemptSelector(
        scorer=_FakeScorer({invalid: _score(1.0), valid: _score(0.6)}),
    )

    selected = selector.select(
        sample_id="HumanEval/0",
        prompt=PROMPT,
        attempts=(
            RetryAttempt(0, 20260713, _result(invalid)),
            RetryAttempt(1, 20260814, _result(valid)),
        ),
    )

    assert selected.attempt_index == 1
    assert selected.result.final_code == valid
    assert selected.no_embedding is False
    assert selected.generation_score == pytest.approx(0.6)
    assert selected.recovered_score == pytest.approx(0.6)
    assert selected.replay_equal is True


def test_selector_uses_detector_score_then_units_fallbacks_and_index() -> None:
    first = PROMPT + "    value = x\n    return value\n"
    second = PROMPT + "    result = x\n    return result\n"
    third = PROMPT + "    answer = x\n    return answer\n"
    scorer = _FakeScorer(
        {
            first: _score(0.7, 1),
            second: _score(0.7, 2),
            third: _score(0.7, 2),
        }
    )
    selector = V2RetryAttemptSelector(scorer=scorer)

    selected = selector.select(
        sample_id="HumanEval/1",
        prompt=PROMPT,
        attempts=(
            RetryAttempt(0, 1, _result(first, fallback=0)),
            RetryAttempt(1, 2, _result(second, fallback=1)),
            RetryAttempt(2, 3, _result(third, fallback=0)),
        ),
    )

    assert selected.attempt_index == 2


def test_selector_all_invalid_falls_back_without_embedding() -> None:
    malformed = PROMPT + "    return (\n"
    placeholder = PROMPT + "    pass\n"
    selector = V2RetryAttemptSelector(
        scorer=_FakeScorer({malformed: _score(1.0), placeholder: _score(0.9)}),
    )

    selected = selector.select(
        sample_id="HumanEval/2",
        prompt=PROMPT,
        attempts=(
            RetryAttempt(0, 1, _result(malformed)),
            RetryAttempt(1, 2, _result(placeholder)),
        ),
    )

    assert selected.attempt_index == 1
    assert selected.no_embedding is True


def test_selector_emits_audit_only_v2_ledger_without_secret() -> None:
    code = PROMPT + "    return x\n"
    selector = V2RetryAttemptSelector(scorer=_FakeScorer({code: _score(0.75)}))

    selected = selector.select(
        sample_id="HumanEval/3",
        prompt=PROMPT,
        attempts=(RetryAttempt(0, 20260713, _result(code, accepted=2)),),
    )

    assert len(selected.ledger_rows) == 1
    row = selected.ledger_rows[0]
    assert row["schema_version"] == V2_RETRY_LEDGER_SCHEMA_VERSION
    assert row["audit_only"] is True
    assert row["detector_input_allowed"] is False
    assert row["selected"] is True
    assert row["generation_score"] == pytest.approx(0.75)
    assert row["recovered_score"] == pytest.approx(0.75)
    assert "secret_key" not in str(row)
    assert row["final_code"] == code


def test_selector_requires_at_least_one_attempt() -> None:
    selector = V2RetryAttemptSelector(scorer=_FakeScorer({}))

    with pytest.raises(ValueError, match="attempts"):
        selector.select(sample_id="HumanEval/4", prompt=PROMPT, attempts=())
