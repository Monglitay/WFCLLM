from __future__ import annotations

import pytest

from wfcllm.sawr.boundary import Candidate
from wfcllm.sawr.rules import HashEmbeddingRule, RuleRequest


def _candidate(text: str) -> Candidate:
    return Candidate(
        text=text,
        candidate_type="simple_statement",
        node_type="return_statement",
        position_id="module.foo.body",
        token_start_idx=0,
        token_count=1,
    )


def test_hash_rule_is_deterministic_for_same_request() -> None:
    rule = HashEmbeddingRule(target_accept_rate=0.5)
    request = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=(_candidate("return value"),),
        seed=17,
        final_flush=False,
    )

    first = rule.evaluate(request)
    second = rule.evaluate(request)

    assert first == second
    assert first.rule_name == "hash"
    assert first.reason.startswith("hash_fraction=")


def test_hash_rule_changes_digest_when_request_changes() -> None:
    rule = HashEmbeddingRule(target_accept_rate=0.5)
    request = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=(_candidate("return value"),),
        seed=17,
        final_flush=False,
    )
    baseline = rule.score_fraction(request)

    assert rule.score_fraction(
        RuleRequest(
            sample_id="sample-2",
            position_id=request.position_id,
            candidates=request.candidates,
            seed=request.seed,
            final_flush=request.final_flush,
        )
    ) != baseline
    assert rule.score_fraction(
        RuleRequest(
            sample_id=request.sample_id,
            position_id="module.bar.body",
            candidates=request.candidates,
            seed=request.seed,
            final_flush=request.final_flush,
        )
    ) != baseline
    assert rule.score_fraction(
        RuleRequest(
            sample_id=request.sample_id,
            position_id=request.position_id,
            candidates=(_candidate("return other_value"),),
            seed=request.seed,
            final_flush=request.final_flush,
        )
    ) != baseline


def test_hash_rule_target_accept_rate_boundaries() -> None:
    request = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=(_candidate("return value"),),
        seed=17,
        final_flush=False,
    )

    assert HashEmbeddingRule(target_accept_rate=0.0).evaluate(request).hit is False
    assert HashEmbeddingRule(target_accept_rate=1.0).evaluate(request).hit is True


def test_hash_rule_rejects_empty_candidate_group() -> None:
    rule = HashEmbeddingRule(target_accept_rate=0.5)
    request = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=(),
        seed=17,
        final_flush=False,
    )

    with pytest.raises(ValueError, match="candidates must not be empty"):
        rule.score_fraction(request)
