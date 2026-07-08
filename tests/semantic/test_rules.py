from __future__ import annotations

import pytest

from wfcllm.sawr import (
    Candidate,
    EmbeddingRule,
    HashEmbeddingRule,
    RuleDecision,
    RuleRequest,
    SemanticLshEmbeddingRule,
)


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
    assert "target_accept_rate=" in first.reason
    assert EmbeddingRule.__name__ == "EmbeddingRule"
    assert isinstance(first, RuleDecision)


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


def test_hash_rule_changes_digest_when_seed_or_final_flush_changes() -> None:
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
            sample_id=request.sample_id,
            position_id=request.position_id,
            candidates=request.candidates,
            seed=18,
            final_flush=request.final_flush,
        )
    ) != baseline
    assert rule.score_fraction(
        RuleRequest(
            sample_id=request.sample_id,
            position_id=request.position_id,
            candidates=request.candidates,
            seed=request.seed,
            final_flush=True,
        )
    ) != baseline


def test_hash_rule_payload_framing_handles_separator_characters() -> None:
    rule = HashEmbeddingRule(target_accept_rate=0.5)
    first = RuleRequest(
        sample_id="a\x1fb",
        position_id="c",
        candidates=(_candidate("d"),),
        seed=1,
        final_flush=False,
    )
    second = RuleRequest(
        sample_id="a",
        position_id="b\x1fc",
        candidates=(_candidate("d"),),
        seed=1,
        final_flush=False,
    )

    assert rule._payload(first) != rule._payload(second)


def test_hash_rule_payload_framing_preserves_candidate_group_structure() -> None:
    rule = HashEmbeddingRule(target_accept_rate=0.5)
    first = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=(_candidate("return x"), _candidate("return y")),
        seed=17,
        final_flush=False,
    )
    second = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=(_candidate("return x\nreturn y"),),
        seed=17,
        final_flush=False,
    )

    assert rule._payload(first) != rule._payload(second)


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


class _FakeVerifyResult:
    def __init__(
        self,
        *,
        passed: bool,
        lsh_signature: tuple[int, ...],
        min_margin: float,
        in_valid_set: bool,
    ) -> None:
        self.passed = passed
        self.lsh_signature = lsh_signature
        self.min_margin = min_margin
        self.in_valid_set = in_valid_set


class _FakeVerifier:
    def __init__(self, result: _FakeVerifyResult) -> None:
        self.result = result
        self.calls: list[tuple[str, frozenset[tuple[int, ...]], float]] = []

    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> _FakeVerifyResult:
        self.calls.append((code_text, valid_set, margin))
        return self.result


class _FakeKeying:
    def __init__(self) -> None:
        self.valid_set = frozenset({(1, 0, 1, 0)})
        self.calls: list[tuple[str, int, int | None]] = []

    def derive(
        self,
        parent_node_type: str,
        *,
        k: int,
        ordinal: int | None,
    ) -> frozenset[tuple[int, ...]]:
        self.calls.append((parent_node_type, k, ordinal))
        return self.valid_set


def test_semantic_lsh_rule_verifies_candidate_group_in_lsh_space() -> None:
    verifier = _FakeVerifier(
        _FakeVerifyResult(
            passed=True,
            lsh_signature=(1, 0, 1, 0),
            min_margin=0.42,
            in_valid_set=True,
        )
    )
    keying = _FakeKeying()
    rule = SemanticLshEmbeddingRule(
        verifier=verifier,
        keying=keying,
        lsh_d=4,
        lsh_gamma=0.75,
        margin=0.05,
        use_ordinal_keying=True,
    )
    first = _candidate("x = 1")
    second = _candidate("return x")
    candidates = (
        Candidate(
            text=first.text,
            candidate_type=first.candidate_type,
            node_type=first.node_type,
            position_id=first.position_id,
            token_start_idx=first.token_start_idx,
            token_count=first.token_count,
            parent_node_type="function_definition",
            ordinal=7,
        ),
        second,
    )
    request = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=candidates,
        seed=17,
        final_flush=False,
    )

    decision = rule.evaluate(request)

    assert decision.hit is True
    assert decision.rule_name == "semantic_lsh"
    assert keying.calls == [("function_definition", 12, 7)]
    assert verifier.calls == [("x = 1\nreturn x", keying.valid_set, 0.05)]
    assert "lsh_signature=(1, 0, 1, 0)" in decision.reason
    assert "in_valid_set=True" in decision.reason
    assert "min_margin=0.420000000000" in decision.reason
    assert "gamma_effective=0.750000000000" in decision.reason


def test_semantic_lsh_rule_omits_ordinal_keying_by_default() -> None:
    verifier = _FakeVerifier(
        _FakeVerifyResult(
            passed=True,
            lsh_signature=(1, 0, 1, 0),
            min_margin=0.42,
            in_valid_set=True,
        )
    )
    keying = _FakeKeying()
    rule = SemanticLshEmbeddingRule(
        verifier=verifier,
        keying=keying,
        lsh_d=4,
        lsh_gamma=0.75,
    )
    candidate = Candidate(
        text="return value",
        candidate_type="simple_statement",
        node_type="return_statement",
        position_id="module.foo.body",
        token_start_idx=0,
        token_count=1,
        parent_node_type="function_definition",
        ordinal=7,
    )

    rule.evaluate(
        RuleRequest(
            sample_id="sample-1",
            position_id="module.foo.body",
            candidates=(candidate,),
            seed=17,
            final_flush=False,
        )
    )

    assert keying.calls == [("function_definition", 12, None)]


def test_semantic_lsh_rule_reports_miss_from_verifier() -> None:
    verifier = _FakeVerifier(
        _FakeVerifyResult(
            passed=False,
            lsh_signature=(0, 0, 0, 1),
            min_margin=0.01,
            in_valid_set=False,
        )
    )
    rule = SemanticLshEmbeddingRule(
        verifier=verifier,
        keying=_FakeKeying(),
        lsh_d=4,
        lsh_gamma=0.75,
        margin=0.05,
    )
    request = RuleRequest(
        sample_id="sample-1",
        position_id="module.foo.body",
        candidates=(_candidate("return value"),),
        seed=17,
        final_flush=True,
    )

    decision = rule.evaluate(request)

    assert decision.hit is False
    assert decision.rule_name == "semantic_lsh"
    assert "in_valid_set=False" in decision.reason


def test_semantic_lsh_rule_rejects_empty_candidate_group() -> None:
    rule = SemanticLshEmbeddingRule(
        verifier=_FakeVerifier(
            _FakeVerifyResult(
                passed=True,
                lsh_signature=(1, 0, 1, 0),
                min_margin=0.42,
                in_valid_set=True,
            )
        ),
        keying=_FakeKeying(),
        lsh_d=4,
        lsh_gamma=0.75,
    )

    with pytest.raises(ValueError, match="candidates must not be empty"):
        rule.evaluate(
            RuleRequest(
                sample_id="sample-1",
                position_id="module.foo.body",
                candidates=(),
                seed=17,
                final_flush=False,
            )
        )
