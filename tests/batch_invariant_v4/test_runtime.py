from __future__ import annotations

import hashlib

import pytest

from wfcllm.batch_invariant_v4.cache import PublicContextCache, public_cache_key
from wfcllm.batch_invariant_v4.context import ContextConfig, StructuralContextExtractor
from wfcllm.batch_invariant_v4.keying import V4SecretKey
from wfcllm.batch_invariant_v4.runtime import CandidateRuntime, RawCandidate


def _candidate(task: str, attempt: int, code: str | None = None) -> RawCandidate:
    final_code = code or f"def f():\n    value = {attempt}\n    return value\n"
    return RawCandidate(
        task_id=task,
        attempt_index=attempt,
        final_code=final_code,
        final_code_sha256=hashlib.sha256(final_code.encode()).hexdigest(),
        quality_tier=1,
        valid=True,
    )


def test_cache_key_accepts_only_public_config_and_context_identity() -> None:
    key = public_cache_key(
        schema_version="wfcllm-batch-invariant-structural-context/v4",
        public_config_sha256="a" * 64,
        context_sha256="b" * 64,
    )
    assert len(key) == 64
    with pytest.raises(TypeError, match="secret"):
        public_cache_key(  # type: ignore[call-arg]
            schema_version="wfcllm-batch-invariant-structural-context/v4",
            public_config_sha256="a" * 64,
            context_sha256="b" * 64,
            secret_key=V4SecretKey.from_material_for_test(b"x" * 32),
        )


def test_cache_hit_miss_flush_order_and_other_candidates_do_not_change_value() -> None:
    extractor = StructuralContextExtractor(ContextConfig())
    context = extractor.extract("def f():\n    return 1\n").contexts[0]
    cache = PublicContextCache(public_config_sha256="a" * 64)

    miss = cache.get_or_create(context, lambda: context.representation_bytes)
    hit = cache.get_or_create(context, lambda: (_ for _ in ()).throw(AssertionError()))
    cache.flush_order(("candidate-z", "candidate-a"))
    after_flush = cache.get_or_create(context, lambda: context.representation_bytes)

    assert miss == hit == after_flush == context.representation_bytes
    assert cache.hits == 2
    assert cache.misses == 1


def test_runtime_preserves_pool_order_and_replays_selected_exactly_once() -> None:
    runtime = CandidateRuntime.for_test(
        key=V4SecretKey.from_material_for_test(b"r" * 32),
        public_config_sha256="c" * 64,
        minimum_independent_units=1,
    )
    candidates = tuple(_candidate("HumanEval/1", index) for index in range(3))

    result = runtime.select(candidates, retry=3)
    replay = runtime.replay_selected(result.selected.final_code)

    assert result.input_pool_sha256 == result.output_pool_sha256
    assert result.candidate_pool_match_rate == 1.0
    assert result.selected_generation_evidence == replay
    assert runtime.selected_final_replay_count == 1
    assert runtime.eos_all_candidate_neural_rescore_count == 0
    with pytest.raises(ValueError, match="exactly once"):
        runtime.replay_selected(result.selected.final_code)


def test_candidate_pool_requires_retry_cardinality_order_hash_and_validity() -> None:
    runtime = CandidateRuntime.for_test(
        key=V4SecretKey.from_material_for_test(b"p" * 32),
        public_config_sha256="d" * 64,
        minimum_independent_units=1,
    )
    candidates = tuple(_candidate("HumanEval/2", index) for index in range(3))
    with pytest.raises(ValueError, match="exactly 3"):
        runtime.select(candidates[:-1], retry=3)
    with pytest.raises(ValueError, match="ordered"):
        runtime.select((candidates[1], candidates[0], candidates[2]), retry=3)
    malformed = RawCandidate(
        task_id="HumanEval/2",
        attempt_index=0,
        final_code="def f():\n return 1\n",
        final_code_sha256="0" * 64,
        quality_tier=1,
        valid=True,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        runtime.select((malformed,), retry=1)


def test_invalid_candidate_is_never_resurrected_by_secret_score() -> None:
    runtime = CandidateRuntime.for_test(
        key=V4SecretKey.from_material_for_test(b"v" * 32),
        public_config_sha256="e" * 64,
        minimum_independent_units=1,
    )
    valid = _candidate("HumanEval/3", 0)
    invalid_base = _candidate(
        "HumanEval/3",
        1,
        "def f():\n    a = 1\n    b = 2\n    return a + b\n",
    )
    invalid = RawCandidate(
        **{**invalid_base.__dict__, "valid": False, "quality_tier": 99}
    )

    result = runtime.select((valid, invalid), retry=2)

    assert result.selected.attempt_index == 0


def test_all_ineligible_pool_uses_frozen_current_quality_fallback() -> None:
    runtime = CandidateRuntime.for_test(
        key=V4SecretKey.from_material_for_test(b"f" * 32),
        public_config_sha256="f" * 64,
        minimum_independent_units=1,
    )
    candidates = []
    for attempt, tier, fallback_count in (
        (0, 1, 0),
        (1, 2, 3),
        (2, 2, 1),
        (3, 2, 1),
    ):
        base = _candidate("HumanEval/5", attempt)
        candidates.append(
            RawCandidate(
                **{
                    **base.__dict__,
                    "valid": False,
                    "quality_tier": tier,
                    "fallback_count": fallback_count,
                }
            )
        )

    result = runtime.select(tuple(candidates), retry=4)

    assert result.selected.attempt_index == 2


def test_raw_candidate_repr_does_not_contain_secret() -> None:
    candidate = _candidate("HumanEval/4", 0)
    assert "secret" not in repr(candidate).lower()
    assert set(candidate.__dict__) == {
        "task_id",
        "attempt_index",
        "final_code",
        "final_code_sha256",
        "quality_tier",
        "valid",
        "fallback_count",
    }
