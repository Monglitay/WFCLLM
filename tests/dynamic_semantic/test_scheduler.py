from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from wfcllm.dynamic_semantic.channel import SemanticChannel
from wfcllm.dynamic_semantic.config import ChannelConfig, SchedulerConfig
from wfcllm.dynamic_semantic.context import CanonicalContext
from wfcllm.dynamic_semantic.keying import SecretKey
from wfcllm.dynamic_semantic.scheduler import DynamicSemanticScheduler


class _FakeEncoder:
    def __init__(self) -> None:
        self.encoder_calls = 0
        self.encoded_contexts = 0
        self.batches: list[tuple[str, ...]] = []

    def encode(self, contexts: tuple[str, ...]) -> torch.Tensor:
        self.encoder_calls += 1
        self.encoded_contexts += len(contexts)
        self.batches.append(tuple(contexts))
        rows = []
        for context in contexts:
            value = float(sum(context.encode("utf-8")) % 17 + 1)
            rows.append((value, value + 1, value + 2, value + 3))
        return torch.tensor(rows, dtype=torch.float32)


class _IdentityWhitening:
    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)


def _context(index: int) -> CanonicalContext:
    return CanonicalContext(
        unit_id=f"unit-{index}",
        role="Assign|FunctionDef|body",
        canonical_previous="<BOS>",
        canonical_current=f"x = {index}",
        serialized=f"context-{index}",
        context_sha256=f"hash-{index}",
        group_index=0,
        unit_position=index,
        is_last_in_group=False,
    )


def _scheduler(
    *,
    target: int = 32,
    maximum: int = 128,
    attempts: int = 4,
) -> tuple[DynamicSemanticScheduler, _FakeEncoder]:
    encoder = _FakeEncoder()
    channel = SemanticChannel(
        SecretKey.from_material_for_test(b"s" * 32),
        ChannelConfig(
            whitening_dimensions=4,
            quantization_scale=32,
            projection_rows=7,
            target_data_bits=4,
            minimum_independent_units=1,
        ),
    )
    scheduler = DynamicSemanticScheduler(
        encoder=encoder,
        whitening=_IdentityWhitening(),
        channel=channel,
        config=SchedulerConfig(
            target_batch_contexts=target,
            max_batch_contexts=maximum,
            max_queue_completed_attempts=attempts,
        ),
    )
    return scheduler, encoder


def test_scheduler_orders_batch_by_attempt_and_closure() -> None:
    scheduler, encoder = _scheduler(target=8)
    scheduler.submit(attempt_index=2, closure_index=1, context=_context(21))
    scheduler.submit(attempt_index=0, closure_index=2, context=_context(2))
    scheduler.submit(attempt_index=0, closure_index=1, context=_context(1))

    scheduler.flush_final()

    assert encoder.batches == [("context-1", "context-2", "context-21")]


def test_scheduler_flushes_at_target_and_final_residual() -> None:
    scheduler, encoder = _scheduler(target=2, maximum=4)

    scheduler.submit(attempt_index=0, closure_index=0, context=_context(0))
    assert encoder.encoder_calls == 0
    scheduler.submit(attempt_index=0, closure_index=1, context=_context(1))
    assert encoder.encoder_calls == 1
    scheduler.submit(attempt_index=0, closure_index=2, context=_context(2))
    scheduler.flush_final()

    assert [len(batch) for batch in encoder.batches] == [2, 1]


def test_scheduler_flushes_after_attempt_latency_bound() -> None:
    scheduler, encoder = _scheduler(target=32, attempts=2)
    scheduler.submit(attempt_index=0, closure_index=0, context=_context(0))

    scheduler.attempt_completed(0)
    assert encoder.encoder_calls == 0
    scheduler.attempt_completed(1)

    assert encoder.encoder_calls == 1


def test_live_context_update_removes_rolled_back_evidence() -> None:
    scheduler, _ = _scheduler(target=1)
    first = _context(0)
    second = _context(1)
    scheduler.submit(attempt_index=0, closure_index=0, context=first)
    scheduler.update_live_contexts(0, (first,))
    assert [item.unit_id for item in scheduler.evidence_for_attempt(0)] == ["unit-0"]

    scheduler.update_live_contexts(0, (second,))
    assert scheduler.evidence_for_attempt(0) == ()
    scheduler.submit(attempt_index=0, closure_index=1, context=second)
    assert [item.unit_id for item in scheduler.evidence_for_attempt(0)] == ["unit-1"]


def test_context_hash_cache_is_shared_across_attempts() -> None:
    scheduler, encoder = _scheduler(target=8)
    context = _context(0)
    scheduler.submit(attempt_index=0, closure_index=0, context=context)
    scheduler.submit(attempt_index=1, closure_index=0, context=context)
    scheduler.update_live_contexts(0, (context,))
    scheduler.update_live_contexts(1, (context,))
    scheduler.flush_final()

    assert encoder.encoded_contexts == 1
    assert len(scheduler.evidence_for_attempt(0)) == 1
    assert len(scheduler.evidence_for_attempt(1)) == 1


def test_one_hundred_contexts_use_four_bounded_batches() -> None:
    scheduler, encoder = _scheduler(target=32, maximum=128)
    for index in range(100):
        scheduler.submit(
            attempt_index=index // 5,
            closure_index=index % 5,
            context=_context(index),
        )
    scheduler.flush_final()

    assert [len(batch) for batch in encoder.batches] == [32, 32, 32, 4]
    assert max(len(batch) for batch in encoder.batches) <= 128


def test_scheduler_has_no_complete_code_scoring_api() -> None:
    scheduler, _ = _scheduler()

    assert not hasattr(scheduler, "score_code")
    assert not hasattr(scheduler, "score_codes")
    with pytest.raises(ValueError, match="CanonicalContext"):
        scheduler.submit(  # type: ignore[arg-type]
            attempt_index=0,
            closure_index=0,
            context="def f(): pass",
        )
