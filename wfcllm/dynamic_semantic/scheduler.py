from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wfcllm.dynamic_semantic.channel import SemanticChannel, UnitEvidence
from wfcllm.dynamic_semantic.config import SchedulerConfig
from wfcllm.dynamic_semantic.context import CanonicalContext


@dataclass(frozen=True)
class _QueuedContext:
    attempt_index: int
    closure_index: int
    context: CanonicalContext


class DynamicSemanticScheduler:
    """Deterministic cross-attempt micro-batching for closure-time contexts."""

    def __init__(
        self,
        *,
        encoder: Any,
        whitening: Any,
        channel: SemanticChannel,
        config: SchedulerConfig,
    ) -> None:
        self._encoder = encoder
        self._whitening = whitening
        self._channel = channel
        self._config = config
        self._queue: list[_QueuedContext] = []
        self._pending_hashes: set[str] = set()
        self._evidence_cache: dict[str, UnitEvidence] = {}
        self._live_by_attempt: dict[int, tuple[tuple[str, str], ...]] = {}
        self._attempts_since_flush = 0
        self._completed_attempts: set[int] = set()
        self.encoded_batch_sizes: list[int] = []

    @property
    def encoder_calls(self) -> int:
        return int(self._encoder.encoder_calls)

    @property
    def encoded_contexts(self) -> int:
        return int(self._encoder.encoded_contexts)

    @property
    def pending_contexts(self) -> int:
        return len(self._queue)

    def submit(
        self,
        *,
        attempt_index: int,
        closure_index: int,
        context: CanonicalContext,
    ) -> None:
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise ValueError("attempt_index must be a non-negative integer")
        if (
            isinstance(closure_index, bool)
            or not isinstance(closure_index, int)
            or closure_index < 0
        ):
            raise ValueError("closure_index must be a non-negative integer")
        if not isinstance(context, CanonicalContext):
            raise ValueError("context must be CanonicalContext")
        context_hash = context.context_sha256
        if context_hash in self._evidence_cache or context_hash in self._pending_hashes:
            return
        self._queue.append(
            _QueuedContext(
                attempt_index=attempt_index,
                closure_index=closure_index,
                context=context,
            )
        )
        self._pending_hashes.add(context_hash)
        if len(self._queue) >= self._config.target_batch_contexts:
            self._flush_one()

    def update_live_contexts(
        self,
        attempt_index: int,
        contexts: tuple[CanonicalContext, ...],
    ) -> None:
        if isinstance(contexts, list):
            contexts = tuple(contexts)
        if not isinstance(contexts, tuple) or any(
            not isinstance(context, CanonicalContext) for context in contexts
        ):
            raise ValueError("contexts must be CanonicalContext records")
        unit_ids = [context.unit_id for context in contexts]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("live contexts contain duplicate unit IDs")
        self._live_by_attempt[attempt_index] = tuple(
            (context.unit_id, context.context_sha256) for context in contexts
        )

    def attempt_completed(self, attempt_index: int) -> None:
        if attempt_index in self._completed_attempts:
            raise ValueError("attempt_completed called more than once")
        self._completed_attempts.add(attempt_index)
        self._attempts_since_flush += 1
        if (
            self._queue
            and self._attempts_since_flush
            >= self._config.max_queue_completed_attempts
        ):
            self._flush_one()

    def flush_final(self) -> None:
        while self._queue:
            self._flush_one()

    def evidence_for_attempt(self, attempt_index: int) -> tuple[UnitEvidence, ...]:
        live = self._live_by_attempt.get(attempt_index, ())
        return tuple(
            self._evidence_cache[context_hash]
            for _, context_hash in live
            if context_hash in self._evidence_cache
        )

    def _flush_one(self) -> None:
        if not self._queue:
            return
        self._queue.sort(
            key=lambda item: (item.attempt_index, item.closure_index)
        )
        count = min(len(self._queue), self._config.max_batch_contexts)
        batch = self._queue[:count]
        del self._queue[:count]
        contexts = tuple(item.context.serialized for item in batch)
        embeddings = self._encoder.encode(contexts)
        whitened = self._whitening.transform(embeddings)
        if len(whitened) != len(batch):
            raise ValueError("whitening output row count does not match batch")
        for queued, vector in zip(batch, whitened, strict=True):
            context = queued.context
            evidence = self._channel.score(
                context.unit_id,
                context.context_sha256,
                vector.tolist(),
            )
            self._evidence_cache[context.context_sha256] = evidence
            self._pending_hashes.remove(context.context_sha256)
        self.encoded_batch_sizes.append(len(batch))
        self._attempts_since_flush = 0
