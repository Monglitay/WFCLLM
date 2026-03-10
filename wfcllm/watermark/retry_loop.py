"""Rejection sampling retry sub-loop for watermark embedding."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.interceptor import InterceptEvent
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import ProjectionVerifier

logger = logging.getLogger(__name__)


@dataclass
class AttemptInfo:
    """Diagnostic info for a single retry attempt."""

    sig: tuple[int, ...] | None = None
    min_margin: float = 0.0
    passed: bool = False
    text: str = ""
    no_block: bool = False


@dataclass
class RetryDiagnostics:
    """Aggregated diagnostics across all retry attempts."""

    per_attempt: list[AttemptInfo] = field(default_factory=list)
    unique_signatures: int = 0
    unique_texts: int = 0


@dataclass
class RetryResult:
    """Result of the retry sub-loop."""

    success: bool
    attempts: int
    final_event: InterceptEvent | None
    diagnostics: RetryDiagnostics


class RetryLoop:
    """Independent retry sub-loop for rejection sampling.

    Each retry:
    1. ctx.rollback(checkpoint) — atomically restore all state
    2. Free-generate until interceptor triggers a new simple block
    3. Verify the block
    4. On failure, collect penalty_ids for next attempt
    """

    def __init__(
        self,
        ctx: GenerationContext,
        config: WatermarkConfig,
        verifier: ProjectionVerifier,
        keying: WatermarkKeying,
        entropy_est: NodeEntropyEstimator,
        structural_token_ids: set[int],
    ):
        self._ctx = ctx
        self._config = config
        self._verifier = verifier
        self._keying = keying
        self._entropy_est = entropy_est
        self._structural_token_ids = structural_token_ids
        self._retry_budget = (
            config.retry_token_budget
            if config.retry_token_budget is not None
            else config.max_new_tokens // 2
        )

    def run(
        self,
        checkpoint: Checkpoint,
        original_event: InterceptEvent,
    ) -> RetryResult:
        """Run the retry loop from checkpoint.

        Args:
            checkpoint: State to rollback to before each retry.
            original_event: The failed event (used for parent_node_type to derive valid_set).

        Returns:
            RetryResult with success status and diagnostics.
        """
        parent = original_event.parent_node_type or "module"
        valid_set = self._keying.derive(parent)

        diagnostics = RetryDiagnostics(per_attempt=[])
        prev_retry_ids: list[int] | None = None
        sigs_seen: set[tuple] = set()
        texts_seen: set[str] = set()

        for attempt_i in range(self._config.max_retries):
            # Atomically restore all state
            self._ctx.rollback(checkpoint)

            # Free-generate until a new simple block
            event = self._generate_until_block(
                penalty_ids=prev_retry_ids,
            )

            if event is None:
                diagnostics.per_attempt.append(AttemptInfo(no_block=True))
                logger.debug(
                    "  [retry %d/%d] sub-loop ended without block",
                    attempt_i + 1, self._config.max_retries,
                )
                continue

            # Verify
            block_entropy = self._entropy_est.estimate_block_entropy(event.block_text)
            margin = self._entropy_est.compute_margin(block_entropy, self._config)
            result = self._verifier.verify(event.block_text, valid_set, margin)

            info = AttemptInfo(
                sig=result.lsh_signature if result.lsh_signature else None,
                min_margin=result.min_margin,
                passed=result.passed,
                text=event.block_text[:80],
            )
            diagnostics.per_attempt.append(info)
            if result.lsh_signature:
                sigs_seen.add(result.lsh_signature)
            texts_seen.add(event.block_text)

            logger.debug(
                "  [retry %d/%d] min_margin=%.4f margin_thresh=%.4f passed=%s\n  text=%r",
                attempt_i + 1, self._config.max_retries,
                result.min_margin, margin, result.passed,
                event.block_text[:80],
            )

            if result.passed:
                diagnostics.unique_signatures = len(sigs_seen)
                diagnostics.unique_texts = len(texts_seen)
                return RetryResult(
                    success=True,
                    attempts=attempt_i + 1,
                    final_event=event,
                    diagnostics=diagnostics,
                )

            # Collect penalty IDs for next attempt
            rollback_idx = len(checkpoint.generated_ids)
            retry_ids = self._ctx.generated_ids[rollback_idx:]
            prev_retry_ids = [
                tid for tid in retry_ids
                if tid not in self._structural_token_ids
            ]

        diagnostics.unique_signatures = len(sigs_seen)
        diagnostics.unique_texts = len(texts_seen)
        return RetryResult(
            success=False,
            attempts=self._config.max_retries,
            final_event=None,
            diagnostics=diagnostics,
        )

    def _generate_until_block(
        self,
        penalty_ids: list[int] | None,
    ) -> InterceptEvent | None:
        """Free-generate tokens until a simple block is detected or budget exhausted."""
        for _ in range(self._retry_budget):
            next_id = self._ctx.forward_and_sample(penalty_ids=penalty_ids)
            if next_id == self._ctx.eos_id:
                return None
            event = self._ctx.last_event
            if event is not None and event.block_type == "simple":
                return event
        return None
