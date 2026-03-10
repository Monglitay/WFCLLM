"""Cascade fallback manager for compound block re-generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.interceptor import InterceptEvent

logger = logging.getLogger(__name__)


@dataclass
class CascadeCheckpoint:
    """Compound block rollback point."""

    checkpoint: Checkpoint
    compound_event: InterceptEvent
    failed_simple_blocks: list[str] = field(default_factory=list)


class CascadeManager:
    """Manage compound block cascade fallback. Default disabled."""

    def __init__(self, config: WatermarkConfig):
        self._enabled = config.enable_cascade
        self._max_depth = config.cascade_max_depth
        self._stack: list[CascadeCheckpoint] = []

    def on_compound_block_start(
        self, ctx: GenerationContext, event: InterceptEvent
    ) -> None:
        """Save a cascade checkpoint when a compound block starts."""
        if not self._enabled:
            return
        cp = CascadeCheckpoint(
            checkpoint=ctx.checkpoint(),
            compound_event=event,
        )
        self._stack.append(cp)
        if len(self._stack) > self._max_depth:
            self._stack.pop(0)

    def on_simple_block_failed(self, block_text: str) -> None:
        """Record a retry-failed simple block."""
        if self._enabled and self._stack:
            self._stack[-1].failed_simple_blocks.append(block_text)

    def should_cascade(self) -> bool:
        """Check if cascade fallback should trigger."""
        if not self._enabled or not self._stack:
            return False
        return len(self._stack[-1].failed_simple_blocks) > 0

    def cascade(self, ctx: GenerationContext) -> CascadeCheckpoint | None:
        """Pop the stack and rollback to compound block start."""
        if not self._stack:
            return None
        cascade_cp = self._stack.pop()
        ctx.rollback(cascade_cp.checkpoint)
        logger.debug(
            "[CASCADE] rolling back to compound block %s, had %d failed simple blocks",
            cascade_cp.compound_event.node_type,
            len(cascade_cp.failed_simple_blocks),
        )
        return cascade_cp
