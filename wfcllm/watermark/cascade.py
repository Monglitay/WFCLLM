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
    checkpoint_key: tuple
    stats_snapshot: object | None = None
    failed_simple_blocks: list[dict[str, object]] = field(default_factory=list)

    def build_diagnostic_metadata(
        self,
        restored_stats: object | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "triggered": True,
            "compound_node_type": self.compound_event.node_type,
            "failed_simple_count_before_cascade": len(self.failed_simple_blocks),
            "replaced_block_ordinals": [
                item["block_ordinal"]
                for item in self.failed_simple_blocks
                if isinstance(item.get("block_ordinal"), int)
            ],
        }
        if isinstance(restored_stats, dict):
            metadata["restored_total_blocks"] = int(restored_stats.get("total_blocks", 0))
            metadata["restored_embedded_blocks"] = int(restored_stats.get("embedded_blocks", 0))
            metadata["restored_failed_blocks"] = int(restored_stats.get("failed_blocks", 0))
        return metadata

    def build_replacement_scope(self) -> dict[str, object]:
        return {
            "checkpoint_key": list(self.checkpoint_key),
            "compound_node_type": self.compound_event.node_type,
            "compound_parent_node_type": self.compound_event.parent_node_type or "module",
            "replaced_block_ordinals": [
                item["block_ordinal"]
                for item in self.failed_simple_blocks
                if isinstance(item.get("block_ordinal"), int)
            ],
        }


class CascadeManager:
    """Manage compound block cascade fallback. Default disabled."""

    def __init__(self, config: WatermarkConfig):
        self._enabled = config.enable_cascade
        self._max_depth = config.cascade_max_depth
        self._stack: list[CascadeCheckpoint] = []
        self._retired_keys: set[tuple] = set()

    def _checkpoint_key(self, checkpoint: Checkpoint) -> tuple:
        """Build a stable key for the compound block start checkpoint.

        Compound blocks are re-emitted multiple times while their body grows,
        but their block-start checkpoint stays the same. Deduplicating by this
        checkpoint prevents repeated cascade rollbacks on the same logical block.
        """
        generated_ids = getattr(checkpoint, "generated_ids", None)
        generated_text = getattr(checkpoint, "generated_text", None)
        if not isinstance(generated_ids, list) or not isinstance(generated_text, str):
            return ("checkpoint-object", id(checkpoint))

        kv_snapshot = getattr(checkpoint, "kv_snapshot", None)
        interceptor_state = getattr(checkpoint, "interceptor_state", None)
        return (
            tuple(generated_ids),
            generated_text,
            getattr(kv_snapshot, "seq_len", None),
            getattr(interceptor_state, "token_idx", None),
        )

    def on_compound_block_start(
        self,
        ctx: GenerationContext,
        event: InterceptEvent,
        stats_snapshot: object | None = None,
    ) -> None:
        """Save a cascade checkpoint when a compound block starts."""
        if not self._enabled:
            return
        # Use last_block_checkpoint (state BEFORE the compound block's first
        # token) rather than ctx.checkpoint() (state AFTER the last token).
        # ctx.checkpoint() captures the compound block END, but the retry loop
        # may later roll ctx back to an earlier simple-block start — a position
        # with fewer KV tokens — making the "end" snapshot stale and causing
        # rollback to raise ValueError.
        block_cp = ctx.last_block_checkpoint
        if block_cp is None:
            return
        checkpoint_key = self._checkpoint_key(block_cp)
        if checkpoint_key in self._retired_keys:
            return
        if any(item.checkpoint_key == checkpoint_key for item in self._stack):
            return
        cp = CascadeCheckpoint(
            checkpoint=block_cp,
            compound_event=event,
            checkpoint_key=checkpoint_key,
            stats_snapshot=stats_snapshot,
        )
        self._stack.append(cp)
        if len(self._stack) > self._max_depth:
            self._stack.pop(0)

    def on_simple_block_failed(
        self,
        block_text: str,
        block_ordinal: int | None = None,
    ) -> None:
        """Record a retry-failed simple block."""
        if self._enabled and self._stack:
            self._stack[-1].failed_simple_blocks.append(
                {
                    "block_text": block_text,
                    "block_ordinal": block_ordinal,
                }
            )

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
        self._retired_keys.add(cascade_cp.checkpoint_key)
        ctx.rollback(cascade_cp.checkpoint)
        logger.debug(
            "[CASCADE] rolling back to compound block %s, had %d failed simple blocks",
            cascade_cp.compound_event.node_type,
            len(cascade_cp.failed_simple_blocks),
        )
        return cascade_cp
