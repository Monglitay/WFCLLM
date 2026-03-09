"""KV-Cache snapshot and rollback for rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheSnapshot:
    """Records the sequence length at snapshot time."""

    seq_len: int


class KVCacheManager:
    """Manage KV-Cache snapshots and rollbacks via truncation."""

    def snapshot(self, past_key_values: tuple) -> CacheSnapshot:
        """Record current sequence length from the KV-Cache.

        Args:
            past_key_values: Tuple of (key, value) tensor pairs per layer.
                Each tensor has shape (batch, heads, seq_len, head_dim).
        """
        seq_len = past_key_values[0][0].shape[2]
        return CacheSnapshot(seq_len=seq_len)

    def snapshot_at(
        self,
        past_key_values: tuple,
        rollback_idx: int,
        current_generated_count: int,
    ) -> CacheSnapshot:
        """Record KV-Cache seq_len corresponding to a past token position.

        Args:
            past_key_values: Current KV-Cache after all tokens generated so far.
            rollback_idx: len(generated_ids) at the desired rollback point.
            current_generated_count: Current len(generated_ids).
        """
        current_seq_len = past_key_values[0][0].shape[2]
        tokens_to_remove = current_generated_count - rollback_idx
        target_len = max(0, current_seq_len - tokens_to_remove)
        return CacheSnapshot(seq_len=target_len)

    def rollback(
        self, past_key_values: tuple, snapshot: CacheSnapshot
    ) -> tuple:
        """Truncate KV-Cache to the snapshot's sequence length.

        Returns a new tuple of cloned truncated (key, value) pairs so that
        the original tensors can be freed by the garbage collector.
        """
        target_len = snapshot.seq_len
        return tuple(
            (k[:, :, :target_len, :].clone(), v[:, :, :target_len, :].clone())
            for k, v in past_key_values
        )
