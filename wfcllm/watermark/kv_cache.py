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

    def rollback(
        self, past_key_values: tuple, snapshot: CacheSnapshot
    ) -> tuple:
        """Truncate KV-Cache to the snapshot's sequence length.

        Returns a new tuple of cloned truncated (key, value) pairs.
        Raises ValueError if snapshot is stale (target > current).
        Short-circuits if target == current (no truncation needed).
        """
        target_len = snapshot.seq_len
        current_len = past_key_values[0][0].shape[2]

        if target_len > current_len:
            raise ValueError(
                f"Snapshot seq_len ({target_len}) > current ({current_len}). "
                "Checkpoint may be stale or from a different generation run."
            )

        if target_len == current_len:
            return past_key_values

        new_kv = tuple(
            (k[:, :, :target_len, :].clone(), v[:, :, :target_len, :].clone())
            for k, v in past_key_values
        )

        # Explicitly release old tensors
        for k, v in past_key_values:
            del k, v
        del past_key_values

        return new_kv
