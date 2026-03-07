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
        # All layers have same seq_len; read from first layer's key tensor
        seq_len = past_key_values[0][0].shape[2]
        return CacheSnapshot(seq_len=seq_len)

    def rollback(
        self, past_key_values: tuple, snapshot: CacheSnapshot
    ) -> tuple:
        """Truncate KV-Cache to the snapshot's sequence length.

        Returns a new tuple of truncated (key, value) pairs.
        """
        target_len = snapshot.seq_len
        return tuple(
            (k[:, :, :target_len, :], v[:, :, :target_len, :])
            for k, v in past_key_values
        )
