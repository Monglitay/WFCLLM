"""Watermark key derivation: valid LSH region set from AST topology."""

from __future__ import annotations

import hashlib
import hmac


class WatermarkKeying:
    """Derive the valid LSH signature set G from secret key and parent node type.

    The seed uses ONLY parent_node_type (not the current node's type) so that
    semantically equivalent transformations of a block do not change its target region.
    """

    def __init__(self, secret_key: str, d: int, gamma: float):
        self._key = secret_key.encode("utf-8")
        self._d = d
        self._gamma = gamma

    def derive(self, parent_node_type: str) -> frozenset[tuple[int, ...]]:
        """Return valid LSH signature set G for a block with given parent node type.

        Args:
            parent_node_type: AST type of the parent node (e.g. "module", "for_statement").

        Returns:
            frozenset of d-bit tuples that constitute the valid region set G.
            A block passes the watermark check iff its LSH signature is in G.
        """
        message = parent_node_type.encode("utf-8")
        digest = hmac.new(self._key, message, hashlib.sha256).digest()

        seed = int.from_bytes(digest[:8], "big")

        # Enumerate all 2^d possible signatures
        all_sigs = [
            tuple(int(b) for b in format(i, f"0{self._d}b"))
            for i in range(2 ** self._d)
        ]

        # Deterministic Fisher-Yates shuffle using seed
        import random
        rng = random.Random(seed)
        shuffled = list(all_sigs)
        rng.shuffle(shuffled)

        k = round(self._gamma * len(all_sigs))
        return frozenset(shuffled[:k])
