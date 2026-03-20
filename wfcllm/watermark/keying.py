"""Watermark key derivation: valid LSH region set from AST topology."""

from __future__ import annotations

import hashlib
import hmac

_MISSING = object()


class WatermarkKeying:
    """Derive the valid LSH signature set G from secret key and parent node type.

    The seed uses ONLY parent_node_type (not the current node's type) so that
    semantically equivalent transformations of a block do not change its target region.
    """

    def __init__(self, secret_key: str, d: int, gamma: float | None = None):
        self._key = secret_key.encode("utf-8")
        self._d = d
        self._legacy_gamma = gamma

    def derive(self, parent_node_type: str, k: int | object = _MISSING) -> frozenset[tuple[int, ...]]:
        """Return valid LSH signature set G for a block with given parent node type.

        Args:
            parent_node_type: AST type of the parent node (e.g. "module", "for_statement").

            k: Number of valid regions to derive. Must satisfy 1 <= k < 2**d.
                If omitted, legacy mode requires constructor `gamma` and uses
                `round(gamma * 2**d)`.

        Returns:
            frozenset of d-bit tuples that constitute the valid region set G.
            A block passes the watermark check iff its LSH signature is in G.
        """
        max_regions = 2 ** self._d
        if k is _MISSING:
            if self._legacy_gamma is None:
                raise TypeError("k is required when legacy gamma is not configured")
            k = round(self._legacy_gamma * max_regions)
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int")
        if not (1 <= k < max_regions):
            raise ValueError("k must satisfy 1 <= k < 2**d")

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

        return frozenset(shuffled[:k])
