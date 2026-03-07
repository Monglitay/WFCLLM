"""Watermark key derivation from secret key and AST topology."""

from __future__ import annotations

import hashlib
import hmac

import torch


class WatermarkKeying:
    """Derive deterministic (direction vector, target bit) from key + topology."""

    def __init__(self, secret_key: str, embed_dim: int):
        self._key = secret_key.encode("utf-8")
        self._embed_dim = embed_dim

    def derive(self, parent_node_type: str, node_type: str) -> tuple[torch.Tensor, int]:
        """Derive (v, t) from local AST topology.

        Args:
            parent_node_type: AST type of the parent node.
            node_type: AST type of the current statement block node.

        Returns:
            v: Unit vector in R^embed_dim (float32).
            t: Target bit in {0, 1}.
        """
        # 1. HMAC-SHA256 of topology feature
        message = f"{parent_node_type}|{node_type}".encode("utf-8")
        digest = hmac.new(self._key, message, hashlib.sha256).digest()

        # 2. Seed PRNG with hash
        seed = int.from_bytes(digest[:8], "big")
        gen = torch.Generator()
        gen.manual_seed(seed)

        # 3. Generate direction vector from standard normal, then normalize
        v = torch.randn(self._embed_dim, generator=gen)
        v = v / v.norm()

        # 4. Target bit from last byte LSB
        t = digest[-1] & 1

        return v, t
