"""Global LSH hyperplane space for watermark embedding and extraction."""

from __future__ import annotations

import hashlib
import hmac

import torch
import torch.nn.functional as F


class LSHSpace:
    """Manage d global hyperplanes derived from a secret key.

    The hyperplanes statically partition the semantic embedding space into
    2^d regions identified by binary LSH signatures.
    """

    def __init__(self, secret_key: str, embed_dim: int, d: int):
        self._d = d
        self._planes = self._init_planes(secret_key, embed_dim, d)

    @staticmethod
    def _init_planes(secret_key: str, embed_dim: int, d: int) -> torch.Tensor:
        key_bytes = secret_key.encode("utf-8")
        digest = hmac.new(key_bytes, b"lsh", hashlib.sha256).digest()
        seed = int.from_bytes(digest[:8], "big")
        gen = torch.Generator()
        gen.manual_seed(seed)
        planes = torch.randn(d, embed_dim, generator=gen)
        return F.normalize(planes, dim=1)

    def sign(self, u: torch.Tensor) -> tuple[int, ...]:
        """Compute d-bit LSH signature for embedding vector u.

        Args:
            u: Embedding vector of shape (embed_dim,).

        Returns:
            Tuple of d bits in {0, 1}.
        """
        u_norm = F.normalize(u.float().unsqueeze(0), dim=1)
        dots = (self._planes.float() @ u_norm.T).squeeze(1)
        return tuple((dots > 0).int().tolist())

    def min_margin(self, u: torch.Tensor) -> float:
        """Return minimum absolute cosine distance from u to all hyperplanes.

        Used as the margin guard: a large value means u is well inside its
        LSH region, far from any decision boundary.

        Args:
            u: Embedding vector of shape (embed_dim,).

        Returns:
            Minimum |cos(u, n_i)| across all d hyperplanes.
        """
        u_norm = F.normalize(u.float().unsqueeze(0), dim=1)
        dots = (self._planes.float() @ u_norm.T).squeeze(1)
        return dots.abs().min().item()
