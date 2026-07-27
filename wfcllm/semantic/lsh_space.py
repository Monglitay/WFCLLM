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

    def __init__(
        self,
        secret_key: str,
        embed_dim: int,
        d: int,
    ):
        self._d = d
        self._planes = self._init_planes(secret_key, embed_dim, d)

    @property
    def planes(self) -> torch.Tensor:
        """Return a defensive copy of the normalized LSH hyperplanes."""
        return self._planes.clone()

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
        """Compute d-bit LSH signature for embedding vector u."""
        return sign_with_planes(u.float(), self._planes)

    def min_margin(self, u: torch.Tensor) -> float:
        """Return minimum absolute cosine distance from u to all hyperplanes."""
        return min_margin_with_planes(u.float(), self._planes)


def sign_with_planes(
    embedding: torch.Tensor, planes: torch.Tensor
) -> tuple[int, ...]:
    if embedding.ndim != 1 or planes.ndim != 2:
        raise ValueError("embedding must be 1-D and planes must be 2-D")
    if planes.shape[1] != embedding.shape[0]:
        raise ValueError("embedding and planes dimensions must match")
    normalized = F.normalize(embedding.float(), dim=0)
    return tuple(int(value >= 0) for value in planes.float() @ normalized)


def min_margin_with_planes(
    embedding: torch.Tensor, planes: torch.Tensor
) -> float:
    if embedding.ndim != 1 or planes.ndim != 2:
        raise ValueError("embedding must be 1-D and planes must be 2-D")
    if planes.shape[1] != embedding.shape[0]:
        raise ValueError("embedding and planes dimensions must match")
    normalized = F.normalize(embedding.float(), dim=0)
    return float(torch.min(torch.abs(planes.float() @ normalized)).item())
