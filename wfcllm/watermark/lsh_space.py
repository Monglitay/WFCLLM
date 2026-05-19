"""Global LSH hyperplane space for watermark embedding and extraction."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class LSHSpace:
    """Manage d global hyperplanes derived from a secret key.

    The hyperplanes statically partition the semantic embedding space into
    2^d regions identified by binary LSH signatures.

    Optionally applies a ZCA whitening transform to the embedding before
    computing the LSH signature, improving region uniformity when the
    encoder's output distribution is skewed.
    """

    def __init__(
        self,
        secret_key: str,
        embed_dim: int,
        d: int,
        whitening_path: str | None = None,
    ):
        self._d = d
        self._planes = self._init_planes(secret_key, embed_dim, d)
        self._W: torch.Tensor | None = None
        self._mu: torch.Tensor | None = None
        if whitening_path is not None:
            self._load_whitening(whitening_path)

    @staticmethod
    def _init_planes(secret_key: str, embed_dim: int, d: int) -> torch.Tensor:
        key_bytes = secret_key.encode("utf-8")
        digest = hmac.new(key_bytes, b"lsh", hashlib.sha256).digest()
        seed = int.from_bytes(digest[:8], "big")
        gen = torch.Generator()
        gen.manual_seed(seed)
        planes = torch.randn(d, embed_dim, generator=gen)
        return F.normalize(planes, dim=1)

    def _load_whitening(self, whitening_path: str) -> None:
        """Load ZCA whitening matrix and mean from .npz file."""
        data = np.load(whitening_path)
        self._W = torch.from_numpy(data["W"].astype(np.float32))   # (embed_dim, embed_dim)
        self._mu = torch.from_numpy(data["mu"].astype(np.float32))  # (embed_dim,)

    def _whiten(self, u: torch.Tensor) -> torch.Tensor:
        """Apply ZCA whitening: u_w = W @ (u - mu)."""
        if self._W is None or self._mu is None:
            return u
        device = u.device
        mu = self._mu.to(device)
        W = self._W.to(device)
        return W @ (u - mu)

    def sign(self, u: torch.Tensor) -> tuple[int, ...]:
        """Compute d-bit LSH signature for embedding vector u."""
        u = self._whiten(u.float())
        u_norm = F.normalize(u.unsqueeze(0), dim=1)
        dots = (self._planes.float() @ u_norm.T).squeeze(1)
        return tuple((dots > 0).int().tolist())

    def min_margin(self, u: torch.Tensor) -> float:
        """Return minimum absolute cosine distance from u to all hyperplanes."""
        u = self._whiten(u.float())
        u_norm = F.normalize(u.unsqueeze(0), dim=1)
        dots = (self._planes.float() @ u_norm.T).squeeze(1)
        return dots.abs().min().item()
