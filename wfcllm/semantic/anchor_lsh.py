"""Tensor helpers for anchored LSH signatures."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def normalize_nonzero(vectors: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize nonzero vectors and preserve exact zeros."""
    vectors = vectors.float()
    norms = torch.linalg.vector_norm(vectors, dim=dim, keepdim=True)
    normalized = vectors / norms.clamp_min(eps)
    return torch.where(norms > eps, normalized, torch.zeros_like(normalized))


def project_planes_orthogonal(
    planes: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project hyperplane normals onto the subspace orthogonal to anchor."""
    planes = planes.float()
    anchor = anchor.float().flatten()
    denom = torch.dot(anchor, anchor)
    if denom.item() <= eps:
        return normalize_nonzero(planes, dim=1, eps=eps)

    coeff = (planes @ anchor).unsqueeze(1) / denom
    projected = planes - coeff * anchor.unsqueeze(0)
    return normalize_nonzero(projected, dim=1, eps=eps)


def sign_with_planes(u: torch.Tensor, planes: torch.Tensor) -> tuple[int, ...]:
    """Compute a binary LSH signature using the provided hyperplanes."""
    u_norm = F.normalize(u.float().flatten().unsqueeze(0), dim=1)
    plane_norm = normalize_nonzero(planes.to(device=u_norm.device), dim=1)
    dots = (plane_norm @ u_norm.T).squeeze(1)
    return tuple((dots > 0).int().tolist())


def min_margin_with_planes(u: torch.Tensor, planes: torch.Tensor) -> float:
    """Return the minimum absolute cosine margin to the provided hyperplanes."""
    u_norm = F.normalize(u.float().flatten().unsqueeze(0), dim=1)
    plane_norm = normalize_nonzero(planes.to(device=u_norm.device), dim=1)
    dots = (plane_norm @ u_norm.T).squeeze(1)
    return float(dots.abs().min().item())


def anchored_signature(
    u: torch.Tensor,
    planes: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[int, ...]:
    """Compute an LSH signature after removing the anchor direction from planes."""
    projected = project_planes_orthogonal(planes, anchor, eps=eps)
    return sign_with_planes(u, projected)


def residual_signature(
    u: torch.Tensor,
    center: torch.Tensor,
    planes: torch.Tensor,
) -> tuple[int, ...]:
    """Compute a signature for u after subtracting a SeqMark-style center."""
    residual = u.float().flatten() - center.float().flatten()
    return sign_with_planes(residual, planes)


def random_anchor(
    secret_key: str,
    context_id: str,
    method: str,
    embed_dim: int,
) -> torch.Tensor:
    """Create a deterministic unit-norm random anchor for a context."""
    key_bytes = secret_key.encode("utf-8")
    message = f"{method}:{context_id}".encode("utf-8")
    digest = hmac.new(key_bytes, message, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big")

    gen = torch.Generator()
    gen.manual_seed(seed)
    anchor = torch.randn(embed_dim, generator=gen)
    return normalize_nonzero(anchor, dim=0)


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the Hamming distance between two equal-width signatures."""
    if len(left) != len(right):
        raise ValueError("Signatures must have the same width")
    return sum(int(a != b) for a, b in zip(left, right))


def pairwise_hamming_diversity(signatures: Sequence[Sequence[int]]) -> float:
    """Average pairwise Hamming distance normalized by signature width."""
    if len(signatures) < 2:
        return 0.0

    width = len(signatures[0])
    if width == 0:
        return 0.0

    total = 0
    pairs = 0
    for i, left in enumerate(signatures):
        if len(left) != width:
            raise ValueError("All signatures must have the same width")
        for right in signatures[i + 1 :]:
            if len(right) != width:
                raise ValueError("All signatures must have the same width")
            total += hamming_distance(left, right)
            pairs += 1

    return total / (pairs * width)
