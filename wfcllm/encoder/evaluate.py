"""Evaluation metrics for the semantic encoder."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def cosine_separation(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> dict[str, float]:
    """Compute mean cosine similarity for positive and negative pairs.

    Returns dict with mean_pos_cos, mean_neg_cos, separation.
    """
    cos_pos = F.cosine_similarity(anchor, positive, dim=1).mean().item()
    cos_neg = F.cosine_similarity(anchor, negative, dim=1).mean().item()
    return {
        "mean_pos_cos": cos_pos,
        "mean_neg_cos": cos_neg,
        "separation": cos_pos - cos_neg,
    }


def recall_at_k(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute Recall@K: fraction of queries whose true match is in top-K.

    Assumes query_embeddings[i]'s ground truth match is candidate_embeddings[i].
    """
    # Cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        query_embeddings.unsqueeze(1),
        candidate_embeddings.unsqueeze(0),
        dim=2,
    )
    # Top-K indices for each query
    topk_indices = sim_matrix.topk(k, dim=1).indices
    # Check if true index is in top-K
    true_indices = torch.arange(len(query_embeddings)).unsqueeze(1)
    hits = (topk_indices == true_indices).any(dim=1).float()
    return hits.mean().item()


def projection_sign_accuracy(
    embeddings: torch.Tensor,
    directions: torch.Tensor,
    target_bits: torch.Tensor,
) -> float:
    """Compute accuracy of projection sign matching target bits.

    Simulates watermark verification: checks if sgn(cos(u, v)) matches target.

    Args:
        embeddings: (N, D) semantic vectors
        directions: (N, D) direction vectors
        target_bits: (N,) values in {0, 1}, mapped to {-1, +1}
    """
    cos_proj = F.cosine_similarity(embeddings, directions, dim=1)
    predicted_sign = torch.sign(cos_proj)
    target_sign = 2.0 * target_bits.float() - 1.0  # {0,1} → {-1,+1}
    correct = (predicted_sign == target_sign).float()
    return correct.mean().item()


def save_evaluation_report(metrics: dict, output_dir: str) -> Path:
    """Save evaluation metrics to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return report_path
