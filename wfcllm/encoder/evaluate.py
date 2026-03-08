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
    sim_matrix = F.cosine_similarity(
        query_embeddings.unsqueeze(1),
        candidate_embeddings.unsqueeze(0),
        dim=2,
    )
    topk_indices = sim_matrix.topk(k, dim=1).indices
    true_indices = torch.arange(len(query_embeddings)).unsqueeze(1)
    hits = (topk_indices == true_indices).any(dim=1).float()
    return hits.mean().item()


def watermark_sign_consistency(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    num_directions: int = 64,
    seed: int = 42,
) -> float:
    """Measure sign consistency of anchor/positive projections onto fixed directions.

    For each (anchor[i], positive[i]) pair, computes the fraction of K fixed
    direction vectors where sign(anchor·d) == sign(positive·d).
    High consistency means semantically similar embeddings agree on watermark bits.

    Args:
        anchor: (N, D) L2-normalized embeddings
        positive: (N, D) L2-normalized embeddings (semantic variants of anchor)
        num_directions: Number of fixed watermark direction vectors K
        seed: Random seed for reproducible direction generation

    Returns:
        Mean sign consistency across all pairs and directions (range [0, 1]).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    directions = torch.randn(anchor.shape[1], num_directions, generator=gen)
    directions = F.normalize(directions, p=2, dim=0)  # (D, K)

    # Project: (N, D) @ (D, K) → (N, K)
    proj_anchor = anchor.float() @ directions
    proj_positive = positive.float() @ directions

    # Sign consistency per direction per pair: (N, K)
    same_sign = (torch.sign(proj_anchor) == torch.sign(proj_positive)).float()
    return same_sign.mean().item()


def mean_reciprocal_rank(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    For each query[i], finds the rank of candidate[i] in the similarity-sorted list.
    MRR = mean(1 / rank_i).

    Args:
        query_embeddings: (N, D) L2-normalized query vectors
        candidate_embeddings: (N, D) L2-normalized candidate vectors

    Returns:
        MRR score in [0, 1].
    """
    sim_matrix = F.cosine_similarity(
        query_embeddings.unsqueeze(1),
        candidate_embeddings.unsqueeze(0),
        dim=2,
    )  # (N, N)
    # Rank of true match: number of candidates with higher similarity + 1
    n = len(query_embeddings)
    true_sims = sim_matrix[torch.arange(n), torch.arange(n)].unsqueeze(1)  # (N, 1)
    ranks = (sim_matrix > true_sims).sum(dim=1).float() + 1.0  # (N,)
    return (1.0 / ranks).mean().item()


def mean_average_precision(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> float:
    """Compute Mean Average Precision (MAP) for single-positive retrieval.

    Each query has exactly one positive candidate (at the same index).
    AP_i = 1 / rank_i, MAP = mean(AP_i).

    Args:
        query_embeddings: (N, D) L2-normalized query vectors
        candidate_embeddings: (N, D) L2-normalized candidate vectors

    Returns:
        MAP score in [0, 1].
    """
    # With one positive per query, MAP == MRR
    return mean_reciprocal_rank(query_embeddings, candidate_embeddings)


def pair_f1_metrics(
    pos_cos_sims: torch.Tensor,
    neg_cos_sims: torch.Tensor,
) -> dict[str, float]:
    """Compute threshold-optimized F1 for positive/negative pair classification.

    Scans thresholds in [0, 1] to find the one that maximizes F1, then reports
    precision, recall, F1, and the optimal threshold.

    Args:
        pos_cos_sims: (N,) cosine similarities for positive (anchor, positive) pairs
        neg_cos_sims: (M,) cosine similarities for negative (anchor, negative) pairs

    Returns:
        Dict with pair_precision, pair_recall, pair_f1, optimal_threshold.
    """
    all_sims = torch.cat([pos_cos_sims, neg_cos_sims])
    labels = torch.cat([
        torch.ones(len(pos_cos_sims)),
        torch.zeros(len(neg_cos_sims)),
    ])

    best_f1 = 0.0
    best_thresh = 0.5
    best_precision = 0.0
    best_recall = 0.0

    # Scan 100 threshold candidates between min and max similarity
    lo = all_sims.min().item()
    hi = all_sims.max().item()
    thresholds = torch.linspace(lo, hi, steps=100)

    for thresh in thresholds:
        predicted_pos = (all_sims >= thresh)
        tp = (predicted_pos & (labels == 1)).sum().float()
        fp = (predicted_pos & (labels == 0)).sum().float()
        fn = (~predicted_pos & (labels == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1.item() > best_f1:
            best_f1 = f1.item()
            best_thresh = thresh.item()
            best_precision = precision.item()
            best_recall = recall.item()

    return {
        "pair_precision": best_precision,
        "pair_recall": best_recall,
        "pair_f1": best_f1,
        "optimal_threshold": best_thresh,
    }


def save_evaluation_report(metrics: dict, output_dir: str) -> Path:
    """Save evaluation metrics to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return report_path
