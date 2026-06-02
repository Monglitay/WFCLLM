"""Candidate-pool quality diagnostics for anchor validation."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, median
from typing import Any

from wfcllm.evaluation.anchor_validation.schema import CandidateContext


def build_pool_quality_summary(contexts: list[CandidateContext]) -> dict[str, Any]:
    candidate_counts = [len(context.candidates) for context in contexts]
    unique_counts = [
        len({candidate.block_text for candidate in context.candidates})
        for context in contexts
    ]
    candidates = [
        candidate
        for context in contexts
        for candidate in context.candidates
    ]
    candidate_lengths = [len(candidate.block_text) for candidate in candidates]
    return {
        "context_count": len(contexts),
        "total_candidates": len(candidates),
        "candidates_per_context": _numeric_summary(candidate_counts),
        "unique_candidates_per_context": _numeric_summary(unique_counts),
        "node_type_distribution": dict(sorted(Counter(context.node_type for context in contexts).items())),
        "parent_node_type_distribution": dict(sorted(Counter(context.parent_node_type for context in contexts).items())),
        "task_distribution": dict(sorted(Counter(context.task_id for context in contexts).items())),
        "candidate_length": _numeric_summary(candidate_lengths),
        "parse_valid_rate": _rate([candidate.parse_valid for candidate in candidates]),
        "syntax_valid_rate": _rate([candidate.syntax_valid for candidate in candidates]),
    }


def enrich_pool_quality_with_embedding_diversity(
    summary: dict[str, Any],
    embeddings_by_context: dict[str, list[tuple[str, tuple[float, ...]]]],
) -> dict[str, Any]:
    distances = [
        _mean_pairwise_cosine_distance([embedding for _, embedding in embeddings])
        for embeddings in embeddings_by_context.values()
        if len(embeddings) >= 2
    ]
    enriched = dict(summary)
    enriched["embedding_pairwise_diversity"] = {
        "context_count": len(distances),
        "mean_cosine_distance": mean(distances) if distances else 0.0,
        "min_cosine_distance": min(distances) if distances else 0.0,
        "max_cosine_distance": max(distances) if distances else 0.0,
    }
    return enriched


def _numeric_summary(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _mean_pairwise_cosine_distance(embeddings: list[tuple[float, ...]]) -> float:
    distances: list[float] = []
    for left_index, left in enumerate(embeddings):
        for right in embeddings[left_index + 1:]:
            distances.append(1.0 - _cosine_similarity(left, right))
    return mean(distances) if distances else 0.0


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same width")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denom = left_norm * right_norm
    if denom == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denom
