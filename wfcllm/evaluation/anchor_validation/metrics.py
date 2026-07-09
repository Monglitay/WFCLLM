"""Region diversity and valid-hit balance metrics for anchor validation."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from wfcllm.evaluation.anchor_validation.schema import RegionMetricRow
from wfcllm.semantic.anchor_lsh import pairwise_hamming_diversity


@dataclass(frozen=True)
class ValidHitBalance:
    hit_rate: float
    gamma_deviation: float


def region_entropy(signatures: list[tuple[int, ...]]) -> float:
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    total = len(signatures)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def normalized_region_entropy(
    signatures: list[tuple[int, ...]],
    region_count: int,
) -> float:
    if not signatures:
        return 0.0
    denom = math.log(min(len(signatures), region_count))
    if denom <= 0.0:
        return 0.0
    return region_entropy(signatures) / denom


def effective_region_count(signatures: list[tuple[int, ...]]) -> float:
    return math.exp(region_entropy(signatures)) if signatures else 0.0


def valid_hit_balance(
    signatures: list[tuple[int, ...]],
    valid_set: frozenset[tuple[int, ...]],
    gamma: float,
) -> ValidHitBalance:
    if not signatures:
        return ValidHitBalance(hit_rate=0.0, gamma_deviation=abs(gamma))
    hits = sum(1 for signature in signatures if signature in valid_set)
    hit_rate = hits / len(signatures)
    return ValidHitBalance(hit_rate=hit_rate, gamma_deviation=abs(hit_rate - gamma))


def summarize_signature_metrics(
    context_id: str,
    dataset: str,
    task_id: str,
    method: str,
    signatures: list[tuple[int, ...]],
    region_count: int,
    projection_key_id: str | None,
    key_id: str | None,
    gamma: float | None,
    valid_set: frozenset[tuple[int, ...]] | None,
    node_type: str | None = None,
    block_ordinal: int | None = None,
) -> RegionMetricRow:
    normalized_entropy = normalized_region_entropy(signatures, region_count)
    balance = (
        valid_hit_balance(signatures, valid_set, gamma)
        if valid_set is not None and gamma is not None
        else None
    )
    return RegionMetricRow(
        context_id=context_id,
        dataset=dataset,
        task_id=task_id,
        method=method,
        projection_key_id=projection_key_id,
        key_id=key_id,
        gamma=gamma,
        candidate_count=len(signatures),
        normalized_entropy=normalized_entropy,
        collapse_ratio=1.0 - normalized_entropy,
        effective_region_count=effective_region_count(signatures),
        hamming_diversity=pairwise_hamming_diversity(signatures),
        node_type=node_type,
        valid_hit_rate=balance.hit_rate if balance else None,
        gamma_deviation=balance.gamma_deviation if balance else None,
        block_ordinal=block_ordinal,
    )


def bootstrap_mean_ci(
    values: list[float],
    iterations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2) * iterations))
    hi_idx = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return means[lo_idx], means[hi_idx]
