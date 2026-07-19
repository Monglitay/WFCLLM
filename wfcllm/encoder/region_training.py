"""Public, key-blind objectives for a region-diverse semantic projection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def split_source_ids(
    source_ids: Sequence[str],
    *,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    """Create a deterministic 80/10/10 split without source leakage."""

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    unique = {
        source_id for source_id in source_ids
        if isinstance(source_id, str) and source_id
    }
    if len(unique) < 10:
        raise ValueError("region training requires at least ten source IDs")
    ordered = sorted(
        unique,
        key=lambda source_id: hashlib.sha256(
            f"{seed}\0{source_id}".encode("utf-8")
        ).digest(),
    )
    validation_count = max(1, len(ordered) // 10)
    test_count = max(1, len(ordered) // 10)
    train_count = len(ordered) - validation_count - test_count
    return {
        "train": tuple(ordered[:train_count]),
        "validation": tuple(ordered[train_count:train_count + validation_count]),
        "test": tuple(ordered[train_count + validation_count:]),
    }


def semantic_region_diversity_loss(
    group_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    public_planes: torch.Tensor,
    *,
    positive_cosine_min: float = 0.80,
    negative_cosine_max: float = 0.30,
    plane_margin: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep variants semantically close while making public LSH codes orthogonal.

    ``group_embeddings`` contains anchor plus three structure-preserving variants.
    The objective never receives a watermark key or an allowed-region set.
    """

    if group_embeddings.ndim != 3 or group_embeddings.shape[1] != 4:
        raise ValueError("group_embeddings must have shape (batch, 4, embed_dim)")
    if negative_embeddings.shape != group_embeddings[:, 0, :].shape:
        raise ValueError("negative_embeddings must have shape (batch, embed_dim)")
    if public_planes.ndim != 2 or public_planes.shape[1] != group_embeddings.shape[2]:
        raise ValueError("public_planes must have shape (bits, embed_dim)")

    groups = F.normalize(group_embeddings.float(), dim=-1)
    negatives = F.normalize(negative_embeddings.float(), dim=-1)
    planes = F.normalize(public_planes.float(), dim=-1)
    anchor = groups[:, 0, :]
    positive_cosines = F.cosine_similarity(
        anchor.unsqueeze(1),
        groups[:, 1:, :],
        dim=-1,
    )
    negative_cosines = F.cosine_similarity(anchor, negatives, dim=-1)
    semantic_loss = F.relu(positive_cosine_min - positive_cosines).mean()
    negative_loss = F.relu(negative_cosines - negative_cosine_max).mean()

    responses = torch.einsum("bve,de->bvd", groups, planes)
    soft_codes = F.normalize(torch.tanh(12.0 * responses), dim=-1)
    gram = torch.matmul(soft_codes, soft_codes.transpose(1, 2))
    identity = torch.eye(4, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    orthogonality_loss = (gram - identity).square().mean()
    margin_loss = F.relu(plane_margin - responses.abs()).mean()
    balance_loss = torch.tanh(12.0 * responses).mean(dim=(0, 1)).square().mean()
    if responses.shape[-1] < 4:
        raise ValueError("public region diversity requires at least four LSH bits")
    prototypes = responses.new_tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    signed_prototype_responses = responses[:, :, :4] * prototypes.unsqueeze(0)
    prototype_loss = F.relu(0.05 - signed_prototype_responses).mean()
    total = (
        semantic_loss
        + negative_loss
        + 2.0 * orthogonality_loss
        + 8.0 * prototype_loss
        + margin_loss
        + 0.1 * balance_loss
    )
    metrics = {
        "semantic": float(semantic_loss.detach()),
        "negative": float(negative_loss.detach()),
        "region_orthogonality": float(orthogonality_loss.detach()),
        "prototype": float(prototype_loss.detach()),
        "prototype_sign_accuracy": float(
            (signed_prototype_responses.detach() > 0).float().mean()
        ),
        "plane_margin": float(margin_loss.detach()),
        "bit_balance": float(balance_loss.detach()),
        "mean_positive_cosine": float(positive_cosines.detach().mean()),
        "mean_negative_cosine": float(negative_cosines.detach().mean()),
    }
    return total, metrics


def build_region_training_groups_from_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    max_groups: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Select deterministic four-view groups with cross-source negatives."""

    if type(max_groups) is not int or max_groups <= 0:
        raise ValueError("max_groups must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    eligible: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        source = block.get("source")
        source_id = block.get("source_id", f"source-{index}")
        positives = block.get("positive_variants", ())
        if (
            isinstance(source, str)
            and source.strip()
            and isinstance(source_id, str)
            and source_id
            and isinstance(positives, Sequence)
            and not isinstance(positives, (str, bytes))
        ):
            unique = tuple(dict.fromkeys(
                value for value in positives
                if isinstance(value, str) and value.strip() and value != source
            ))
            if len(unique) >= 3:
                eligible.append({
                    "anchor": source,
                    "positives": unique[:3],
                    "source_id": source_id,
                    "selection_id": f"{source_id}\0{index}\0{source}",
                })
    ordered = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['selection_id']}".encode("utf-8")
        ).digest(),
    )[:max_groups]
    if len({row["source_id"] for row in eligible}) < 2:
        raise ValueError("region training requires at least two source IDs")
    result: list[dict[str, Any]] = []
    for row in ordered:
        negative_pool = [item for item in eligible if item["source_id"] != row["source_id"]]
        negative = min(
            negative_pool,
            key=lambda item: hashlib.sha256(
                f"{seed}\0{row['selection_id']}\0{item['selection_id']}".encode("utf-8")
            ).digest(),
        )
        result.append({
            "anchor": row["anchor"],
            "positives": row["positives"],
            "negative": negative["anchor"],
            "source_id": row["source_id"],
            "negative_source_id": negative["source_id"],
        })
    return tuple(result)
