"""Triplet code dataset for contrastive learning."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def build_triplets_from_blocks(
    blocks: list[dict],
    negative_ratio: float = 0.5,
    seed: int = 42,
    all_sources: list[str] | None = None,
) -> list[dict]:
    """Build (anchor, positive, negative) triplets from transformed blocks.

    Args:
        blocks: List of dicts with keys 'source', 'positive_variants', 'negative_variants'.
        negative_ratio: Fraction of negatives that are hard (from negative_variants).
            0.0 = all random, 1.0 = all hard.
        seed: Random seed for reproducibility.
        all_sources: Pool of all block sources for random negative sampling.
            If None, collected from blocks.

    Returns:
        List of {"anchor": str, "positive": str, "negative": str}.
    """
    rng = random.Random(seed)

    if all_sources is None:
        all_sources = [b["source"] for b in blocks]

    triplets: list[dict] = []

    for block in blocks:
        source = block["source"]
        positives = block.get("positive_variants", [])
        negatives = block.get("negative_variants", [])

        if not positives:
            continue

        for pos in positives:
            # Decide negative type
            use_hard = rng.random() < negative_ratio and negatives
            if use_hard:
                neg = rng.choice(negatives)
            else:
                # Random negative from other blocks
                candidates = [s for s in all_sources if s != source]
                if not candidates:
                    continue
                neg = rng.choice(candidates)

            triplets.append({
                "anchor": source,
                "positive": pos,
                "negative": neg,
            })

    return triplets


class TripletCodeDataset(Dataset):
    """PyTorch Dataset yielding tokenized (anchor, positive, negative) triplets."""

    def __init__(
        self,
        triplets: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        triplet = self.triplets[idx]
        result = {}
        for key in ("anchor", "positive", "negative"):
            encoded = self.tokenizer(
                triplet[key],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result[f"{key}_input_ids"] = encoded["input_ids"].squeeze(0)
            result[f"{key}_attention_mask"] = encoded["attention_mask"].squeeze(0)
        return result
