"""Deterministic token-channel partition protocol."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class PartitionResult:
    """Green/red partition reconstructed from keyed full-vocabulary pairing."""

    prefix_key: tuple[int, ...]
    sorted_token_ids: list[int]
    green_token_ids: frozenset[int]
    red_token_ids: frozenset[int]


def make_prefix_key(prefix_ids: Iterable[int]) -> tuple[int, ...]:
    """Return a stable hashable prefix key for repeat filtering."""

    return tuple(int(token_id) for token_id in prefix_ids)


def make_scored_token_key(prefix_ids: Iterable[int], token_id: int) -> tuple[int, ...]:
    """Return a stable hashable key for a scored prefix/token event."""

    return (*make_prefix_key(prefix_ids), int(token_id))


def build_partition(
    logits: torch.Tensor,
    prefix_ids: tuple[int, ...] | list[int],
    secret_key: str,
) -> PartitionResult:
    """Build the keyed V1 green/red partition from full-vocabulary logits."""

    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor")

    prefix_key = make_prefix_key(prefix_ids)
    sorted_token_ids = torch.argsort(logits, descending=True, stable=True).tolist()

    green_token_ids: set[int] = set()
    red_token_ids: set[int] = set()
    bit_stream = _bit_stream(secret_key=secret_key, prefix_key=prefix_key)

    for pair_start in range(0, len(sorted_token_ids) - 1, 2):
        first_token_id = sorted_token_ids[pair_start]
        second_token_id = sorted_token_ids[pair_start + 1]
        if next(bit_stream) == 0:
            green_token_ids.add(first_token_id)
            red_token_ids.add(second_token_id)
        else:
            green_token_ids.add(second_token_id)
            red_token_ids.add(first_token_id)

    if len(sorted_token_ids) % 2 == 1:
        final_token_id = sorted_token_ids[-1]
        if next(bit_stream) == 0:
            green_token_ids.add(final_token_id)
        else:
            red_token_ids.add(final_token_id)

    return PartitionResult(
        prefix_key=prefix_key,
        sorted_token_ids=sorted_token_ids,
        green_token_ids=frozenset(green_token_ids),
        red_token_ids=frozenset(red_token_ids),
    )


def _bit_stream(secret_key: str, prefix_key: tuple[int, ...]):
    seed_material = _make_seed_material(secret_key=secret_key, prefix_key=prefix_key)
    counter = 0
    while True:
        block = hashlib.sha256(seed_material + counter.to_bytes(8, "big")).digest()
        for byte in block:
            for bit_index in range(8):
                yield (byte >> bit_index) & 1
        counter += 1


def _make_seed_material(secret_key: str, prefix_key: tuple[int, ...]) -> bytes:
    prefix_bytes = ",".join(str(token_id) for token_id in prefix_key).encode("utf-8")
    return hmac.new(secret_key.encode("utf-8"), prefix_bytes, hashlib.sha256).digest()
