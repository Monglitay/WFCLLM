"""Tests for the token-channel partition protocol."""

from __future__ import annotations

import torch

from wfcllm.watermark.token_channel.protocol import build_partition
from wfcllm.watermark.token_channel.protocol import make_prefix_key
from wfcllm.watermark.token_channel.protocol import make_scored_token_key


def test_partition_uses_full_vocab_pairing() -> None:
    logits = torch.tensor([0.2, 1.1, -0.5, 0.7, 0.1, 0.9])

    partition = build_partition(logits=logits, prefix_ids=(1, 2), secret_key="k")

    assert partition.sorted_token_ids == [1, 5, 3, 0, 4, 2]
    assert len(partition.green_token_ids) == 3
    assert len(partition.red_token_ids) == 3
    assert partition.green_token_ids | partition.red_token_ids == set(range(6))
    assert partition.green_token_ids.isdisjoint(partition.red_token_ids)


def test_partition_is_deterministic_for_same_key_and_prefix() -> None:
    logits = torch.tensor([0.9, 0.8, 0.2, 0.1])

    first = build_partition(logits=logits, prefix_ids=(4, 5), secret_key="shared-key")
    second = build_partition(logits=logits, prefix_ids=(4, 5), secret_key="shared-key")

    assert first == second


def test_partition_has_stable_known_output_for_secret_and_prefix() -> None:
    logits = torch.tensor([0.9, 0.8, 0.2, 0.1])

    baseline = build_partition(logits=logits, prefix_ids=(4, 5), secret_key="shared-key")
    changed_key = build_partition(logits=logits, prefix_ids=(4, 5), secret_key="other-key")
    changed_prefix = build_partition(logits=logits, prefix_ids=(4, 6), secret_key="shared-key")

    assert baseline.green_token_ids == {0, 3}
    assert baseline.red_token_ids == {1, 2}
    assert changed_key.green_token_ids == {1, 3}
    assert changed_key.red_token_ids == {0, 2}
    assert changed_prefix.green_token_ids == {0, 2}
    assert changed_prefix.red_token_ids == {1, 3}


def test_partition_handles_odd_vocab_size_deterministically() -> None:
    logits = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])

    first = build_partition(logits=logits, prefix_ids=(7,), secret_key="odd-key")
    second = build_partition(logits=logits, prefix_ids=(7,), secret_key="odd-key")

    assert first == second
    assert len(first.green_token_ids) + len(first.red_token_ids) == 5
    assert first.green_token_ids | first.red_token_ids == set(range(5))
    assert first.green_token_ids.isdisjoint(first.red_token_ids)


def test_make_prefix_key_returns_hashable_tuple() -> None:
    prefix_ids = [3, 1, 4]

    prefix_key = make_prefix_key(prefix_ids)

    assert prefix_key == (3, 1, 4)
    assert isinstance(prefix_key, tuple)
    assert {prefix_key} == {(3, 1, 4)}


def test_make_scored_token_key_appends_token_id_to_prefix_key() -> None:
    scored_token_key = make_scored_token_key([3, 1, 4], 9)

    assert scored_token_key == (3, 1, 4, 9)
    assert isinstance(scored_token_key, tuple)
    assert {scored_token_key} == {(3, 1, 4, 9)}
