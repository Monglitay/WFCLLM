"""Tests for wfcllm.encoder.evaluate."""

import pytest
import torch

from wfcllm.encoder.evaluate import (
    cosine_separation,
    recall_at_k,
    projection_sign_accuracy,
)


class TestCosineSeparation:
    def test_perfect_separation(self):
        anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # identical
        negative = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # orthogonal
        result = cosine_separation(anchor, positive, negative)
        assert result["mean_pos_cos"] > result["mean_neg_cos"]
        assert result["separation"] > 0

    def test_keys(self):
        anchor = torch.randn(4, 16)
        positive = torch.randn(4, 16)
        negative = torch.randn(4, 16)
        result = cosine_separation(anchor, positive, negative)
        assert "mean_pos_cos" in result
        assert "mean_neg_cos" in result
        assert "separation" in result


class TestRecallAtK:
    def test_perfect_recall(self):
        # anchor[i] is closest to candidates[i]
        embeddings = torch.eye(5)
        candidates = torch.eye(5)
        r = recall_at_k(embeddings, candidates, k=1)
        assert r == 1.0

    def test_recall_at_5(self):
        embeddings = torch.eye(5)
        candidates = torch.eye(5)
        r = recall_at_k(embeddings, candidates, k=5)
        assert r == 1.0


class TestProjectionSignAccuracy:
    def test_all_correct(self):
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target_bits = torch.tensor([1, 0])  # 1 → positive, 0 → negative
        acc = projection_sign_accuracy(embeddings, directions, target_bits)
        assert acc == 1.0

    def test_all_wrong(self):
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target_bits = torch.tensor([0, 1])  # reversed
        acc = projection_sign_accuracy(embeddings, directions, target_bits)
        assert acc == 0.0
