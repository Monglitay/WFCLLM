"""Tests for wfcllm.encoder.evaluate."""

import pytest
import torch

from wfcllm.encoder.evaluate import (
    cosine_separation,
    recall_at_k,
    watermark_sign_consistency,
    mean_reciprocal_rank,
    mean_average_precision,
    pair_f1_metrics,
)


class TestCosineSeparation:
    def test_perfect_separation(self):
        anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        negative = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
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
        embeddings = torch.eye(5)
        candidates = torch.eye(5)
        r = recall_at_k(embeddings, candidates, k=1)
        assert r == 1.0

    def test_recall_at_5(self):
        embeddings = torch.eye(5)
        candidates = torch.eye(5)
        r = recall_at_k(embeddings, candidates, k=5)
        assert r == 1.0


class TestWatermarkSignConsistency:
    def test_identical_embeddings_high_consistency(self):
        # anchor == positive → 投影符号完全相同 → consistency=1.0
        anchor = torch.randn(10, 32)
        anchor = torch.nn.functional.normalize(anchor, dim=1)
        positive = anchor.clone()
        score = watermark_sign_consistency(anchor, positive, num_directions=64, seed=42)
        assert score == pytest.approx(1.0)

    def test_random_embeddings_near_half(self):
        # 完全随机的 anchor/positive → 期望一致性接近 0.5
        torch.manual_seed(0)
        anchor = torch.nn.functional.normalize(torch.randn(200, 64), dim=1)
        positive = torch.nn.functional.normalize(torch.randn(200, 64), dim=1)
        score = watermark_sign_consistency(anchor, positive, num_directions=64, seed=42)
        assert 0.3 < score < 0.7

    def test_output_in_range(self):
        anchor = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
        positive = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
        score = watermark_sign_consistency(anchor, positive, num_directions=16, seed=0)
        assert 0.0 <= score <= 1.0

    def test_deterministic_with_same_seed(self):
        anchor = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
        positive = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
        s1 = watermark_sign_consistency(anchor, positive, num_directions=16, seed=99)
        s2 = watermark_sign_consistency(anchor, positive, num_directions=16, seed=99)
        assert s1 == s2


class TestMeanReciprocalRank:
    def test_perfect_ranking(self):
        # anchor[i] 与 candidates[i] 完全相同 → 每个 rank=1 → MRR=1.0
        embeddings = torch.eye(5)
        mrr = mean_reciprocal_rank(embeddings, embeddings)
        assert mrr == pytest.approx(1.0)

    def test_worst_ranking(self):
        # query[0]=[1,0,0], candidates[0]=[0,0,1] 排最后 → MRR 接近 0
        query = torch.eye(4)
        # 候选按反序排列：query[i] 与 candidates[3-i] 最相似
        candidates = torch.eye(4).flip(0)
        mrr = mean_reciprocal_rank(query, candidates)
        assert mrr <= 0.5

    def test_output_in_range(self):
        query = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
        candidates = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
        mrr = mean_reciprocal_rank(query, candidates)
        assert 0.0 <= mrr <= 1.0


class TestMeanAveragePrecision:
    def test_perfect_map(self):
        embeddings = torch.eye(5)
        map_score = mean_average_precision(embeddings, embeddings)
        assert map_score == pytest.approx(1.0)

    def test_output_in_range(self):
        query = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
        candidates = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
        map_score = mean_average_precision(query, candidates)
        assert 0.0 <= map_score <= 1.0


class TestPairF1Metrics:
    def test_perfect_separation(self):
        # 正对相似度全为 0.9，负对全为 0.1 → F1=1.0
        pos_sims = torch.full((10,), 0.9)
        neg_sims = torch.full((10,), 0.1)
        result = pair_f1_metrics(pos_sims, neg_sims)
        assert result["pair_f1"] == pytest.approx(1.0)
        assert result["pair_precision"] == pytest.approx(1.0)
        assert result["pair_recall"] == pytest.approx(1.0)

    def test_no_separation(self):
        # 正负对相似度完全相同 → F1 约 0.5
        sims = torch.full((20,), 0.5)
        result = pair_f1_metrics(sims, sims)
        assert 0.0 <= result["pair_f1"] <= 1.0

    def test_output_keys(self):
        pos_sims = torch.rand(10)
        neg_sims = torch.rand(10)
        result = pair_f1_metrics(pos_sims, neg_sims)
        assert "pair_precision" in result
        assert "pair_recall" in result
        assert "pair_f1" in result
        assert "optimal_threshold" in result

    def test_threshold_in_range(self):
        pos_sims = torch.rand(10)
        neg_sims = torch.rand(10)
        result = pair_f1_metrics(pos_sims, neg_sims)
        assert 0.0 <= result["optimal_threshold"] <= 1.0
