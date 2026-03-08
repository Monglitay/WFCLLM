# Encoder Evaluation Metrics Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 删除无意义的 `projection_sign_accuracy`，新增 `watermark_sign_consistency`、`mean_reciprocal_rank`、`mean_average_precision`、`pair_f1_metrics` 四个评估函数，并更新训练脚本的评估调用。

**Architecture:** 只改 `evaluate.py`（函数增删）和 `train.py`（调用更新），使用 TDD 先写测试再实现。损失函数、模型结构、数据管道均不动。

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: 更新测试文件——删除旧测试，新增四个新指标的测试

**Files:**
- Modify: `tests/encoder/test_evaluate.py`

**Step 1: 删除旧的 `TestProjectionSignAccuracy` 类，替换为四个新测试类**

将 `tests/encoder/test_evaluate.py` 全部替换为以下内容：

```python
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
        assert mrr < 0.5

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
```

**Step 2: 验证测试失败（函数还未实现）**

```bash
conda run -n WFCLLM pytest tests/encoder/test_evaluate.py -v 2>&1 | head -40
```

期望：`ImportError` 或多个 `FAILED`（`watermark_sign_consistency` 等未定义）

---

### Task 2: 实现四个新评估函数，删除旧函数

**Files:**
- Modify: `wfcllm/encoder/evaluate.py`

**Step 1: 替换整个 `evaluate.py`**

```python
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
```

**Step 2: 运行测试验证通过**

```bash
conda run -n WFCLLM pytest tests/encoder/test_evaluate.py -v
```

期望：所有测试 PASS

**Step 3: Commit**

```bash
git add wfcllm/encoder/evaluate.py tests/encoder/test_evaluate.py
git commit -m "feat: redesign encoder evaluation metrics, replace projection_sign_accuracy with watermark_sign_consistency, add MRR/MAP/pair_F1"
```

---

### Task 3: 更新 `train.py` 的评估调用

**Files:**
- Modify: `wfcllm/encoder/train.py:209-234`

**Step 1: 更新 import 和评估调用段**

将 `train.py` 中 L26-31 的 import 块替换为：

```python
from wfcllm.encoder.evaluate import (
    cosine_separation,
    recall_at_k,
    watermark_sign_consistency,
    mean_reciprocal_rank,
    mean_average_precision,
    pair_f1_metrics,
    save_evaluation_report,
)
```

将 L209-224 的评估计算段（从 `sep_metrics = ...` 到 `sign_acc = ...`）替换为：

```python
    sep_metrics = cosine_separation(anchor_embs, pos_embs, neg_embs)
    r1 = recall_at_k(anchor_embs, pos_embs, k=1)
    r5 = recall_at_k(anchor_embs, pos_embs, k=5)
    r10 = recall_at_k(anchor_embs, pos_embs, k=10)
    mrr = mean_reciprocal_rank(anchor_embs, pos_embs)
    map_score = mean_average_precision(anchor_embs, pos_embs)
    wsc = watermark_sign_consistency(anchor_embs, pos_embs, num_directions=64, seed=42)

    # pair_f1: positive pairs vs negative pairs
    pos_cos = torch.nn.functional.cosine_similarity(anchor_embs, pos_embs, dim=1)
    neg_cos = torch.nn.functional.cosine_similarity(anchor_embs, neg_embs, dim=1)
    f1_metrics = pair_f1_metrics(pos_cos, neg_cos)
```

将 L217-224 的 `eval_metrics` 构建段替换为：

```python
    eval_metrics = {
        **sep_metrics,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "mrr": mrr,
        "map": map_score,
        "watermark_sign_consistency": wsc,
        **f1_metrics,
        **best_metrics,
    }
```

**Step 2: 运行 train 相关测试确认无回归**

```bash
conda run -n WFCLLM pytest tests/encoder/test_train.py tests/encoder/test_evaluate.py -v
```

期望：所有测试 PASS

**Step 3: Commit**

```bash
git add wfcllm/encoder/train.py
git commit -m "feat: update train.py evaluation to use new metrics (MRR, MAP, watermark_sign_consistency, pair_f1)"
```

---

### Task 4: 运行完整测试套件确认无回归

**Step 1: 运行全部测试**

```bash
conda run -n WFCLLM pytest tests/ -v
```

期望：所有测试 PASS，无 FAILED/ERROR

**Step 2: 如有失败，检查错误信息并修复**

常见原因：
- `test_train.py` 中有 mock 了旧的 `projection_sign_accuracy` 的测试 → 更新对应 mock/import
- `test_evaluate.py` import 路径问题 → 确认 `wfcllm/encoder/evaluate.py` 已正确更新

**Step 3: 全部通过后最终 commit**

```bash
git add -p  # 确认无意外文件
git commit -m "test: verify all tests pass after eval metrics redesign"
```
