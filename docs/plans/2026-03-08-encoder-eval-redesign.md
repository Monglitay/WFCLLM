# 评估指标重设计方案

**日期：** 2026-03-08
**范围：** `wfcllm/encoder/evaluate.py`、`wfcllm/encoder/train.py`
**目标：** 修复无意义的 `projection_sign_accuracy`，新增更丰富的评估指标

---

## 背景

当前 `projection_sign_accuracy` 使用完全随机的 directions 和 target_bits，期望结果
恒为 ~0.5，无论编码器质量如何，该指标无任何区分度。

---

## 新评估指标设计

### 1. `watermark_sign_consistency`（替换 projection_sign_accuracy）

**语义：** 对同一语义块的不同变体（anchor/positive 对），在 K 个固定方向向量
（watermark key）上，投影符号的一致性比例。

```
对每个 (anchor_emb, positive_emb) 对：
  用 K 个固定 L2-normalized 随机向量 d_1..d_K（seed 固定）
  consistency_i = mean_k( sign(anchor·d_k) == sign(positive·d_k) )
watermark_sign_consistency = mean_i( consistency_i )
```

**期望值：** 编码器越好，正变体语义越近，符号越一致，目标 >0.85。
**意义：** 直接衡量阶段二水印可嵌入性的前提条件。

### 2. `mrr`（Mean Reciprocal Rank）

```
MRR = mean( 1 / rank_i )
```

其中 rank_i 是 query_i 的真实正样本在相似度排名中的位置（1-based）。
比 Recall@K 更细粒度，反映排名质量。

### 3. `map`（Mean Average Precision）

对每个 query，计算其正样本排在前几位的 Average Precision：

```
AP_i = 1 / rank_i    （每个 query 只有一个正样本时等价于 1/rank）
MAP = mean(AP_i)
```

### 4. `pair_f1_metrics`（基于阈值的二分类 F1）

将所有正对（anchor-positive）和负对（anchor-negative）的余弦相似度做二分类：

- **正对标签 = 1**（cos_pos），**负对标签 = 0**（cos_neg）
- 在 [0, 1] 区间扫描阈值，找最大化 F1 的最优阈值
- 输出：`pair_precision`、`pair_recall`、`pair_f1`、`optimal_threshold`

---

## 改动范围

### `wfcllm/encoder/evaluate.py`

- **删除** `projection_sign_accuracy` 函数
- **新增** `watermark_sign_consistency(anchor, positive, num_directions=64, seed=42)`
- **新增** `mean_reciprocal_rank(query_embeddings, candidate_embeddings)`
- **新增** `mean_average_precision(query_embeddings, candidate_embeddings)`
- **新增** `pair_f1_metrics(pos_cos_sims, neg_cos_sims)`

### `wfcllm/encoder/train.py`

- 更新 L210-215 的评估调用：去掉随机 directions/target_bits
- 新增调用上述四个函数
- 更新 `eval_metrics` dict 字段名

### `tests/encoder/test_evaluate.py`

- 删除 `test_projection_sign_accuracy`
- 新增对四个新函数的测试

---

## 不改动

- 损失函数（Triplet Loss）
- 模型架构（SemanticEncoder）
- 数据管道（TripletCodeDataset）
- 训练超参（EncoderConfig）
