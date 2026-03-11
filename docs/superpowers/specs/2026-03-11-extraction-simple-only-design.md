# 提取模块重构：Simple-Block-Only 检测

## 问题诊断

在 HumanEval 164 样本上，嵌入率中位数 100%、均值 76.9%，但检测率仅 8.5%（14/164）。

根本原因：**提取端对 compound block 评分引入了噪声**。

### Bug 描述

- **嵌入端**（`generator.py:132-137`）：只对 simple block 做常规验证和重试嵌入。compound block 仅作为可选的被动 fallback（`_try_passive_fallback`，当 `enable_fallback=True` 且有 pending 失败块时才触发，此时会计入 `total_blocks`），但这不是常规嵌入路径。
- **提取端**（`scorer.py:32-33`）：`score_all()` 对所有 block（含 compound）评分。compound block 未被嵌入水印，其 hit rate ≈ 50%（随机），本质上是噪声。
- **DP selector 放大了问题**：一个随机命中（score=1）的 compound block 可能在 DP 中替代其多个已嵌入水印的 simple block 子块，将多个独立样本替换为一个噪声样本。

### 数据佐证

| 指标 | 值 |
|------|-----|
| 总样本 | 164 |
| 检测通过 | 14 (8.5%) |
| 近失败 (z∈[2.0, 2.79)) | 23 个 |
| blocks≥10 平均命中率 | 67.9% |
| blocks<5 平均命中率 | 59.2% |
| 被检测样本最少 independent_blocks | 8 |

## 设计方案

### 核心原则

提取端应忠实反映嵌入端的行为：嵌入端只对 simple block 嵌入水印，提取端就只对 simple block 评分检测。

### 关键洞察

Simple block 是 AST 叶子节点，**天然互不嵌套**。因此：
- 不需要 DP 去重即可保证统计独立性
- 所有 simple block 直接构成独立样本集

### 修改范围

3 个生产文件修改 + 对应测试更新。`dp_selector.py` 保留不删除。

#### 1. `wfcllm/extract/detector.py`

- 移除 `self._dp = DPSelector()` 及 `DPSelector` import（不再使用 DP）
- `detect()` 方法：过滤 simple block，只对它们评分，跳过 DP 直接统计

```python
def detect(self, code: str) -> DetectionResult:
    blocks = extract_statement_blocks(code)
    if not blocks:
        return ...empty...

    # 只对 simple block 评分
    simple_blocks = [b for b in blocks if b.block_type == "simple"]
    if not simple_blocks:
        return ...empty...

    # all_blocks 作为第二参数传入，因为 simple block 的 parent_id
    # 可能指向 compound block，_resolve_parent_type 需要完整 block 列表查找
    scores = self._scorer.score_all(simple_blocks, blocks)

    # Simple blocks 天然独立（AST 叶子节点，不互相嵌套），跳过 DP
    result = self._tester.test(scores, total_blocks=len(simple_blocks))
    for s in scores:
        s.selected = True  # 所有 simple block 均被选中（向后兼容）
    result.block_details = scores
    return result
```

#### 2. `wfcllm/extract/scorer.py`

`score_all` 签名变更，接受两个参数：

```python
def score_all(
    self,
    target_blocks: list[StatementBlock],
    all_blocks: list[StatementBlock],
) -> list[BlockScore]:
    """Score target blocks. all_blocks needed for parent_id → node_type lookup."""
    return [self.score_block(b, all_blocks) for b in target_blocks]
```

`score_block` 和 `_resolve_parent_type` 不变。

#### 3. `wfcllm/extract/calibrator.py`

- 移除 `self._dp = DPSelector()` 及 `DPSelector` import
- 同步过滤 simple block，直接统计（不经过 DP）：

```python
simple_blocks = [b for b in blocks if b.block_type == "simple"]
if not simple_blocks:
    continue
scores = self._scorer.score_all(simple_blocks, blocks)
m = len(scores)
x = sum(1 for s in scores if s.score == 1)
```

#### 4. `wfcllm/extract/pipeline.py`

无需代码修改。pipeline 调用 `detector.detect()` ——内部逻辑已变更。

### 语义变化说明

| 字段 | 变更前 | 变更后 |
|------|--------|--------|
| `DetectionResult.total_blocks` | 所有 block 数（含 compound） | simple block 数 |
| `DetectionResult.independent_blocks` | DP 选出的独立块数 | 等于 total_blocks（所有 simple block 天然独立） |
| `DetectionResult.block_details` | 所有 block 的评分 | 仅 simple block 的评分 |
| `BlockScore.selected` | DP 选中为 True | 始终为 True（将来可废弃） |

### 死代码处理

- `detector.py` 中 `self._dp` 和 `DPSelector` import：**移除**
- `calibrator.py` 中 `self._dp` 和 `DPSelector` import：**移除**
- `dp_selector.py` 文件本身：**保留**，将来启用 compound block 嵌入时可重新接入

### 测试变更

需要更新以下测试文件：

- **`tests/extract/test_scorer.py`**：`score_all` 调用改为双参数 `(target_blocks, all_blocks)`
- **`tests/extract/test_detector.py`**：`total_blocks` 断言改为 simple block 数；`block_details` 断言改为仅包含 simple block
- **`tests/extract/test_calibrator.py`**：mock 的 `score_all` 调用模式适配双参数签名；移除 DP 相关 mock

### 保留不变

- `dp_selector.py`：代码保留，不删除。将来启用 compound block 嵌入时可重新接入。
- `hypothesis.py`：不变。
- `ast_parser.py`：不变，仍提取完整层级结构。
- `wfcllm/watermark/*`：完全不动。

## 后续步骤

1. 修改代码
2. 运行测试：`HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v`
3. 重新校准 fpr_threshold（用修复后的提取管线跑负样本）
4. 重新提取/检测（用现有水印数据，不需重新生成）

## 预期效果

- compound block 噪声消除，z-score 整体上升
- 23 个近失败样本中部分翻转为检测通过
- fpr_threshold 可能变化（负样本的 z-score 分布也会变窄）
- 性能提升：compound block 不再走编码器推理

## 已知局限

- **短代码检测力不足**：当 simple block 仅 1-5 个时，即使 100% 命中也难以超过阈值。这是基于统计检验的水印方案的固有限制，不在本次修复范围内。
- **被动 fallback 的不对称性**：嵌入端在 `_try_passive_fallback` 中会验证 compound block 并计入 total_blocks，但提取端不再评分 compound block。这是合理的不对称——fallback 路径产生的水印信号已被其 simple block 子块覆盖。
