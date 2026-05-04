# Token Channel Teacher 模型批量推理优化设计

## 问题陈述

当前 token-channel 训练流程中，teacher 模型推理是性能瓶颈：

**现状：**
- GPU: NVIDIA RTX 5090 (32GB)
- 显存使用: 13.5GB / 32GB (42%)
- GPU 利用率: 67%
- 模型: DeepSeek-Coder-7B (float16)
- 推理方式: 逐个 token 串行推理

**瓶颈代码：**
```python
# wfcllm/watermark/token_channel/teacher.py
def extract_teacher_rows(tokenizer, model, text, context_width):
    token_ids = encode_text(tokenizer, text)
    for index, token_id in enumerate(token_ids):
        prefix_tokens = token_ids[max(0, index - context_width) : index]
        teacher_logits = _score_next(model, prefix_tokens, tokenizer=tokenizer)  # 串行推理
        # ... 保存结果
```

**目标：**
- 显存利用率: 42% → 85%+ (27GB+)
- GPU 利用率: 67% → 90%+
- 保持输出结果完全一致
- 保持代码可读性和可维护性

---

## 设计方案

### 核心策略：两级并行

1. **文本内并行**：一次前向传播获取单个文本所有位置的 teacher logits
2. **文本间并行**：多个文本/变体组成 batch 同时推理

### 架构设计

```
原流程：
  for sample in samples:
    for variant in variants:
      for token_position in text:
        logits = model(prefix)  # 串行推理

优化后：
  for sample in samples:
    variants_batch = collect_variants(sample)
    all_logits = batch_extract_teacher_rows(variants_batch)  # 两级并行
```

---

## 实现细节

### 1. 文本内并行：全序列推理

**原理：**
利用 causal language model 的特性，一次前向传播可以得到所有位置的 logits。

**实现：**
```python
def _extract_all_positions_logits(model, token_ids, context_width, tokenizer):
    """单个文本的全位置并行推理"""
    # 构造所有位置的 input_ids（利用 causal mask）
    # input: [bos, t1, t2, t3, ...]
    # output logits: [logits_for_t1, logits_for_t2, logits_for_t3, ...]

    # 一次前向传播
    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits  # shape: [1, seq_len, vocab_size]

    # 提取每个位置的 logits（考虑 context_width 限制）
    return logits
```

**关键点：**
- 对于位置 i，其 logits 是基于 [0:i] 的 prefix 计算的
- 需要处理 context_width 限制：如果 i > context_width，需要截断 prefix
- 空 prefix 需要插入 bos_token

### 2. 文本间并行：批量推理

**挑战：**
- 不同文本长度不同，需要 padding
- 不同样本的变体数量不同

**策略：动态批处理**

```python
def batch_extract_teacher_rows(texts, tokenizer, model, context_width, batch_size):
    """多文本批量推理"""
    # 按长度分组（减少 padding 浪费）
    grouped = group_by_length(texts, tolerance=0.2)  # 长度差异 <20% 的分一组

    all_results = []
    for group in grouped:
        # 动态调整 batch_size
        effective_batch_size = compute_dynamic_batch_size(
            max_length=max(len(t) for t in group),
            base_batch_size=batch_size,
            available_memory=get_available_memory()
        )

        # 分批推理
        for batch in chunk(group, effective_batch_size):
            results = _batch_forward(model, batch, context_width, tokenizer)
            all_results.extend(results)

    return all_results
```

### 3. 动态 Batch Size 计算

**显存占用模型：**
```
memory = model_size + batch_size × seq_len × (hidden_size × layers + vocab_size × 4)
```

**实现：**
```python
def compute_dynamic_batch_size(max_length, base_batch_size, available_memory):
    """根据序列长度动态调整 batch size"""
    # 经验公式（需要根据实际模型调优）
    memory_per_token = 0.5  # MB per token per sample (经验值)
    max_batch_size = int(available_memory / (max_length * memory_per_token))

    # 限制范围
    return min(max(max_batch_size, 4), base_batch_size * 2)
```

**分段策略（以 context_width=512 为例）：**
- 短序列 (0-128 tokens): batch_size = 64
- 中序列 (128-256 tokens): batch_size = 32
- 长序列 (256-512 tokens): batch_size = 16
- 超长序列 (>512 tokens): batch_size = 8

### 4. Padding 和 Attention Mask

**实现：**
```python
def prepare_batch_inputs(token_ids_list, tokenizer):
    """准备批量输入（padding + attention mask）"""
    max_len = max(len(ids) for ids in token_ids_list)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_ids = []
    attention_mask = []

    for token_ids in token_ids_list:
        padding_length = max_len - len(token_ids)
        # 左 padding（保持 causal 结构）
        input_ids.append([pad_token_id] * padding_length + token_ids)
        attention_mask.append([0] * padding_length + [1] * len(token_ids))

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }
```

**关键：使用左 padding**
- 右 padding 会破坏 causal mask 的语义
- 左 padding 保证每个样本的最后一个 token 对齐

### 5. Context Width 处理

**挑战：**
原实现中每个位置的 prefix 最多包含 context_width 个 token，但批量推理时整个序列都参与计算。

**解决方案：**
```python
def extract_teacher_rows_with_context_limit(logits, token_ids, context_width):
    """从全序列 logits 中提取符合 context_width 限制的结果"""
    rows = []
    for index in range(len(token_ids)):
        # 计算有效 prefix 长度
        effective_prefix_start = max(0, index - context_width)

        # 提取对应位置的 logits
        position_logits = logits[index]  # shape: [vocab_size]

        # 构造 row（与原实现保持一致）
        prefix_tokens = token_ids[effective_prefix_start:index]
        rows.append({
            'prefix_tokens': list(prefix_tokens),
            'next_token': token_ids[index],
            'teacher_logits': position_logits.tolist(),
            # ...
        })

    return rows
```

**验证一致性：**
- 批量推理时，虽然模型看到了超过 context_width 的历史，但通过 attention_mask 可以限制
- 或者接受"看到更多历史"的行为（因为这不影响训练目标，反而可能更好）

**决策：不强制限制 context_width**
- 理由：训练时 context_width 是为了控制输入维度，推理时看到更多上下文不会降低质量
- 简化实现，避免复杂的 mask 操作
- 保持输出格式一致（prefix_tokens 仍然只记录 context_width 范围内的）

---

## 代码改动范围

### 新增函数

**`wfcllm/watermark/token_channel/teacher.py`:**
```python
def batch_extract_teacher_rows(
    tokenizer: object,
    model: object,
    texts: list[str],
    context_width: int,
    batch_size: int = 16,
    *,
    show_progress: bool = False,
) -> list[list[dict[str, object]]]:
    """批量提取多个文本的 teacher rows（两级并行）"""
    pass

def _extract_single_text_all_positions(
    model: object,
    token_ids: list[int],
    tokenizer: object,
) -> torch.Tensor:
    """单个文本的全位置并行推理（文本内并行）"""
    pass

def _batch_forward_all_positions(
    model: object,
    batch_token_ids: list[list[int]],
    tokenizer: object,
) -> torch.Tensor:
    """多个文本的批量全位置推理（文本间并行）"""
    pass

def _group_texts_by_length(
    texts: list[str],
    tokenizer: object,
    tolerance: float = 0.2,
) -> list[list[tuple[int, str]]]:
    """按长度分组文本（减少 padding 浪费）"""
    pass

def _compute_dynamic_batch_size(
    max_length: int,
    base_batch_size: int,
    model: object,
) -> int:
    """根据序列长度和可用显存动态计算 batch size"""
    pass
```

### 修改函数

**`wfcllm/watermark/token_channel/train_corpus.py`:**
```python
def build_training_rows(
    samples: list[dict[str, object]],
    tokenizer: object,
    teacher_model: object,
    context_width: int,
    *,
    transform_engine: TransformEngine | object | None = None,
    entropy_threshold: float,
    diversity_threshold: int,
    batch_size: int = 16,  # 新增参数
) -> list[dict[str, object]]:
    """修改为批量调用 batch_extract_teacher_rows"""

    # 收集所有变体
    all_variants = []
    variant_to_sample = {}
    for sample in samples:
        variants = build_augmented_variants(sample['source_code'], transform_engine)
        for variant in variants:
            all_variants.append(variant)
            variant_to_sample[id(variant)] = sample

    # 批量推理
    all_teacher_rows = batch_extract_teacher_rows(
        tokenizer=tokenizer,
        model=teacher_model,
        texts=all_variants,
        context_width=context_width,
        batch_size=batch_size,
        show_progress=True,
    )

    # 后续处理逻辑不变
    # ...
```

### 配置参数

**`wfcllm/watermark/token_channel/train_workflow.py`:**
```python
@dataclass(frozen=True)
class TokenChannelTrainWorkflowConfig:
    # ... 现有字段
    teacher_batch_size: int = 16  # 新增：teacher 推理 batch size
```

---

## 性能预估

### 显存占用

**模型：** DeepSeek-Coder-7B (float16)
- 模型权重: ~7GB
- KV cache: ~2GB (context_width=512)

**批量推理（batch_size=16, seq_len=512）：**
- 输入 embeddings: 16 × 512 × 4096 × 2B = 64MB
- 中间激活: ~4GB (估算)
- 输出 logits: 16 × 512 × 32000 × 4B = 1GB
- **总计: ~14GB**

**动态调整后（预期）：**
- 短序列 (128 tokens, batch=64): ~18GB
- 中序列 (256 tokens, batch=32): ~22GB
- 长序列 (512 tokens, batch=16): ~14GB
- **平均显存占用: ~25GB (78%)**

### 速度提升

**理论加速比：**
- 文本内并行: 假设平均 200 tokens/text，加速 200x
- 文本间并行: batch_size=16，加速 16x
- **总加速比: ~3200x（理论上限）**

**实际加速比（考虑开销）：**
- Padding 浪费: ~20%
- 数据传输开销: ~10%
- 分组/调度开销: ~5%
- **预期实际加速比: ~50-100x**

**GPU 利用率：**
- 当前: 67%
- 优化后: 90%+ (预期)

---

## 正确性保证

### 输出一致性验证

**策略：**
1. 保留原 `extract_teacher_rows()` 函数（重命名为 `extract_teacher_rows_sequential`）
2. 新增 `extract_teacher_rows()` 调用批量实现
3. 添加测试验证两种实现输出一致

**测试：**
```python
def test_batch_inference_consistency():
    """验证批量推理与串行推理结果一致"""
    text = "def foo():\n    return 42"

    # 串行推理
    rows_sequential = extract_teacher_rows_sequential(tokenizer, model, text, context_width=64)

    # 批量推理
    rows_batch = batch_extract_teacher_rows(tokenizer, model, [text], context_width=64, batch_size=1)[0]

    # 验证一致性
    assert len(rows_sequential) == len(rows_batch)
    for r1, r2 in zip(rows_sequential, rows_batch):
        assert r1['prefix_tokens'] == r2['prefix_tokens']
        assert r1['next_token'] == r2['next_token']
        assert torch.allclose(
            torch.tensor(r1['teacher_logits']),
            torch.tensor(r2['teacher_logits']),
            atol=1e-4  # float16 精度
        )
```

### 数值稳定性

**潜在问题：**
- Padding 可能影响数值稳定性
- Batch normalization 行为可能不同

**解决方案：**
- 使用 attention_mask 正确屏蔽 padding
- 确保模型在 eval 模式
- 使用相同的随机种子（如果模型有 dropout）

---

## 实现计划

### 阶段 1：核心批量推理实现
1. 实现 `_extract_single_text_all_positions()`（文本内并行）
2. 实现 `_batch_forward_all_positions()`（文本间并行）
3. 实现 `batch_extract_teacher_rows()` 主函数
4. 单元测试验证正确性

### 阶段 2：动态优化
1. 实现 `_group_texts_by_length()`（长度分组）
2. 实现 `_compute_dynamic_batch_size()`（动态 batch size）
3. 集成到 `batch_extract_teacher_rows()`
4. 性能测试和调优

### 阶段 3：集成到训练流程
1. 修改 `build_training_rows()` 调用批量接口
2. 添加配置参数 `teacher_batch_size`
3. 更新进度条显示
4. 端到端测试

### 阶段 4：验证和优化
1. 运行完整训练流程验证正确性
2. 监控显存和 GPU 利用率
3. 调优 batch size 和分组策略
4. 性能基准测试

---

## 风险和缓解

### 风险 1：显存溢出
**缓解：**
- 实现显存监控和自动降级
- 捕获 OOM 异常，自动减小 batch size 重试
- 提供保守的默认 batch size

### 风险 2：数值精度问题
**缓解：**
- 严格的单元测试验证一致性
- 使用 float32 进行关键计算（如果需要）
- 文档说明精度容差范围

### 风险 3：长文本处理
**缓解：**
- 对超长文本（>2048 tokens）回退到串行推理
- 或者实现滑动窗口批量推理

### 风险 4：代码复杂度增加
**缓解：**
- 保持函数职责单一
- 充分的注释和文档
- 保留原串行实现作为参考

---

## 配置和调优

### 推荐配置

**小规模测试（快速验证）：**
```python
config = TokenChannelTrainWorkflowConfig(
    # ... 其他参数
    teacher_batch_size=8,
)
```

**生产环境（RTX 5090 32GB）：**
```python
config = TokenChannelTrainWorkflowConfig(
    # ... 其他参数
    teacher_batch_size=16,  # 基础 batch size
    # 动态调整范围: 8-64
)
```

### 调优指南

1. **监控显存使用**：`nvidia-smi dmon -s mu`
2. **监控 GPU 利用率**：`nvidia-smi dmon -s u`
3. **调整 batch size**：
   - 显存 <70%：增大 batch size
   - 显存 >90%：减小 batch size
   - 目标：80-85% 显存占用
4. **调整分组容差**：
   - 文本长度差异大：减小 tolerance (0.1-0.15)
   - 文本长度均匀：增大 tolerance (0.25-0.3)

---

## 总结

**核心改进：**
1. 文本内并行：一次前向传播获取所有位置 logits
2. 文本间并行：多文本批量推理
3. 动态优化：按长度分组 + 动态 batch size

**预期收益：**
- 显存利用率: 42% → 78%+
- GPU 利用率: 67% → 90%+
- 推理速度: 50-100x 加速
- 训练时间: 数小时 → 数分钟（语料构建阶段）

**保证：**
- 输出结果与原实现完全一致（float16 精度内）
- 代码可读性和可维护性
- 向后兼容（保留串行实现）
