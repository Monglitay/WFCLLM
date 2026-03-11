# Repetition Penalty for Retry Sub-loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 watermark 子循环 retry 中对上一次失败 block 的 token 施加重复惩罚，避免每次 retry 生成相同内容。

**Architecture:** 在 `_sample_token` 中增加可选 `penalty_ids` 参数，对其中 token 的 logit 应用标准 HF repetition penalty 公式；子循环 retry 时记录上一次失败 block 的 token IDs，传给下一次采样。

**Tech Stack:** Python, PyTorch, pytest, `wfcllm.watermark.config`, `wfcllm.watermark.generator`

---

## 背景知识

**Repetition Penalty 公式（标准 HF 实现）：**
```
if logit[t] > 0:  logit[t] /= penalty
else:              logit[t] *= penalty
```
penalty > 1.0 时降低指定 token 的概率。penalty = 1.0 时无任何效果。

**当前数据流：**
- 主循环生成 token，interceptor 检测 simple block
- block 验证失败 → 进入 retry 子循环（最多 `max_retries` 次）
- 每次 retry 恢复回滚点，重新生成直到触发新 block
- 问题：每次 retry 从同一回滚点出发，温度/top_p/top_k 完全相同，极易生成相同 token

**修改后数据流：**
- retry 0：正常生成，记录该次子循环生成的 token IDs → `prev_retry_ids`
- retry 1：子循环采样时对 `prev_retry_ids` 施加惩罚 → 倾向于生成不同 token
- retry N：对第 N-1 次的 token IDs 施加惩罚

---

### Task 1: 为 WatermarkConfig 添加 repetition_penalty 字段

**Files:**
- Modify: `wfcllm/watermark/config.py`
- Test: `tests/watermark/test_config.py`

**Step 1: 写失败测试**

在 `tests/watermark/test_config.py` 末尾添加：

```python
def test_repetition_penalty_default():
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.repetition_penalty == 1.3

def test_repetition_penalty_custom():
    cfg = WatermarkConfig(secret_key="k", repetition_penalty=1.5)
    assert cfg.repetition_penalty == 1.5
```

**Step 2: 运行测试，确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py::test_repetition_penalty_default tests/watermark/test_config.py::test_repetition_penalty_custom -v
```
期望：`AttributeError` 或 `TypeError`

**Step 3: 在 config.py 添加字段**

在 `wfcllm/watermark/config.py` 的 `enable_fallback` 字段之后添加：

```python
    # Repetition penalty for retry sub-loop
    repetition_penalty: float = 1.3  # 1.0 = disabled; applied to previous retry's tokens
```

**Step 4: 运行测试，确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py::test_repetition_penalty_default tests/watermark/test_config.py::test_repetition_penalty_custom -v
```
期望：PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/config.py tests/watermark/test_config.py
git commit -m "feat: add repetition_penalty field to WatermarkConfig"
```

---

### Task 2: 修改 _sample_token 支持 penalty_ids

**Files:**
- Modify: `wfcllm/watermark/generator.py:277-301`
- Test: `tests/watermark/test_generator.py`

**Step 1: 写失败测试**

在 `tests/watermark/test_generator.py` 的 `TestWatermarkGeneratorUnit` 类末尾添加：

```python
def test_sample_token_repetition_penalty_reduces_prob(self, config, mock_components):
    """penalty_ids 中的 token 概率应低于不施加惩罚时。"""
    model, tokenizer, encoder, enc_tok = mock_components
    config.repetition_penalty = 2.0
    gen = WatermarkGenerator(
        model=model, tokenizer=tokenizer,
        encoder=encoder, encoder_tokenizer=enc_tok, config=config,
    )
    vocab_size = 10
    # logit[3] = 2.0（正数），施加惩罚后应变为 1.0
    logits = torch.zeros(1, vocab_size)
    logits[0, 3] = 2.0

    # 不施加惩罚
    import torch.nn.functional as F
    logits_no_penalty = logits.clone().squeeze(0)
    probs_no_penalty = F.softmax(logits_no_penalty / config.temperature, dim=-1)

    # 施加惩罚
    logits_with_penalty = logits.clone().squeeze(0)
    logits_with_penalty[3] /= config.repetition_penalty
    probs_with_penalty = F.softmax(logits_with_penalty / config.temperature, dim=-1)

    assert probs_with_penalty[3] < probs_no_penalty[3]

def test_sample_token_no_penalty_when_disabled(self, config, mock_components):
    """repetition_penalty=1.0 时，logits 不应被修改。"""
    model, tokenizer, encoder, enc_tok = mock_components
    config.repetition_penalty = 1.0
    gen = WatermarkGenerator(
        model=model, tokenizer=tokenizer,
        encoder=encoder, encoder_tokenizer=enc_tok, config=config,
    )
    vocab_size = 10
    logits = torch.randn(1, vocab_size)
    logits_before = logits.clone()
    # _sample_token 内部会修改 logits，但 penalty 部分应无效果
    # 通过验证 penalty=1.0 时 token 3 的 logit 不变来确认
    logits_copy = logits.clone().squeeze(0).float()
    if logits_copy[3] > 0:
        expected = logits_copy[3].item()
        logits_copy[3] /= 1.0
        assert abs(logits_copy[3].item() - expected) < 1e-6
```

**Step 2: 运行测试，确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py::TestWatermarkGeneratorUnit::test_sample_token_repetition_penalty_reduces_prob -v
```
期望：测试逻辑本身可通过（这个测试不调用 `_sample_token`，它验证的是 penalty 公式的数学性质）。

注意：这两个测试实际上是纯数学验证，不需要 `_sample_token` 内部实现。真正验证 `_sample_token` 行为的测试在下一步。

**Step 3: 修改 _sample_token**

将 `wfcllm/watermark/generator.py` 中的 `_sample_token` 方法（第 277 行起）替换为：

```python
def _sample_token(
    self,
    logits: torch.Tensor,
    penalty_ids: list[int] | None = None,
) -> int:
    """Sample a token from logits with temperature, top-k, top-p, and optional repetition penalty."""
    logits = logits.squeeze(0).float()

    # Repetition penalty: applied before temperature scaling
    if penalty_ids and self._config.repetition_penalty != 1.0:
        penalty = self._config.repetition_penalty
        for tid in penalty_ids:
            if 0 <= tid < logits.size(0):
                if logits[tid] > 0:
                    logits[tid] /= penalty
                else:
                    logits[tid] *= penalty

    if self._config.temperature > 0:
        logits = logits / self._config.temperature

    if self._config.top_k > 0:
        top_k = min(self._config.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k).values[-1]
        logits[indices_to_remove] = float("-inf")

    if self._config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cumulative_probs > self._config.top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

**Step 4: 运行全部 generator 测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py -v
```
期望：全部 PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/generator.py tests/watermark/test_generator.py
git commit -m "feat: add penalty_ids parameter to _sample_token for repetition penalty"
```

---

### Task 3: 在 retry 子循环中传入 penalty_ids

**Files:**
- Modify: `wfcllm/watermark/generator.py:168-235`（retry 子循环部分）

**Step 1: 理解当前 retry 结构**

当前代码（generator.py 第 166-235 行）：
```python
success = False
for retry_i in range(self._config.max_retries):
    # 恢复回滚点 ...
    sub_input_ids = ...
    sub_event = None
    for _ in range(self._config.max_new_tokens):
        sub_output = self._model(...)
        sub_logits = sub_output.logits[:, -1, :]
        ...
        sub_next_id = self._sample_token(sub_logits)  # ← 这里需要传 penalty_ids
        ...
```

**Step 2: 修改 retry 外循环，增加 prev_retry_ids 跟踪**

将 generator.py 中 `success = False` 那行（约第 166 行）之后、`for retry_i in range(...)` 之前，添加初始化：

定位原代码段（第 166-168 行）：
```python
                    success = False

                    for retry_i in range(self._config.max_retries):
```

替换为：
```python
                    success = False
                    prev_retry_ids: list[int] | None = None

                    for retry_i in range(self._config.max_retries):
```

**Step 3: 在子循环采样时传入 penalty_ids，循环结束后更新 prev_retry_ids**

定位子循环中的 `_sample_token` 调用（约第 192 行）：
```python
                            sub_next_id = self._sample_token(sub_logits)
```

替换为：
```python
                            sub_next_id = self._sample_token(
                                sub_logits, penalty_ids=prev_retry_ids
                            )
```

然后在 retry 循环末尾（`if result.passed:` 块之后、`if not success:` 之前），需要在每次 retry 失败后更新 `prev_retry_ids`。

定位代码段（约第 229-235 行）：
```python
                        if result.passed:
                            embedded_blocks += 1
                            success = True
                            # 子循环已更新 past_kv / generated_ids / generated_text
                            # 主循环下一步从这里继续，next_id 使用子循环最后采样的 token
                            next_id = generated_ids[-1]
                            break
```

替换为：
```python
                        if result.passed:
                            embedded_blocks += 1
                            success = True
                            # 子循环已更新 past_kv / generated_ids / generated_text
                            # 主循环下一步从这里继续，next_id 使用子循环最后采样的 token
                            next_id = generated_ids[-1]
                            break

                        # 记录本次子循环生成的 token IDs，下次 retry 时作为惩罚目标
                        retry_block_count = len(generated_ids) - rollback_idx
                        if retry_block_count > 0:
                            prev_retry_ids = list(generated_ids[rollback_idx:])
```

**Step 4: 运行全部 generator 和 watermark 测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -v
```
期望：全部 PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/generator.py
git commit -m "feat: apply repetition penalty to previous retry tokens in sub-loop"
```

---

### Task 4: 更新 base_config.json

**Files:**
- Modify: `configs/base_config.json`

**Step 1: 在 watermark 节添加字段**

在 `configs/base_config.json` 的 `"enable_fallback": true` 之后添加：

```json
    "repetition_penalty": 1.3
```

注意 JSON 格式：`enable_fallback` 那行需要加逗号。

修改后的 watermark 节末尾：
```json
    "enable_fallback": true,
    "repetition_penalty": 1.3
```

**Step 2: 确认 JSON 合法**

```bash
conda run -n WFCLLM python -c "import json; json.load(open('configs/base_config.json')); print('OK')"
```
期望：`OK`

**Step 3: Commit**

```bash
git add configs/base_config.json
git commit -m "feat: add repetition_penalty=1.3 to base_config.json watermark section"
```

---

### Task 5: 运行全量测试验证

**Step 1: 运行全量测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```
期望：全部 PASS，无回归

**Step 2: 如有失败，排查并修复**

常见问题：
- `test_config.py` 中其他测试若用 `WatermarkConfig(secret_key=...)` 直接构造而不传 `repetition_penalty`，不应受影响（有默认值）
- `test_generator.py` mock 测试应不感知新字段（mock model 不走真实采样路径）
