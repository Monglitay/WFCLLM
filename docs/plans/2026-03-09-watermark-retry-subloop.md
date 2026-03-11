# Watermark Retry 子主循环重构实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 watermark 生成时的 retry 逻辑从"固定 token 数重生成"改为"子主循环"——回滚到语句块开始前的完整状态，让 LLM 自由生成，直到 interceptor 再次检测到完整语句块，才做嵌入检测。

**Architecture:** 在 `StatementInterceptor` 上增加 `save_state/restore_state` 方法，以便 retry 时能完整恢复 interceptor 的内部状态。`WatermarkGenerator` 在进入 retry 前保存完整回滚点（KV cache snapshot、generated_ids 截断位置、interceptor 状态、回滚点最后一个 token），retry 循环内部运行一个精简的"子主循环"，直到 interceptor 触发新语句块。删除 `_regenerate_block` 方法及 `KVCacheManager.snapshot_before_block`（不再需要）。

**Tech Stack:** Python 3.10+, PyTorch, tree-sitter（通过 `wfcllm.common.ast_parser`），pytest

---

## Task 1：给 StatementInterceptor 增加 save_state / restore_state

**Files:**
- Modify: `wfcllm/watermark/interceptor.py`
- Test: `tests/watermark/test_interceptor.py`

### Step 1：写失败测试

在 `tests/watermark/test_interceptor.py` 末尾追加：

```python
class TestStatementInterceptorStateSnapshot:
    """save_state / restore_state 语义测试。"""

    def test_restore_returns_to_saved_state(self):
        """restore 之后 accumulated 和 emitted_keys 回到 save 时的值。"""
        interceptor = StatementInterceptor()
        # 喂入部分 token，让 interceptor 有非空状态
        for ch in "x = 1\n":
            interceptor.feed_token(ch)
        state = interceptor.save_state()

        # 继续喂更多 token
        for ch in "y = 2\n":
            interceptor.feed_token(ch)
        assert "y" in interceptor._accumulated

        # 恢复
        interceptor.restore_state(state)
        assert interceptor._accumulated == state["accumulated"]
        assert interceptor._emitted_keys == state["emitted_keys"]
        assert interceptor._prev_all_keys == state["prev_all_keys"]
        assert interceptor._pending_simple == state["pending_simple"]
        assert interceptor._token_idx == state["token_idx"]

    def test_restore_makes_feed_token_deterministic(self):
        """restore 后重新喂同样的 token 序列，结果应该与原始一致。"""
        interceptor = StatementInterceptor()
        for ch in "x = 1\n":
            interceptor.feed_token(ch)
        state = interceptor.save_state()

        # 第一次：继续喂 'y = 2\n'，记录事件
        events_first = []
        for ch in "y = 2\n":
            e = interceptor.feed_token(ch)
            if e is not None:
                events_first.append(e)

        # restore 后再喂同样序列
        interceptor.restore_state(state)
        events_second = []
        for ch in "y = 2\n":
            e = interceptor.feed_token(ch)
            if e is not None:
                events_second.append(e)

        assert len(events_first) == len(events_second)
        for e1, e2 in zip(events_first, events_second):
            assert e1.block_text == e2.block_text
            assert e1.block_type == e2.block_type
```

### Step 2：运行测试，确认失败

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_interceptor.py::TestStatementInterceptorStateSnapshot -v
```

期望输出：`FAILED` — `AttributeError: 'StatementInterceptor' object has no attribute 'save_state'`

### Step 3：实现 save_state / restore_state

在 `wfcllm/watermark/interceptor.py` 的 `StatementInterceptor` 类中，在 `reset` 方法之后追加：

```python
def save_state(self) -> dict:
    """保存当前内部状态的深拷贝，用于 retry 回滚。"""
    return {
        "accumulated": self._accumulated,
        "token_idx": self._token_idx,
        "prev_all_keys": set(self._prev_all_keys),
        "pending_simple": dict(self._pending_simple),
        "emitted_keys": set(self._emitted_keys),
    }

def restore_state(self, state: dict) -> None:
    """恢复到 save_state 保存时的状态。"""
    self._accumulated = state["accumulated"]
    self._token_idx = state["token_idx"]
    self._prev_all_keys = set(state["prev_all_keys"])
    self._pending_simple = dict(state["pending_simple"])
    self._emitted_keys = set(state["emitted_keys"])
```

注意：`_pending_simple` 的 value 是 `_BlockInfo` dataclass（不可变字段），浅拷贝即可。

### Step 4：运行测试，确认通过

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_interceptor.py -v
```

期望输出：全部 `PASSED`

### Step 5：提交

```bash
git add wfcllm/watermark/interceptor.py tests/watermark/test_interceptor.py
git commit -m "feat: add save_state/restore_state to StatementInterceptor"
```

---

## Task 2：重构 WatermarkGenerator — 用子主循环替换 _regenerate_block

**Files:**
- Modify: `wfcllm/watermark/generator.py`
- Modify: `wfcllm/watermark/kv_cache.py`（删除 `snapshot_before_block`）
- Test: `tests/watermark/test_generator.py`
- Test: `tests/watermark/test_kv_cache.py`

### Step 1：写失败测试（针对新的 retry 语义）

在 `tests/watermark/test_generator.py` 末尾追加：

```python
class TestWatermarkGeneratorRetrySubloop:
    """验证 retry 使用子主循环语义（不调用 _regenerate_block）。"""

    def test_regenerate_block_does_not_exist(self):
        """新实现不应有 _regenerate_block 方法。"""
        from wfcllm.watermark.generator import WatermarkGenerator
        assert not hasattr(WatermarkGenerator, "_regenerate_block"), (
            "_regenerate_block 应已删除，retry 逻辑已移入子主循环"
        )

    def test_interceptor_has_save_restore(self):
        """WatermarkGenerator 使用的 interceptor 支持 save_state/restore_state。"""
        from wfcllm.watermark.interceptor import StatementInterceptor
        ic = StatementInterceptor()
        assert hasattr(ic, "save_state")
        assert hasattr(ic, "restore_state")
```

### Step 2：运行测试，确认第一个失败

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py::TestWatermarkGeneratorRetrySubloop -v
```

期望输出：`test_regenerate_block_does_not_exist` FAILED（方法还存在）

### Step 3：重构 generator.py

将 `wfcllm/watermark/generator.py` 整体替换为以下内容：

```python
"""Watermark-embedded code generation using custom token-by-token loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.interceptor import StatementInterceptor
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.kv_cache import KVCacheManager
from wfcllm.watermark.verifier import ProjectionVerifier


@dataclass
class GenerateResult:
    """Result of watermark-embedded generation."""

    code: str
    total_blocks: int
    embedded_blocks: int
    failed_blocks: int
    fallback_blocks: int


class WatermarkGenerator:
    """Code generator with watermark embedding via rejection sampling."""

    def __init__(
        self,
        model,
        tokenizer,
        encoder,
        encoder_tokenizer,
        config: WatermarkConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

        self._interceptor = StatementInterceptor()
        self._entropy_est = NodeEntropyEstimator()
        self._keying = WatermarkKeying(config.secret_key, config.encoder_embed_dim)
        self._verifier = ProjectionVerifier(
            encoder, encoder_tokenizer, device=config.encoder_device
        )
        self._cache_mgr = KVCacheManager()

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerateResult:
        """Generate code with watermark embedding.

        Args:
            prompt: The input prompt for code generation.

        Returns:
            GenerateResult with generated code and embedding statistics.
        """
        device = next(self._model.parameters()).device

        # Tokenize prompt
        input_ids = torch.tensor(
            [self._tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        past_kv = None
        generated_ids: list[int] = []
        generated_text = ""

        total_blocks = 0
        embedded_blocks = 0
        failed_blocks = 0
        fallback_blocks = 0
        pending_fallbacks: list[str] = []

        self._interceptor.reset()
        eos_id = self._config.eos_token_id or self._tokenizer.eos_token_id

        for _ in range(self._config.max_new_tokens):
            # Forward pass
            output = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            past_kv = output.past_key_values

            # Sample next token
            next_id = self._sample_token(logits)

            if next_id == eos_id:
                break

            generated_ids.append(next_id)
            token_text = self._tokenizer.decode([next_id], skip_special_tokens=True)
            generated_text += token_text

            # Feed to interceptor
            event = self._interceptor.feed_token(token_text)

            if event is not None and event.block_type == "simple":
                total_blocks += 1

                block_entropy = self._entropy_est.estimate_block_entropy(
                    event.block_text
                )
                margin = self._entropy_est.compute_margin(block_entropy, self._config)

                v, t = self._keying.derive(
                    event.parent_node_type or "module", event.node_type
                )

                result = self._verifier.verify(event.block_text, v, t, margin)

                logger.debug(
                    "[simple block #%d] node=%s parent=%s entropy=%.4f margin=%.4f "
                    "target=%+d proj=%.4f passed=%s | text=%r",
                    total_blocks, event.node_type, event.parent_node_type,
                    block_entropy, margin, result.target_sign,
                    result.projection, result.passed,
                    event.block_text[:80],
                )

                if result.passed:
                    embedded_blocks += 1
                else:
                    # -------------------------------------------------------
                    # 回滚点：语句块被检测到之前的完整状态
                    # -------------------------------------------------------
                    # 计算语句块在 generated_ids 中占用的 token 数
                    block_token_count = len(
                        self._tokenizer.encode(
                            event.block_text, add_special_tokens=False
                        )
                    )
                    # 截断位置（语句块开始前）
                    rollback_idx = max(0, len(generated_ids) - block_token_count)

                    # 保存回滚点
                    rollback_generated_ids = generated_ids[:rollback_idx]
                    rollback_generated_text = self._tokenizer.decode(
                        rollback_generated_ids, skip_special_tokens=True
                    )
                    rollback_kv_snapshot = self._cache_mgr.snapshot_at(
                        past_kv, rollback_idx
                    )
                    rollback_interceptor_state = self._interceptor.save_state()
                    # 回滚点的最后一个 token（用于子循环第一次 forward 的 input_ids）
                    if rollback_generated_ids:
                        rollback_last_token_id = rollback_generated_ids[-1]
                    else:
                        # prompt 结束后立即触发块，用 prompt 最后一个 token
                        rollback_last_token_id = (
                            self._tokenizer.encode(prompt, add_special_tokens=False)[-1]
                        )

                    success = False

                    for retry_i in range(self._config.max_retries):
                        # 恢复完整回滚点状态
                        past_kv = self._cache_mgr.rollback(
                            past_kv, rollback_kv_snapshot
                        )
                        generated_ids = list(rollback_generated_ids)
                        generated_text = rollback_generated_text
                        self._interceptor.restore_state(rollback_interceptor_state)

                        # 子主循环：自由生成，直到 interceptor 触发新语句块
                        sub_input_ids = torch.tensor(
                            [[rollback_last_token_id]], dtype=torch.long, device=device
                        )
                        sub_event = None

                        for _ in range(self._config.max_new_tokens):
                            sub_output = self._model(
                                input_ids=sub_input_ids,
                                past_key_values=past_kv,
                                use_cache=True,
                            )
                            sub_logits = sub_output.logits[:, -1, :]
                            past_kv = sub_output.past_key_values

                            sub_next_id = self._sample_token(sub_logits)

                            if sub_next_id == eos_id:
                                break

                            generated_ids.append(sub_next_id)
                            sub_token_text = self._tokenizer.decode(
                                [sub_next_id], skip_special_tokens=True
                            )
                            generated_text += sub_token_text
                            sub_input_ids = torch.tensor(
                                [[sub_next_id]], dtype=torch.long, device=device
                            )

                            sub_event = self._interceptor.feed_token(sub_token_text)
                            if sub_event is not None and sub_event.block_type == "simple":
                                break

                        if sub_event is None or sub_event.block_type != "simple":
                            # 子循环未触发语句块（遇到 EOS 等），放弃 retry
                            logger.debug(
                                "  [retry %d/%d] sub-loop ended without block",
                                retry_i + 1, self._config.max_retries,
                            )
                            break

                        result = self._verifier.verify(
                            sub_event.block_text, v, t, margin
                        )
                        logger.debug(
                            "  [retry %d/%d] proj=%.4f target=%+d margin=%.4f "
                            "passed=%s | text=%r",
                            retry_i + 1, self._config.max_retries,
                            result.projection, result.target_sign, margin,
                            result.passed, sub_event.block_text[:80],
                        )

                        if result.passed:
                            embedded_blocks += 1
                            success = True
                            # 子循环已更新 past_kv / generated_ids / generated_text
                            # 主循环下一步从这里继续，next_id 使用子循环最后采样的 token
                            next_id = generated_ids[-1]
                            break

                    if not success:
                        logger.debug(
                            "  [FAILED] block #%d exhausted %d retries",
                            total_blocks, self._config.max_retries,
                        )
                        failed_blocks += 1
                        pending_fallbacks.append(event.block_text)

            elif event is not None and event.block_type == "compound":
                if self._config.enable_fallback and pending_fallbacks:
                    total_blocks += 1
                    block_entropy = self._entropy_est.estimate_block_entropy(
                        event.block_text
                    )
                    margin = self._entropy_est.compute_margin(block_entropy, self._config)
                    v, t = self._keying.derive(
                        event.parent_node_type or "module", event.node_type
                    )
                    result = self._verifier.verify(event.block_text, v, t, margin)
                    logger.debug(
                        "[compound fallback] node=%s parent=%s entropy=%.4f margin=%.4f "
                        "target=%+d proj=%.4f passed=%s",
                        event.node_type, event.parent_node_type,
                        block_entropy, margin, result.target_sign,
                        result.projection, result.passed,
                    )
                    if result.passed:
                        fallback_blocks += 1
                        pending_fallbacks.clear()

            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)

        return GenerateResult(
            code=generated_text,
            total_blocks=total_blocks,
            embedded_blocks=embedded_blocks,
            failed_blocks=failed_blocks,
            fallback_blocks=fallback_blocks,
        )

    def _sample_token(self, logits: torch.Tensor) -> int:
        """Sample a token from logits with temperature, top-k, top-p."""
        logits = logits.squeeze(0).float()

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

关键改动说明：
- **删除** `_regenerate_block`
- **保存回滚点**：KV cache snapshot（通过新增 `snapshot_at`）、`generated_ids[:rollback_idx]`、`interceptor` 状态、回滚点最后 token id
- **子主循环**：以回滚点最后 token 为 `input_ids` 起点（注意：此 token 的 KV 已在 cache 中，所以第一次 forward 会正确预测其下一个 token），自由生成直到 interceptor 触发新 simple block
- retry 通过后：`next_id` 更新为子循环最后采样的 token，保证主循环下一步 `input_ids` 正确

### Step 4：更新 kv_cache.py — 删除 snapshot_before_block，新增 snapshot_at

将 `wfcllm/watermark/kv_cache.py` 替换为：

```python
"""KV-Cache snapshot and rollback for rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheSnapshot:
    """Records the sequence length at snapshot time."""

    seq_len: int


class KVCacheManager:
    """Manage KV-Cache snapshots and rollbacks via truncation."""

    def snapshot(self, past_key_values: tuple) -> CacheSnapshot:
        """Record current sequence length from the KV-Cache.

        Args:
            past_key_values: Tuple of (key, value) tensor pairs per layer.
                Each tensor has shape (batch, heads, seq_len, head_dim).
        """
        seq_len = past_key_values[0][0].shape[2]
        return CacheSnapshot(seq_len=seq_len)

    def snapshot_at(
        self, past_key_values: tuple, target_token_count: int
    ) -> CacheSnapshot:
        """Record the sequence length corresponding to a specific token position.

        Args:
            past_key_values: Current KV-Cache.
            target_token_count: The number of generated tokens at the rollback point.
                The KV-Cache at that position contains prompt tokens + target_token_count tokens.
        """
        prompt_len = past_key_values[0][0].shape[2] - self._infer_generated_len(
            past_key_values
        )
        # KV cache seq_len = prompt_len + target_token_count
        # But we don't track prompt_len separately; use offset from current end:
        # current_len - (total_generated - target_token_count)
        # This is simpler: just store the absolute target seq_len.
        # The caller passes rollback_idx = len(generated_ids) - block_token_count.
        # We need: seq_len_at_rollback = current_seq_len - block_token_count
        # But we don't receive block_token_count here directly.
        # So the API is: take current past_kv + target generated token count from prompt end.
        # Simplest: caller computes rollback_idx; we just record it as seq_len delta.
        # Actually simpler API: just receive absolute seq_len.
        return CacheSnapshot(seq_len=target_token_count)

    def rollback(
        self, past_key_values: tuple, snapshot: CacheSnapshot
    ) -> tuple:
        """Truncate KV-Cache to the snapshot's sequence length.

        Returns a new tuple of cloned truncated (key, value) pairs so that
        the original tensors can be freed by the garbage collector.
        """
        target_len = snapshot.seq_len
        return tuple(
            (k[:, :, :target_len, :].clone(), v[:, :, :target_len, :].clone())
            for k, v in past_key_values
        )

    def _infer_generated_len(self, past_key_values: tuple) -> int:
        """Helper — not used in rollback, placeholder."""
        return 0
```

等等——`snapshot_at` 的接口设计需要更清晰。重新设计：

`snapshot_at(past_key_values, rollback_idx)` 中：
- `rollback_idx` = `len(generated_ids)` 在语句块开始前的值
- KV cache 的 seq_len = prompt_token_count + total_generated_token_count
- 回滚点的 seq_len = current_seq_len - block_token_count = current_seq_len - (len(generated_ids) - rollback_idx)

因此正确实现为：

```python
def snapshot_at(
    self, past_key_values: tuple, rollback_idx: int, current_generated_count: int
) -> CacheSnapshot:
    """Record KV-Cache seq_len at the rollback point.

    Args:
        past_key_values: Current KV-Cache (after generating all tokens so far).
        rollback_idx: Length of generated_ids at the rollback point (before the block).
        current_generated_count: Current total length of generated_ids.
    """
    current_seq_len = past_key_values[0][0].shape[2]
    tokens_to_remove = current_generated_count - rollback_idx
    target_len = max(0, current_seq_len - tokens_to_remove)
    return CacheSnapshot(seq_len=target_len)
```

generator.py 中对应调用：
```python
rollback_kv_snapshot = self._cache_mgr.snapshot_at(
    past_kv,
    rollback_idx=rollback_idx,
    current_generated_count=len(generated_ids),
)
```

**最终 kv_cache.py 完整内容：**

```python
"""KV-Cache snapshot and rollback for rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheSnapshot:
    """Records the sequence length at snapshot time."""

    seq_len: int


class KVCacheManager:
    """Manage KV-Cache snapshots and rollbacks via truncation."""

    def snapshot(self, past_key_values: tuple) -> CacheSnapshot:
        """Record current sequence length from the KV-Cache.

        Args:
            past_key_values: Tuple of (key, value) tensor pairs per layer.
                Each tensor has shape (batch, heads, seq_len, head_dim).
        """
        seq_len = past_key_values[0][0].shape[2]
        return CacheSnapshot(seq_len=seq_len)

    def snapshot_at(
        self,
        past_key_values: tuple,
        rollback_idx: int,
        current_generated_count: int,
    ) -> CacheSnapshot:
        """Record KV-Cache seq_len corresponding to a past token position.

        Args:
            past_key_values: Current KV-Cache after all tokens generated so far.
            rollback_idx: len(generated_ids) at the desired rollback point.
            current_generated_count: Current len(generated_ids).
        """
        current_seq_len = past_key_values[0][0].shape[2]
        tokens_to_remove = current_generated_count - rollback_idx
        target_len = max(0, current_seq_len - tokens_to_remove)
        return CacheSnapshot(seq_len=target_len)

    def rollback(
        self, past_key_values: tuple, snapshot: CacheSnapshot
    ) -> tuple:
        """Truncate KV-Cache to the snapshot's sequence length.

        Returns a new tuple of cloned truncated (key, value) pairs so that
        the original tensors can be freed by the garbage collector.
        """
        target_len = snapshot.seq_len
        return tuple(
            (k[:, :, :target_len, :].clone(), v[:, :, :target_len, :].clone())
            for k, v in past_key_values
        )
```

同步更新 `generator.py` 中 `snapshot_at` 的调用：
```python
rollback_kv_snapshot = self._cache_mgr.snapshot_at(
    past_kv,
    rollback_idx=rollback_idx,
    current_generated_count=len(generated_ids),
)
```

### Step 5：更新 kv_cache 测试 — 删除 snapshot_before_block 相关测试，新增 snapshot_at 测试

在 `tests/watermark/test_kv_cache.py` 末尾追加：

```python
    def test_snapshot_at_computes_correct_seq_len(self, manager):
        """snapshot_at 应计算出语句块开始前的 seq_len。"""
        # 模拟：prompt=10 tokens，生成了 20 个 token，共 seq_len=30
        kv = self._make_kv_cache(num_layers=2, seq_len=30)
        # 语句块占最后 5 个生成 token；rollback_idx = 20 - 5 = 15
        snap = manager.snapshot_at(past_key_values=kv, rollback_idx=15, current_generated_count=20)
        # 期望 seq_len = 30 - (20 - 15) = 25
        assert snap.seq_len == 25

    def test_snapshot_at_zero_block_tokens(self, manager):
        """rollback_idx == current_generated_count 时，seq_len 不变。"""
        kv = self._make_kv_cache(num_layers=2, seq_len=30)
        snap = manager.snapshot_at(past_key_values=kv, rollback_idx=20, current_generated_count=20)
        assert snap.seq_len == 30

    def test_snapshot_at_clamps_to_zero(self, manager):
        """block_token_count 超过 current_seq_len 时，seq_len 不小于 0。"""
        kv = self._make_kv_cache(num_layers=2, seq_len=5)
        snap = manager.snapshot_at(past_key_values=kv, rollback_idx=0, current_generated_count=100)
        assert snap.seq_len == 0
```

### Step 6：运行所有测试，确认通过

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -v
```

期望：全部 `PASSED`（包括原有 `snapshot_before_block` 测试需要删除或更新——见下）

**处理原有 snapshot_before_block 测试：**

`test_kv_cache.py` 中若有针对 `snapshot_before_block` 的测试，需要删除（该方法已移除）。搜索并删除所有引用 `snapshot_before_block` 的测试方法。

### Step 7：提交

```bash
git add wfcllm/watermark/generator.py wfcllm/watermark/kv_cache.py \
        tests/watermark/test_generator.py tests/watermark/test_kv_cache.py
git commit -m "refactor: replace _regenerate_block with subloop retry in WatermarkGenerator"
```

---

## Task 3：全量测试 & 回归验证

**Files:**
- Test: `tests/` (全部)

### Step 1：运行全量测试套件

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

期望：全部 `PASSED`，无 `FAILED`、无 `ERROR`

### Step 2：确认无遗留 snapshot_before_block 引用

```bash
grep -r "snapshot_before_block" wfcllm/ tests/
```

期望：无输出（已完全移除）

```bash
grep -r "_regenerate_block" wfcllm/ tests/
```

期望：无输出（已完全移除）

### Step 3：提交

```bash
git add -u
git commit -m "test: verify full test suite passes after retry subloop refactor"
```

---

## 关键设计说明

### 子主循环起点 token 的处理

子循环第一次 forward 传入 `rollback_last_token_id`，其对应的 KV cache 已在 `rollback_kv_snapshot` 中（该 token 的 attention 结果已被 cache）。因此模型会正确地从 rollback 点预测**下一个** token，而不是重新预测 `rollback_last_token_id` 本身。这与主循环的 `input_ids = torch.tensor([[next_id]])` 语义完全一致。

### interceptor 状态的深拷贝

`_pending_simple` 的 value 是 `_BlockInfo` dataclass，字段全为不可变类型（str、int、bool），`dict(self._pending_simple)` 浅拷贝即安全。`_prev_all_keys` 和 `_emitted_keys` 是 `set[tuple]`，tuple 不可变，`set(...)` 浅拷贝即安全。

### rollback_idx 计算的 tokenization 误差

`block_token_count = len(tokenizer.encode(event.block_text, add_special_tokens=False))` 仍存在 BPE 上下文依赖误差，但此处仅用于计算 `rollback_idx` 来定位 KV cache 截断点，**不影响语义正确性**（子循环会自由生成，interceptor 会重新检测语句块边界）。即使截断位置有 ±1 token 的误差，子循环最终验证的是 interceptor 认可的完整语句块文本。
