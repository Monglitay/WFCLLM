# Watermark Generator 全面重构设计

> 日期：2026-03-10
> 状态：已批准

## 问题背景

当前 `wfcllm/watermark/generator.py` 存在两个核心问题：

1. **回滚状态不一致**：KV Cache 回滚到语句块之前，但 interceptor 的 `_accumulated` 文本仍包含旧语句块前缀，导致 retry 生成残缺语句块
   - `return the` → retry 得到 `the`（丢失 `return`）
   - `import doctest` → retry 得到 `doctest`（丢失 `import`）
   - `open_count -= 1` → retry 得到 `_count -= 1`（丢失 `open`）
2. **重试成功率极低**：50 次重试仍无法产生通过验证的语句块，根因待诊断

## 重构方案：状态机 + Context 对象

将 `generate()` 的 350 行单方法拆解为状态机驱动的协调层，所有可回滚状态由 `GenerationContext` 统一管理。

### 架构总览

```
wfcllm/watermark/
├── generator.py       → 瘦协调层（~80 行主循环）
├── context.py (新)    → GenerationContext：统一状态管理 + checkpoint/rollback
├── retry_loop.py (新) → RetryLoop：独立的重试子循环
├── cascade.py (新)    → CascadeManager：级联回退到复合块（默认关闭）
├── kv_cache.py        → 增强：显存安全 + 删除 snapshot_at()
├── interceptor.py     → 增强：类型化 checkpoint/rollback API
├── config.py          → 新增 enable_cascade / cascade_max_depth
└── (其余模块不变)
```

---

## 模块设计

### 1. GenerationContext — 统一状态管理

**职责**：封装生成过程的所有可变状态，checkpoint/rollback 作为原子操作。

```python
@dataclass
class Checkpoint:
    """回滚点的完整快照。"""
    generated_ids: list[int]
    generated_text: str
    kv_snapshot: CacheSnapshot
    interceptor_state: InterceptorState

class GenerationContext:
    """封装生成过程的所有可变状态。"""

    def __init__(self, model, tokenizer, config, prompt):
        self.generated_ids: list[int] = []
        self.generated_text: str = ""
        self.past_kv = None  # prefill 后初始化
        self.interceptor = StatementInterceptor()
        self._cache_mgr = KVCacheManager()

    def checkpoint(self) -> Checkpoint:
        """原子地保存当前所有状态。"""
        return Checkpoint(
            generated_ids=list(self.generated_ids),
            generated_text=self.generated_text,
            kv_snapshot=self._cache_mgr.snapshot(self.past_kv),
            interceptor_state=self.interceptor.checkpoint(),
        )

    def rollback(self, cp: Checkpoint) -> None:
        """原子地恢复到 checkpoint 时的状态，显式释放 KV Cache。"""
        old_kv = self.past_kv
        self.past_kv = self._cache_mgr.rollback(self.past_kv, cp.kv_snapshot)
        del old_kv
        torch.cuda.empty_cache()
        self.generated_ids = list(cp.generated_ids)
        self.generated_text = cp.generated_text
        self.interceptor.rollback(cp.interceptor_state)

    def forward_and_sample(self, penalty_ids=None) -> int:
        """单步 forward + sample，自动同步更新所有状态。"""
        # model forward → sample → 更新 generated_ids / text / past_kv / interceptor
```

**`forward_and_sample()` 详细规格**：

```python
def prefill(self, prompt: str) -> None:
    """对 prompt 做一次 model forward，初始化 past_kv。在 generate() 开头调用一次。"""
    input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
    outputs = self._model(input_ids=input_ids, use_cache=True)
    self.past_kv = outputs.past_key_values
    self._next_input_id = None  # prefill 不产生 token；第一次 forward_and_sample 用 argmax/sample 从 outputs.logits 取

def forward_and_sample(self, penalty_ids: list[int] | None = None) -> int:
    """单步 forward + sample，自动同步更新所有状态。

    1. input_ids = 上一步 sample 出的 token（1×1 tensor）
    2. model forward with past_kv → logits, new_past_kv
    3. 对 logits 施加 temperature / top_p / top_k / repetition_penalty(penalty_ids)
    4. sample → next_token_id
    5. 同步更新：
       - self.past_kv = new_past_kv
       - self.generated_ids.append(next_token_id)
       - self.generated_text += tokenizer.decode(next_token_id)
       - self.interceptor.feed_token(decoded_text)
       - self.last_event = interceptor 返回的事件（InterceptEvent | None）
    6. return next_token_id
    """

@property
def last_event(self) -> InterceptEvent | None:
    """最近一次 forward_and_sample 中 interceptor 触发的事件。每次 forward 后重置。"""
```

**关键设计决策**：
- **延迟 checkpoint**：不在每个 token 前做 checkpoint（开销过大）。改为在 `forward_and_sample()` 内部、`feed_token()` 之前缓存一个轻量级 `_pre_feed_state`（仅 interceptor 状态 + generated_ids 长度），当 interceptor 实际触发事件时才创建完整 Checkpoint。这样大部分 token（无事件）零开销，仅在检测到语句块时才有拷贝成本
- rollback 中 `torch.cuda.empty_cache()` 可通过 `config.cuda_empty_cache_interval` 控制频率（默认每 10 次 rollback 调用一次），避免热循环中的 CUDA 同步开销
- 所有状态更新通过 `forward_and_sample()` 单一入口

### 2. RetryLoop — 独立的重试子循环

**职责**：从 checkpoint 位置开始，最多重试 max_retries 次，寻找通过验证的语句块。

```python
@dataclass
class RetryResult:
    success: bool
    attempts: int
    final_event: InterceptEvent | None
    diagnostics: RetryDiagnostics

@dataclass
class RetryDiagnostics:
    per_attempt: list[AttemptInfo]
    unique_signatures: int
    unique_texts: int

class RetryLoop:
    """独立的重试子循环。

    构造时注入所有依赖：
    - ctx: GenerationContext — 通过构造函数注入，持有引用
    - config: WatermarkConfig — max_retries, repetition_penalty 等
    - verifier: ProjectionVerifier — LSH 验证
    - keying: WatermarkKeying — 有效集 G 派生
    - entropy_est: NodeEntropyEstimator — 块熵估计
    - structural_token_ids: set[int] — 惩罚时排除的结构关键字 token
    - retry_token_budget: int — 子循环单次 retry 的最大 token 数（默认 max_new_tokens // 2，
      避免单次 retry 耗尽全部 token budget）
    """

    def __init__(self, ctx, config, verifier, keying, entropy_est, structural_token_ids):
        self._ctx = ctx
        self._config = config
        self._verifier = verifier
        # ...

    def run(self, checkpoint: Checkpoint, event: InterceptEvent, verify_result) -> RetryResult:
        """每次 retry：
        1. ctx.rollback(checkpoint) — 原子恢复所有状态
        2. 自由生成直到 interceptor 触发新语句块（受 retry_token_budget 限制）
        3. 验证新语句块
        4. 失败则记录 penalty_ids 供下次使用
        """
```

**与现有代码的关键区别**：
- 每次 retry 开头调用 `ctx.rollback(checkpoint)`，checkpoint 在块检测**之前**保存 — 直接修复残缺语句块 bug
- `RetryDiagnostics` 记录每次尝试的 sig/margin/text，支持根因诊断
- `retry_token_budget` 限制单次 retry 的 token 数，防止子循环耗尽全部生成预算

### 3. CascadeManager — 级联回退到复合块

**职责**：简单块 retry 失败时，可选地回退到包含它的复合块，用复合块整体做水印嵌入。

```python
class CascadeManager:
    """默认关闭 (enable_cascade=False)。"""

    def on_compound_block_start(self, ctx, event):
        """复合块开始时保存 cascade checkpoint。"""

    def on_simple_block_failed(self, block_text):
        """记录 retry 失败的简单块。"""

    def should_cascade(self) -> bool:
        """是否应触发级联回退。"""

    def cascade(self, ctx) -> CascadeCheckpoint | None:
        """弹出栈顶，回滚到复合块开始前。"""
```

**与现有 compound fallback 的区别**：

| | 现有方案 | 新方案 |
|---|---|---|
| 触发时机 | 复合块**结束后**被动检查 | 简单块 retry 失败**立即**触发 |
| 回滚能力 | 无回滚 | 真正回滚到复合块开始前重新生成 |
| 多层支持 | 不支持 | 栈结构，`cascade_max_depth` 层 |
| 默认状态 | `enable_fallback=True` | `enable_cascade=False` |

现有 `enable_fallback` 被动机制保留不动，与 `enable_cascade` 独立。

**Cascade 与 Fallback 的交互决策树**：
```
simple block retry 失败
├─ enable_cascade=True → 主动级联回退（回滚到复合块前，重新生成）
│   └─ cascade retry 也失败 → 标记 failed，继续生成
├─ enable_cascade=False, enable_fallback=True → 不做回滚，等复合块结束后被动检查
└─ 两者都 False → 直接标记 failed，继续生成
```
当两者同时启用时，cascade 优先（因为主动回滚比被动碰运气更可靠）。cascade 成功则跳过 fallback；cascade 失败则 fallback 仍可在复合块结束时尝试。

### 4. KVCacheManager 增强

**改进点**：
- **删除 `snapshot_at()`**：通过"当前 seq_len - token 差值"倒推目标位置的逻辑是状态不同步 bug 的温床。改为在正确时机调用 `snapshot()`
- **显式释放**：rollback 后 `del` 旧 tensors + `torch.cuda.empty_cache()`
- **安全检查**：`snapshot.seq_len > current_len` 时抛出 `ValueError`
- **短路优化**：`target_len == current_len` 时直接返回

### 5. Interceptor API 改进

**改进点**：
- **新增 `checkpoint()` / `rollback()` 公开 API**，返回类型化的 `InterceptorState`（替代 dict）
- **删除 `get_pre_event_state()`**：这是当前 bug 的根源。checkpoint 时机改由 `GenerationContext` 控制
- 旧 API `save_state()` / `restore_state()` 保留为别名（一个版本的过渡期）

```python
@dataclass
class InterceptorState:
    accumulated: str
    token_idx: int
    prev_all_keys: set[tuple]
    pending_simple: dict          # 值为 _BlockInfo 的深拷贝（_BlockInfo 是可变 dataclass）
    emitted_keys: set[tuple]
    token_boundaries: list[int]
```

**`pending_simple` 深拷贝说明**：`_BlockInfo` 是可变 dataclass，`checkpoint()` 时必须对每个值做 `copy.deepcopy()`（而非当前的浅拷贝 `dict(self._pending_simple)`），否则 rollback 后 `_BlockInfo` 对象仍被共享，导致状态污染。

### 6. Generator 重组

重构后的 `generate()` 变成清晰的状态机（~80 行主循环）：

```
while not ctx.is_finished():
    next_id = ctx.forward_and_sample()
    # forward_and_sample 内部：feed_token 前缓存 _pre_feed_state
    # 当 interceptor 触发事件时，自动从 _pre_feed_state 构造完整 Checkpoint

    event = ctx.last_event

    if event is None → continue
    if compound → cascade_mgr.on_compound_block_start() + passive fallback
    if simple → verify → passed? continue : retry_loop.run(ctx.last_block_checkpoint, ...)
        → retry failed? → cascade_mgr.on_simple_block_failed()
            → should_cascade? → cascade
```

**`ctx.last_block_checkpoint`**：仅在 `last_event` 非 None 时有效，是 `forward_and_sample` 内部在 `feed_token` 之前从 `_pre_feed_state` 构造的完整 Checkpoint。这样避免了每个 token 都做完整 checkpoint 的开销。

### 7. 配置变更

`WatermarkConfig` 新增：
```python
enable_cascade: bool = False       # 级联回退，默认关闭
cascade_max_depth: int = 1         # 最大回退层数
cuda_empty_cache_interval: int = 10  # 每 N 次 rollback 调用一次 empty_cache()
retry_token_budget: int | None = None  # 单次 retry 的 token 上限（None 时取 max_new_tokens // 2）
```

`GenerateResult` 增强：
```python
@dataclass
class EmbedStats:
    total_blocks: int = 0
    embedded_blocks: int = 0
    failed_blocks: int = 0
    fallback_blocks: int = 0
    cascade_blocks: int = 0
    retry_diagnostics: list[RetryDiagnostics] = field(default_factory=list)

@dataclass
class GenerateResult:
    code: str
    stats: EmbedStats

    # 向后兼容 property：代理到 stats
    @property
    def total_blocks(self) -> int: return self.stats.total_blocks
    @property
    def embedded_blocks(self) -> int: return self.stats.embedded_blocks
    @property
    def failed_blocks(self) -> int: return self.stats.failed_blocks
    @property
    def fallback_blocks(self) -> int: return self.stats.fallback_blocks
```

---

## 测试设计

### 测试文件结构

```
tests/watermark/
├── test_context.py               # 新增：GenerationContext
├── test_retry_loop.py            # 新增：RetryLoop
├── test_cascade.py               # 新增：CascadeManager
├── test_rollback_scenarios.py    # 新增：场景驱动功能测试
├── test_kv_cache.py              # 增强
├── test_interceptor.py           # 增强
├── test_generator.py             # 重写
├── test_generator_integration.py # 新增：端到端 + 回归测试
├── test_config.py                # 增强
└── (其余不变)
```

### 单元测试

**test_context.py**（GenerationContext）：
- `test_checkpoint_captures_all_state` — checkpoint 后修改任何状态分量，rollback 都恢复
- `test_rollback_restores_exact_interceptor_accumulated` — 回归：rollback 后 accumulated 不含旧块
- `test_rollback_restores_exact_generated_ids` — rollback 后 ids 一致
- `test_rollback_restores_kv_cache_seq_len` — rollback 后 seq_len 一致
- `test_multiple_checkpoint_rollback_cycles` — 多轮 checkpoint/rollback
- `test_checkpoint_is_independent_copy` — 深拷贝验证
- `test_rollback_to_empty_state` — 0 token 时的边界
- `test_forward_and_sample_updates_all_state` — 同步更新
- `test_forward_and_sample_feeds_interceptor` — 自动 feed_token
- `test_forward_and_sample_with_penalty_ids` — 惩罚机制
- `test_rollback_releases_kv_memory` — 显存释放
- `test_repeated_rollback_no_memory_leak` — 无内存泄漏

**test_retry_loop.py**（RetryLoop）：
- 成功路径：首次成功 / 第 N 次成功
- 失败路径：耗尽重试 / 无块生成 / 达到 max_tokens
- 回滚正确性：每次从干净 checkpoint 开始 / 无状态残留
- 惩罚机制：排除结构关键字 / 使用上次 token / 首次无惩罚
- 诊断信息：记录完整 / unique_signatures / unique_texts

**test_cascade.py**（CascadeManager）：
- 默认关闭：no-op 验证
- 正常流程：存储 checkpoint / 记录失败 / 触发判断 / 回滚恢复
- 栈深度：超深淘汰 / pop 行为
- 边界：空栈 / 无复合块

**test_kv_cache.py**（增强）：
- `test_rollback_safety_check_stale_snapshot` — 过期快照检查
- `test_rollback_same_length_returns_same_kv` — 短路优化
- `test_rollback_old_tensors_freed` — 旧 tensor 释放
- `test_snapshot_at_removed` — 废弃 API 确认

**test_interceptor.py**（增强）：
- 新 API：类型化状态 / rollback 恢复 / checkpoint-rollback 等价
- 回归：accumulated 干净 / 重新检测新块 / 保留已 emit 的块
- 废弃 API：`get_pre_event_state` 删除确认

### 场景驱动功能测试

**test_rollback_scenarios.py** — 10 个具体代码样本场景：

| # | 场景 | 验证重点 |
|---|---|---|
| 1 | return 语句回滚 | retry 生成完整 `return result`，非残缺 `result` |
| 2 | import 语句回滚 | 回滚点在 `import` 之前，accumulated 干净 |
| 3 | 赋值语句回滚不污染后续块 | block #1 成功不受 block #2 回滚影响 |
| 4 | if 块内表达式回滚 | 嵌套位置精确回滚，非 `_count -= 1` |
| 5 | 所有重试耗尽 | max_retries 次后正确标记 failed，diagnostics 完整 |
| 6 | 级联回退到 for 循环 | cascade 回滚到复合块前，复合块整体验证 |
| 7 | retry 中遇到 EOS | 不崩溃，正确标记 no_block 继续下一次 |
| 8 | 多 token 语句块回滚精度 | 8 个 token 的块，回滚后 ids/kv/accumulated 精确对齐 |
| 9 | 第一个块就失败 | 回滚到 prompt 末尾（空 generated_ids），边界情况 |
| 10 | 连续两个块都 retry | 两个块的 checkpoint/rollback 完全独立 |

**测试基础设施**：

- `MockLM`：确定性假 LM，根据 `(seq_len, generation_round)` 返回预设 token 序列
- `MockEncoder`：返回固定 embedding，可按文本配置不同返回值
- `RollbackTracer`：包装 GenerationContext，记录每次 checkpoint/rollback 的状态，支持 `assert_rollback_clean()`

### 端到端集成测试

**test_generator_integration.py**：
- 全部通过 / 部分 retry 成功 / 全部失败 / 级联回退 / 被动 fallback
- 内存安全（50 块 × 50 retry）
- EOS 处理 / 空 prompt / 向后兼容
- 3 个回归测试（return/import/open_count 截断 bug）

---

## 不变的模块

以下模块在本次重构中**不涉及改动**：
- `lsh_space.py` / `keying.py` / `verifier.py` / `entropy.py` — LSH 核心逻辑
- `pipeline.py` — 仅适配 `GenerateResult` 的新结构
- `wfcllm/extract/` — 检测侧不受影响
- `wfcllm/common/` — AST 解析不变
- `wfcllm/encoder/` — 编码器不变
