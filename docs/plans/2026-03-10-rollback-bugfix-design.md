# 回滚机制 Bug 修复设计文档

**日期：** 2026-03-10
**作者：** Claude
**依据日志：** `logs/0309_213321_watermark.log`

---

## 背景

子主循环回滚机制（`fa98407` 起）存在 4 个 bug，导致大量 simple block 嵌入失败，compound fallback 完全无效。本文档描述每个 bug 的根因与修复方案。

---

## Bug 清单与根因

### Bug 1：`rollback_idx` 计算偏差 → 截断变量名

**现象：** `open_braces += 1` retry 全部变成 `_braces += 1`；`result.append(s)` → `append(s)`

**根因：**
```python
block_token_count = len(self._tokenizer.encode(event.block_text, add_special_tokens=False))
```
BPE 分词具有上下文依赖性。`open_braces` 在连续 token 流中分为 `open` + `_braces`（2 tokens），但独立编码为 `open_braces`（1 token）。`block_token_count` 少算 → `rollback_idx` 偏大 → KV 截断后仍包含 block 的第一个 token → 子循环从错误位置开始。

---

### Bug 2：interceptor `_emitted_keys` 污染 → 子循环无法触发事件

**现象：** 大量 `[retry N/50] sub-loop ended without block`（日志中出现 51 次）

**根因：**
```python
# 顺序错误：先触发事件（key 加入 _emitted_keys），再保存状态
rollback_interceptor_state = self._interceptor.save_state()
```
事件触发后，block 的 `(node_type, start_byte, end_byte)` 已在 `_emitted_keys`。子循环 restore 后，`_accumulated` 回到 block 开始前，模型重新生成同一语句，AST 解析出相同的 key，被 `_emitted_keys` 过滤，永远不触发事件。

---

### Bug 3：sub-loop 遇 EOS/无块时 `break` 浪费剩余 retry

**现象：** 第 1 次 sub-loop EOS，直接判 FAILED，后面 49 次 retry 全部放弃。

**根因：**
```python
if sub_event is None or sub_event.block_type != "simple":
    logger.debug("  [retry %d/%d] sub-loop ended without block", ...)
    break  # ← 应为 continue
```

---

### Bug 4：repetition penalty 惩罚结构关键字

**现象：** `import doctest` → retry 全部变成 `doctest`（`import` 被惩罚后概率极低）

**根因：** `prev_retry_ids` 包含整个 retry block 的所有 token，包括 `import`、`return`、`def` 等结构关键字，penalty 使模型跳过它们。

---

## 修复方案

### Fix 1：interceptor 精确追踪 token-字节边界

**文件：** `wfcllm/watermark/interceptor.py`

新增字段 `_token_boundaries: list[int]`，记录每个 token 喂入后的累计 UTF-8 字节偏移量。

```python
# __init__
self._token_boundaries: list[int] = [0]  # boundaries[i] = 前i个token后的字节位置

# feed_token，追加在 self._accumulated += token_text 之后：
self._token_boundaries.append(len(self._accumulated.encode("utf-8")))

# _make_event，用 bisect 精确定位 token 区间：
import bisect
start_tok = bisect.bisect_right(self._token_boundaries, block.start_byte) - 1
end_tok   = bisect.bisect_left(self._token_boundaries, block.end_byte)
# token_start_idx = start_tok，token_count = end_tok - start_tok
```

`save_state` / `restore_state` 同步保存 `_token_boundaries`（list 浅拷贝即安全，元素为 int）。

`InterceptEvent.token_count` 改为精确值。

**generator 改动：**
```python
# 删除：
block_token_count = len(self._tokenizer.encode(event.block_text, add_special_tokens=False))
# 改为：
block_token_count = event.token_count
```

---

### Fix 2：interceptor 在 emit 前保存快照

**文件：** `wfcllm/watermark/interceptor.py`

新增内部机制：每次即将 emit 事件前（`_emitted_keys.add(key)` **之前**）自动保存一份快照到 `self._pre_event_state`。

```python
# 新增字段：
self._pre_event_state: dict | None = None

# 每个 emit 点（_emitted_keys.add(key) 之前）：
self._pre_event_state = self._make_snapshot()
self._emitted_keys.add(key)
...
return self._make_event(block)

# 新增公开方法：
def get_pre_event_state(self) -> dict:
    """返回最近一次事件触发前的状态快照（不含该事件的 key）。"""
    assert self._pre_event_state is not None
    return self._pre_event_state
```

`_make_snapshot()` 是 `save_state()` 的内部版（返回相同结构）。

**generator 改动：**
```python
# 删除：
rollback_interceptor_state = self._interceptor.save_state()
# 改为（在事件触发后立即获取触发前的快照）：
rollback_interceptor_state = self._interceptor.get_pre_event_state()
```

---

### Fix 3：`break` → `continue`

**文件：** `wfcllm/watermark/generator.py`

```python
# 将：
break
# 改为：
continue
```
位于 `if sub_event is None or sub_event.block_type != "simple":` 分支内。

---

### Fix 4：penalty 过滤结构关键字

**文件：** `wfcllm/watermark/generator.py`

在 `WatermarkGenerator.__init__` 构建结构 token ID 集合：

```python
STRUCTURAL_KEYWORDS = [
    "import", "return", "def", "class", "if", "else", "elif",
    "for", "while", "with", "try", "except", "finally", "pass",
    "break", "continue", "raise", "yield", "lambda",
    "and", "or", "not", "in", "is", "from", "as", "assert",
    "del", "global", "nonlocal", "\n", " ", "\t",
]
self._structural_token_ids: set[int] = {
    tid
    for kw in STRUCTURAL_KEYWORDS
    for tid in self._tokenizer.encode(kw, add_special_tokens=False)
}
```

计算 `prev_retry_ids` 时过滤：
```python
prev_retry_ids = [
    tid for tid in generated_ids[rollback_idx:]
    if tid not in self._structural_token_ids
]
```

---

## 涉及文件

| 文件 | 改动类型 |
|---|---|
| `wfcllm/watermark/interceptor.py` | Fix 1 + Fix 2 |
| `wfcllm/watermark/generator.py` | Fix 1 (调用侧) + Fix 3 + Fix 4 |
| `tests/watermark/test_interceptor.py` | Fix 1 + Fix 2 的测试 |
| `tests/watermark/test_generator.py` | Fix 3 + Fix 4 的测试 |

---

## 不涉及的内容

- `kv_cache.py`：逻辑正确，无需修改
- compound fallback：当前仅做检测，不回滚，这是设计意图，不是 bug
- `WatermarkConfig`：无新字段
