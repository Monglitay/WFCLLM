# Watermark Verbose Logging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在水印嵌入时通过 `logging.DEBUG` 输出每个语句块的完整 LSH 签名信息（签名 tuple、是否在 valid_set、min_margin 等）。

**Architecture:** 扩展 `VerifyResult` 新增 `lsh_signature` 字段（带默认值保持向后兼容），`verify()` 将已计算的 `sig` 放入返回值；`generator.py` 在所有三处语句块处理点（simple block 首次检测、retry 子循环、compound fallback）使用扩展后的 `result` 输出详细 debug 日志。

**Tech Stack:** Python, logging 标准库，已有 `wfcllm.watermark.verifier.VerifyResult`、`ProjectionVerifier`、`WatermarkGenerator`

---

## Chunk 1: 扩展 VerifyResult 并更新日志

### Task 1: 扩展 `VerifyResult`，在 `verifier.py` 中返回 LSH 签名

**Files:**
- Modify: `wfcllm/watermark/verifier.py`
- Test: `tests/watermark/test_verifier.py`

- [ ] **Step 1: 写失败测试**

在 `tests/watermark/test_verifier.py` 末尾新增测试，使用与该文件已有测试相同的 inline 构造风格（`_make_lsh_space`、`_make_encoder_returning`、`_make_tokenizer`）：

```python
def test_verify_result_contains_lsh_signature():
    """VerifyResult should expose the LSH signature used in the decision."""
    lsh = _make_lsh_space(d=3)
    u = torch.randn(128)
    sig = lsh.sign(u)
    valid_set = frozenset([sig])

    verifier = ProjectionVerifier(
        _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
    )
    result = verifier.verify("x = 1", valid_set, margin=0.0)

    assert hasattr(result, "lsh_signature")
    assert isinstance(result.lsh_signature, tuple)
    assert len(result.lsh_signature) == lsh._d  # same d used in construction
    assert all(b in (0, 1) for b in result.lsh_signature)
    assert result.lsh_signature == sig  # must match what lsh.sign(u) returns
```

同时，编辑 `TestVerifyResult` 中已有的两个测试方法，在 `VerifyResult(...)` 构造调用中传入 `lsh_signature=` 参数（只修改这两处构造调用，类结构不变）：

```python
    def test_passed_true(self):
        r = VerifyResult(passed=True, min_margin=0.5, lsh_signature=(1, 0, 1))
        assert r.passed is True

    def test_passed_false(self):
        r = VerifyResult(passed=False, min_margin=0.05, lsh_signature=(0, 1, 0))
        assert r.passed is False
```

- [ ] **Step 2: 运行测试确认新测试失败、旧测试通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_verifier.py -v
```

期望：`test_verify_result_contains_lsh_signature` 失败（`AttributeError: 'VerifyResult' object has no attribute 'lsh_signature'`），其余测试通过。

- [ ] **Step 3: 修改 `wfcllm/watermark/verifier.py`**

1. 在 `VerifyResult` dataclass 新增字段（带默认值保持向后兼容）：

```python
@dataclass
class VerifyResult:
    passed: bool
    min_margin: float
    lsh_signature: tuple[int, ...] = ()
```

2. 在 `ProjectionVerifier.verify()` 中将 `sig` 放入返回值（`sig` 已经在第 57 行计算，只需修改 return 语句）：

```python
        sig = self._lsh_space.sign(u)
        mm = self._lsh_space.min_margin(u)

        passed = (sig in valid_set) and (mm > margin)
        return VerifyResult(passed=passed, min_margin=mm, lsh_signature=sig)
```

- [ ] **Step 4: 运行测试确认全部通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_verifier.py -v
```

期望：全部 7 个测试通过（2 个 `TestVerifyResult` + 4 个 `TestProjectionVerifier` + 1 个新增）。

- [ ] **Step 5: 提交**

```bash
git add wfcllm/watermark/verifier.py tests/watermark/test_verifier.py
git commit -m "feat: expose lsh_signature in VerifyResult"
```

---

### Task 2: 更新 `generator.py` 的 debug 日志，输出完整语句块信息

**Files:**
- Modify: `wfcllm/watermark/generator.py`

- [ ] **Step 1: 替换 simple block 首次检测的日志**

在 `generator.py` 中找到包含 `"[simple block #%d]"` 的 `logger.debug(...)` 调用（位于 `if result.passed:` 之前），将整个调用替换为：

```python
logger.debug(
    "[simple block #%d] node=%s parent=%s entropy=%.4f margin_thresh=%.4f\n"
    "  sig=%s in_valid=%s valid_set_size=%d min_margin=%.4f passed=%s\n"
    "  text=%r",
    total_blocks, event.node_type, event.parent_node_type,
    block_entropy, margin,
    result.lsh_signature,
    result.lsh_signature in valid_set,
    len(valid_set),
    result.min_margin,
    result.passed,
    event.block_text[:80],
)
```

- [ ] **Step 2: 替换 retry 子循环的日志**

在 `generator.py` 中找到包含 `"[retry %d/%d]"` 的 `logger.debug(...)` 调用（位于 `if result.passed:` 之前），将整个调用替换为：

```python
logger.debug(
    "  [retry %d/%d] sig=%s in_valid=%s min_margin=%.4f margin_thresh=%.4f passed=%s\n"
    "  text=%r",
    retry_i + 1, self._config.max_retries,
    result.lsh_signature,
    result.lsh_signature in valid_set,
    result.min_margin,
    margin,
    result.passed,
    sub_event.block_text[:80],
)
```

- [ ] **Step 3: 替换 compound fallback 的日志**

在 `generator.py` 中找到包含 `"[compound fallback]"` 的 `logger.debug(...)` 调用，将整个调用替换为：

```python
logger.debug(
    "[compound fallback] node=%s parent=%s entropy=%.4f margin_thresh=%.4f\n"
    "  sig=%s in_valid=%s valid_set_size=%d min_margin=%.4f passed=%s",
    event.node_type, event.parent_node_type,
    block_entropy, margin,
    result.lsh_signature,
    result.lsh_signature in valid_set,
    len(valid_set),
    result.min_margin,
    result.passed,
)
```

- [ ] **Step 4: 运行全量测试确认无回归**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

期望：全部通过，无新增失败。

- [ ] **Step 5: 提交**

```bash
git add wfcllm/watermark/generator.py
git commit -m "feat: add detailed LSH signature info to watermark embed debug logs"
```
