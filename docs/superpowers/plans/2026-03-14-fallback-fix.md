# Fallback Path Fix: 废弃 passive fallback，改用 cascade 重生成 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复嵌入端 passive fallback 路径使用 compound block 中间态文本的问题，确保嵌入端验证的 simple block 文本与提取端 AST 解析文本严格一致。

**Architecture:**
`_try_passive_fallback` 在 compound block 首次进入增量 AST 时触发，验证的是仅含 header 的不完整文本；提取端 `ast_parser.extract_statement_blocks()` 解析完整代码，看到的是含完整 body 的文本。两者永远不同，导致 passive fallback 的"嵌入成功"无法被检测端感知。

**修复方向：** 废弃 `_try_passive_fallback`；依赖已有的 `CascadeManager` —— cascade 回滚至 compound block 起始后，主循环继续生成，内部 simple blocks 逐个被 interceptor 捕获完整文本，走正常嵌入路径，与提取端完全一致。

**TDD 顺序：** 先写验证新行为的失败测试（Chunk 1），再做代码删除（Chunk 2），最后处理配套工作（Chunk 3-5）。

**Tech Stack:** Python 3.11, pytest, tree-sitter, WatermarkGenerator, CascadeManager

---

## Known Issues（本 PR 不处理，记录备查）

- `_try_cascade`（`generator.py`）在内部循环停在第一个 compound event 时也验证了中间态文本。与 passive fallback 同性质的问题，但 cascade 后主循环继续，内部 simple blocks 仍可独立正确嵌入。本次 PR 不修复，Test 5 中有 `pytest.mark.xfail` 占位测试记录。
- `fallback_blocks` 字段在 `EmbedStats`/`GenerateResult`/`pipeline.py` 中**保留**（backward-compatible，始终为 0，不再被赋值），JSONL 输出中该字段值将始终为 0。这是接受的兼容性权衡。
- `diagnostic_generator.py`（实验代码）在 `_diag_try_passive_fallback` 中也检查了 `self._config.enable_fallback`，Task 8 中一并处理。

---

## Chunk 1: 写失败测试（TDD 第一步）

本 Chunk 只写测试，不改代码。运行后预期全部 FAIL，这是正确的。

### Task 1: 写废弃 fallback 的失败测试

**Files:**
- Test: `tests/watermark/test_generator.py`

- [ ] **Step 1: 在 test_generator.py 末尾追加**

```python
class TestFallbackDeprecated:
    """passive fallback 已废弃：generator 不应再有 _try_passive_fallback。"""

    def test_generator_has_no_try_passive_fallback(self):
        """废弃后 WatermarkGenerator 实例不应存在 _try_passive_fallback 方法。"""
        import torch
        from unittest.mock import MagicMock
        from wfcllm.watermark.generator import WatermarkGenerator
        from wfcllm.watermark.config import WatermarkConfig
        config = WatermarkConfig(secret_key="k", encoder_device="cpu")
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        encoder = MagicMock()
        enc_tok = MagicMock()
        gen = WatermarkGenerator(model, tokenizer, encoder, enc_tok, config)
        assert not hasattr(gen, "_try_passive_fallback"), (
            "WatermarkGenerator._try_passive_fallback 应已删除"
        )

    def test_enable_fallback_field_removed_from_config(self):
        """WatermarkConfig 不再有 enable_fallback 字段。"""
        from wfcllm.watermark.config import WatermarkConfig
        cfg = WatermarkConfig(secret_key="k")
        assert not hasattr(cfg, "enable_fallback"), (
            "WatermarkConfig.enable_fallback 已废弃，应已删除"
        )

    def test_enable_cascade_defaults_true(self):
        """enable_cascade 默认值改为 True。"""
        from wfcllm.watermark.config import WatermarkConfig
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.enable_cascade is True
```

- [ ] **Step 2: 运行，确认全部 FAIL**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py::TestFallbackDeprecated -v
```

Expected: 3 FAIL

---

### Task 2: 写 cascade 文本一致性失败测试

**Files:**
- Test: `tests/watermark/test_cascade.py`

核心测试思路（针对 compound 中间态问题）：
- 只喂到 compound block header 结束后立即采样 event，此时 body 尚未生成，保证中间态
- 与 `extract_statement_blocks` 对完整代码的结果比对，两者必须不同

- [ ] **Step 1: 在 test_cascade.py 末尾追加**

```python
class TestCascadeTextConsistency:
    """cascade 路径的文本一致性：compound 中间态 vs 提取端完整文本。"""

    def test_compound_event_fires_before_body_complete(self):
        """compound event 在 body 尚未生成时就触发，此时文本为不完整中间态。

        证明方法：喂完 header + 换行后立即检查，此时 body 行还没有输入，
        interceptor 已经触发了 compound event（中间态）。
        而 ast_parser 对完整代码（含 body）的结果是完整文本。
        两者必须不同。
        """
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        ic = StatementInterceptor()
        # 只喂 header，不喂 body
        header = "for i in range(n):\n"
        compound_event = None
        for ch in header:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "compound":
                compound_event = ev
                break

        # compound event 必须在 header 结束时就触发
        assert compound_event is not None, (
            "interceptor 应在 compound header 完成后触发 compound event"
        )

        # 对包含完整 body 的代码运行 ast_parser（提取端的视角）
        full_code = "for i in range(n):\n    x = i\n    y = i + 1\n"
        all_blocks = extract_statement_blocks(full_code)
        compound_blocks = [b for b in all_blocks if b.block_type == "compound"]
        assert len(compound_blocks) >= 1

        ev_text = compound_event.block_text.strip()
        final_src = compound_blocks[0].source.strip()

        # 关键断言：中间态文本与最终完整文本不同
        assert ev_text != final_src, (
            f"compound event 应为中间态（不含 body），但等于最终完整文本\n"
            f"event:  {ev_text!r}\n"
            f"source: {final_src!r}"
        )
        # 辅助断言：中间态是完整文本的前缀（确认是"同一个 block 的早期版本"）
        assert final_src.startswith(ev_text), (
            f"compound event 文本应是最终 source 的前缀\n"
            f"event:  {ev_text!r}\n"
            f"source: {final_src!r}"
        )

    def test_after_cascade_rollback_simple_blocks_match_final_ast(self):
        """cascade rollback 后，interceptor 捕获的 simple block 文本
        与对最终代码运行 ast_parser 的结果严格一致。"""
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        ic = StatementInterceptor()
        compound_start_cp = ic.checkpoint()

        # 第一次生成（模拟需要 cascade 回滚的路径）
        for ch in "for i in range(n):\n    x = old_val\n":
            ic.feed_token(ch)

        # Cascade rollback 到 compound block 起始
        ic.rollback(compound_start_cp)

        # 重新生成（cascade 后 LLM 的新输出）
        final_code = "for i in range(n):\n    x = new_val\n    y = i\n"
        events = []
        for ch in final_code:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "simple":
                events.append(ev)

        # 对重生成的完整代码运行 ast_parser（模拟提取端）
        all_blocks = extract_statement_blocks(final_code)
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        assert len(events) == len(simple_blocks), (
            f"interceptor 捕获 {len(events)} 个 simple block，"
            f"ast_parser 找到 {len(simple_blocks)} 个\n"
            f"interceptor: {[e.block_text for e in events]}\n"
            f"ast_parser:  {[b.source for b in simple_blocks]}"
        )
        for ev, blk in zip(events, simple_blocks):
            assert ev.block_text.strip() == blk.source.strip(), (
                f"嵌入端与提取端文本不一致！\n"
                f"interceptor: {ev.block_text!r}\n"
                f"ast_parser:  {blk.source!r}"
            )

    def test_after_cascade_rollback_parent_type_matches_ast(self):
        """cascade rollback 后，simple block 的 parent_node_type 与 ast_parser 一致。"""
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        ic = StatementInterceptor()
        cp = ic.checkpoint()
        for ch in "for i in range(n):\n    x = old\n":
            ic.feed_token(ch)
        ic.rollback(cp)

        final_code = "for i in range(n):\n    result = new_val\n"
        events = []
        for ch in final_code:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "simple":
                events.append(ev)

        all_blocks = extract_statement_blocks(final_code)
        block_by_id = {b.block_id: b for b in all_blocks}
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        assert len(events) == len(simple_blocks)
        for ev, blk in zip(events, simple_blocks):
            if blk.parent_id is not None:
                expected_parent = block_by_id[blk.parent_id].node_type
            else:
                expected_parent = "module"
            assert ev.parent_node_type == expected_parent, (
                f"parent_node_type 不一致！\n"
                f"interceptor: {ev.parent_node_type!r}\n"
                f"ast_parser:  {expected_parent!r}"
            )

    @pytest.mark.xfail(reason="Known issue: _try_cascade verifies compound header (incomplete text). "
                               "To be fixed in a separate PR after passive fallback removal.")
    def test_try_cascade_compound_text_matches_final_ast(self):
        """已知问题占位：_try_cascade 内部验证 compound event 文本（中间态），
        与提取端完整文本不一致。本 PR 不修复，记录为 xfail。"""
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        ic = StatementInterceptor()
        cp = ic.checkpoint()

        # 模拟 _try_cascade 触发：停在第一个 compound event 处
        code = "for i in range(n):\n    x = i\n"
        compound_ev = None
        for ch in code:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "compound":
                compound_ev = ev
                break  # _try_cascade 也在这里停下来验证

        assert compound_ev is not None

        all_blocks = extract_statement_blocks(code)
        compound_blocks = [b for b in all_blocks if b.block_type == "compound"]
        assert len(compound_blocks) >= 1

        # 这个断言在修复 _try_cascade 前会 FAIL（即 xfail）
        assert compound_ev.block_text.strip() == compound_blocks[0].source.strip()
```

- [ ] **Step 2: 运行，确认前三个 FAIL，xfail 标记正确**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_cascade.py::TestCascadeTextConsistency -v
```

Expected:
- `test_compound_event_fires_before_body_complete` — FAIL（cascade 删除前，passive fallback 存在，此测试实际应 PASS，因为 interceptor 行为本身没问题。但 Task 1 失败是 `test_generator.py::TestFallbackDeprecated` 的测试，这里预期都能 PASS）
- 实际上：Task 2 的这三个测试验证的是 interceptor 的行为，在修改任何代码前应该已经 PASS

重要：如果 `test_cascade.py::TestCascadeTextConsistency` 里任何测试在修改代码前就 FAIL，说明 interceptor 本身有问题，需要先修复 interceptor。

---

### Task 3: 补充 rollback 场景 11 测试

**Files:**
- Test: `tests/watermark/test_rollback_scenarios.py`

- [ ] **Step 1: 在 test_rollback_scenarios.py 末尾追加**

```python
class TestScenario11CascadeInnerBlocksTextConsistency:
    """Scenario 11: cascade rollback 后，compound 内部 simple blocks 完整性。

    文档性断言：compound event 文本是最终 source 的前缀（已知行为，passive fallback 不可用原因）。
    正确性断言：cascade 后 simple blocks 与 ast_parser 结果一致（修复核心不变量）。
    """

    def test_compound_event_is_prefix_of_final_source(self):
        """compound event 触发时的 block_text 是最终代码 compound source 的前缀。

        文档性测试：记录 interceptor 对 compound block 的触发时机，
        证明 passive fallback 不可用（中间态文本与最终文本不同）。
        """
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        ic = StatementInterceptor()
        header = "for i in range(n):\n"
        compound_event = None
        for ch in header:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "compound":
                compound_event = ev
                break

        assert compound_event is not None

        full_code = "for i in range(n):\n    total += arr[i]\n"
        all_blocks = extract_statement_blocks(full_code)
        compound_blocks = [b for b in all_blocks if b.block_type == "compound"]
        assert len(compound_blocks) >= 1

        final_src = compound_blocks[0].source.strip()
        ev_text = compound_event.block_text.strip()

        assert final_src.startswith(ev_text), (
            f"compound event 文本应是最终 source 的前缀\n"
            f"event:  {ev_text!r}\n"
            f"source: {final_src!r}"
        )
        assert ev_text != final_src, "compound event 应为中间态（不含完整 body）"

    def test_cascade_inner_simple_blocks_match_ast(self):
        """cascade rollback 后重新生成，内部 simple blocks 文本与最终代码 AST 严格一致。

        正确性断言：嵌入端与提取端文本一致，是 passive fallback 修复的核心不变量。
        """
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        ic = StatementInterceptor()
        cp = ic.checkpoint()

        for ch in "for i in range(n):\n    result = old_expr()\n":
            ic.feed_token(ch)
        ic.rollback(cp)

        new_code = "for i in range(n):\n    result = new_expr()\n    count += 1\n"
        events = []
        for ch in new_code:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "simple":
                events.append(ev)

        assert len(events) >= 1, "cascade 后应至少有一个 simple block"

        all_blocks = extract_statement_blocks(new_code)
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        assert len(events) == len(simple_blocks)
        for ev, blk in zip(events, simple_blocks):
            assert ev.block_text.strip() == blk.source.strip(), (
                f"文本不一致！\n"
                f"interceptor: {ev.block_text!r}\n"
                f"ast_parser:  {blk.source!r}"
            )
```

- [ ] **Step 2: 运行，确认 PASS（验证 interceptor 行为本身是对的）**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_rollback_scenarios.py::TestScenario11CascadeInnerBlocksTextConsistency -v
```

Expected: 2 PASS

如果 FAIL，说明 interceptor 存在 bug，需先修复 `wfcllm/watermark/interceptor.py` 再继续。

---

### Task 4: 写集成测试（embed 与 extract 对齐）

**Files:**
- Test: `tests/watermark/test_generator_integration.py`

- [ ] **Step 1: 在 test_generator_integration.py 末尾追加**

```python
class TestEmbedExtractTextAlignment:
    """集成测试：嵌入端 interceptor 捕获的文本与提取端 ast_parser 结果一致。

    核心不变量：embed 端和 extract 端对同一段代码的 simple block 文本必须相同，
    否则 LSH 签名不同，水印无法被检测。
    """

    def test_simple_block_text_matches_ast_parser(self):
        """interceptor 捕获的 simple block 文本与 ast_parser 结果一致。"""
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        code = "x = 1\ny = x + 2\nreturn y\n"
        ic = StatementInterceptor()
        events = []
        for ch in code:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "simple":
                events.append(ev)

        all_blocks = extract_statement_blocks(code)
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        assert len(events) == len(simple_blocks), (
            f"interceptor: {[e.block_text for e in events]}\n"
            f"ast_parser:  {[b.source for b in simple_blocks]}"
        )
        for ev, blk in zip(events, simple_blocks):
            assert ev.block_text.strip() == blk.source.strip(), (
                f"文本不一致!\ninterceptor: {ev.block_text!r}\nast_parser:  {blk.source!r}"
            )

    def test_simple_block_parent_type_matches_ast_parser(self):
        """interceptor 推断的 parent_node_type 与 ast_parser 解析 parent 一致。"""
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        code = "for i in range(n):\n    x = i\n"
        ic = StatementInterceptor()
        events = []
        for ch in code:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "simple":
                events.append(ev)

        all_blocks = extract_statement_blocks(code)
        block_by_id = {b.block_id: b for b in all_blocks}
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        assert len(events) == len(simple_blocks)
        for ev, blk in zip(events, simple_blocks):
            expected = block_by_id[blk.parent_id].node_type if blk.parent_id else "module"
            assert ev.parent_node_type == expected, (
                f"parent_node_type 不一致!\n"
                f"interceptor: {ev.parent_node_type!r}\n"
                f"ast_parser:  {expected!r}"
            )

    def test_after_cascade_rollback_simple_blocks_match_ast(self):
        """cascade rollback 后重新生成，simple block 文本与最终代码 AST 严格一致。"""
        from wfcllm.watermark.interceptor import StatementInterceptor
        from wfcllm.common.ast_parser import extract_statement_blocks

        ic = StatementInterceptor()
        cp = ic.checkpoint()

        for ch in "for i in range(n):\n    x = old\n":
            ic.feed_token(ch)
        ic.rollback(cp)

        final_code = "for i in range(n):\n    x = new\n    y = i\n"
        events = []
        for ch in final_code:
            ev = ic.feed_token(ch)
            if ev is not None and ev.block_type == "simple":
                events.append(ev)

        all_blocks = extract_statement_blocks(final_code)
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        assert len(events) == len(simple_blocks)
        for ev, blk in zip(events, simple_blocks):
            assert ev.block_text.strip() == blk.source.strip()
```

- [ ] **Step 2: 运行，确认 PASS**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator_integration.py::TestEmbedExtractTextAlignment -v
```

Expected: 3 PASS（这些测试验证 interceptor 已有行为）

- [ ] **Step 3: Commit Chunk 1（所有失败测试已写完）**

```bash
git add tests/watermark/test_generator.py \
  tests/watermark/test_cascade.py \
  tests/watermark/test_rollback_scenarios.py \
  tests/watermark/test_generator_integration.py
git commit -m "$(cat <<'EOF'
test: 写 passive fallback 废弃和 cascade 文本一致性的失败测试

TestFallbackDeprecated：验证废弃后 _try_passive_fallback 不存在、
  enable_fallback 字段不存在、enable_cascade 默认 True（当前全部 FAIL）
TestCascadeTextConsistency：验证 compound event 中间态 vs 完整文本差异，
  cascade rollback 后 simple blocks 与 ast_parser 一致（当前应 PASS）
TestScenario11CascadeInnerBlocksTextConsistency：cascade 路径正确性场景测试
TestEmbedExtractTextAlignment：embed 与 extract 文本对齐集成测试

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 2: 实现代码修改

### Task 5: 修改 WatermarkConfig

**Files:**
- Modify: `wfcllm/watermark/config.py`

- [ ] **Step 1: 修改 config.py**

删除 `enable_fallback: bool = True` 字段；将 `enable_cascade: bool = False` 改为 `enable_cascade: bool = True`。

```python
# 删除整行：
#     enable_fallback: bool = True

# 修改：
    enable_cascade: bool = True
```

- [ ] **Step 2: 运行 Task 1 中的 config 相关测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_generator.py::TestFallbackDeprecated::test_enable_fallback_field_removed_from_config \
  tests/watermark/test_generator.py::TestFallbackDeprecated::test_enable_cascade_defaults_true -v
```

Expected: 2 PASS

---

### Task 6: 修改 WatermarkGenerator，删除 `_try_passive_fallback`

**Files:**
- Modify: `wfcllm/watermark/generator.py`

- [ ] **Step 1: 更新主循环**

找到（约第 132-137 行）：

```python
            if event.block_type == "compound":
                cascade_mgr.on_compound_block_start(ctx, event)
                self._try_passive_fallback(
                    ctx, event, stats, pending_fallbacks
                )
                continue
```

改为：

```python
            if event.block_type == "compound":
                cascade_mgr.on_compound_block_start(ctx, event)
                continue
```

- [ ] **Step 2: 删除 `_try_passive_fallback` 方法**

完整删除 `_try_passive_fallback` 方法（含 docstring，约 20 行）。

保留 `EmbedStats.fallback_blocks` 字段（backward-compatible）；`_try_passive_fallback` 删除后该字段始终为 0，但不从 `EmbedStats` 中删除。

- [ ] **Step 3: 运行废弃测试，确认全部通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py::TestFallbackDeprecated -v
```

Expected: 3 PASS

---

### Task 7: 修复因移除 `enable_fallback` 导致的已有测试失败

**Files:**
- Modify: `tests/watermark/test_config.py`
- Modify: 其他含 `enable_fallback` 引用的文件

`enable_fallback` 的完整影响范围（需逐一检查并修复）：

| 文件 | 内容 | 处理方式 |
|------|------|---------|
| `tests/watermark/test_config.py` | `assert cfg.enable_fallback is True` | 删除该断言 |
| `tests/watermark/test_cascade.py` | `WatermarkConfig(enable_fallback=False)` | 删除该参数 |
| `wfcllm/watermark/generator.py` | `_try_passive_fallback` 中使用（已在 Task 6 删除） | 已处理 |
| `experiment/embed_extract_alignment/diagnostic_generator.py` | `self._config.enable_fallback` | Task 8 中处理 |

`fallback_blocks` 字段（保留，不改）：`EmbedStats`、`GenerateResult.fallback_blocks` 属性、`watermark/pipeline.py` 中的 JSONL 输出均保留，值始终为 0。

- [ ] **Step 1: 找出所有 `enable_fallback` 引用**

```bash
grep -rn "enable_fallback" tests/ wfcllm/ run.py configs/ experiment/
```

- [ ] **Step 2: 逐一修复**

对每处引用：
- 测试中 `WatermarkConfig(enable_fallback=...)` → 删除该参数
- 测试中 `assert cfg.enable_fallback` → 删除该断言
- `run.py` 中 `enable_fallback=wm_cfg.get(...)` → 在 Task 10 中处理

- [ ] **Step 3: 运行 watermark 测试套件**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -v
```

Expected: 所有 PASS

- [ ] **Step 4: Commit Chunk 2**

```bash
git add wfcllm/watermark/config.py wfcllm/watermark/generator.py \
  tests/watermark/test_config.py tests/watermark/test_cascade.py
git commit -m "$(cat <<'EOF'
refactor: 废弃 passive fallback，enable_cascade 默认开启

passive fallback 在 compound block header 刚出现于增量 AST 时验证中间态文本，
与提取端 ast_parser 看到的完整 body 文本不同，嵌入结果永远无法被检测端感知。

改用 cascade：rollback 到 compound block 起始后，主循环继续生成，
内部 simple blocks 各自经 interceptor 捕获完整文本，走正常嵌入路径，
与提取端严格一致。

fallback_blocks 字段保留（backward-compatible，始终为 0）。

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 3: DiagnosticGenerator 同步更新（实验代码）

### Task 8: 更新 DiagnosticGenerator

**Files:**
- Modify: `experiment/embed_extract_alignment/diagnostic_generator.py`
- Test: `tests/experiment/embed_extract_alignment/test_aligner.py`

- [ ] **Step 1: 写失败测试**

在 `tests/experiment/embed_extract_alignment/test_aligner.py` 末尾追加：

```python
class TestDiagnosticGeneratorNoFallback:
    def test_no_diag_try_passive_fallback(self):
        """DiagnosticGenerator 不应有 _diag_try_passive_fallback 方法。"""
        from experiment.embed_extract_alignment.diagnostic_generator import DiagnosticGenerator
        assert not hasattr(DiagnosticGenerator, "_diag_try_passive_fallback"), (
            "DiagnosticGenerator._diag_try_passive_fallback 应已删除（与主系统对齐）"
        )
```

- [ ] **Step 2: 运行，确认 FAIL**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/experiment/embed_extract_alignment/test_aligner.py::TestDiagnosticGeneratorNoFallback -v
```

Expected: FAIL

- [ ] **Step 3: 修改 diagnostic_generator.py**

主循环中删除 `_diag_try_passive_fallback` 调用，删除该方法全文。

同时删除 `if not self._config.enable_fallback` 的检查（在 `_diag_try_passive_fallback` 内，随方法一起删除）。

- [ ] **Step 4: 确认 PASS**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/experiment/embed_extract_alignment/ -v
```

Expected: 所有 PASS

- [ ] **Step 5: Commit Chunk 3**

```bash
git add experiment/embed_extract_alignment/diagnostic_generator.py \
  tests/experiment/embed_extract_alignment/test_aligner.py
git commit -m "$(cat <<'EOF'
refactor: DiagnosticGenerator 同步删除 _diag_try_passive_fallback

与主系统 WatermarkGenerator 保持一致，移除中间态 compound block 验证路径。

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 4: 配置文件和 run.py 更新

### Task 9: 更新 configs/base_config.json

**Files:**
- Modify: `configs/base_config.json`
- Test: `tests/test_run_config.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_run_config.py` 末尾追加：

```python
class TestBaseConfigFallbackCascade:
    def test_no_enable_fallback_in_watermark_config(self):
        """base_config.json watermark 节不应有 enable_fallback 字段（已废弃）。"""
        import json
        from pathlib import Path
        cfg = json.loads(Path("configs/base_config.json").read_text())
        assert "enable_fallback" not in cfg.get("watermark", {}), (
            "base_config.json 的 watermark 节不应再有 enable_fallback"
        )

    def test_enable_cascade_true_in_watermark_config(self):
        """base_config.json watermark 节的 enable_cascade 应为 true。"""
        import json
        from pathlib import Path
        cfg = json.loads(Path("configs/base_config.json").read_text())
        assert cfg.get("watermark", {}).get("enable_cascade") is True
```

- [ ] **Step 2: 运行，确认失败状态（enable_cascade 未设置）**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py::TestBaseConfigFallbackCascade -v
```

- [ ] **Step 3: 修改 base_config.json**

在 `watermark` 节删除 `"enable_fallback"` 行（如存在），添加：

```json
"enable_cascade": true,
"cascade_max_depth": 1,
```

- [ ] **Step 4: 确认 PASS**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py::TestBaseConfigFallbackCascade -v
```

Expected: 2 PASS

---

### Task 10: 更新 run.py

**Files:**
- Modify: `run.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_run.py` 末尾追加：

```python
class TestRunWatermarkConfigNoFallback:
    def test_run_watermark_no_enable_fallback(self):
        """run.py 构建 WatermarkConfig 不传 enable_fallback（已废弃）。"""
        import ast
        from pathlib import Path
        source = Path("run.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "enable_fallback":
                raise AssertionError("run.py 仍传递了已废弃的 enable_fallback 参数")

    def test_run_watermark_has_enable_cascade(self):
        """run.py 构建 WatermarkConfig 传递 enable_cascade。"""
        import ast
        from pathlib import Path
        source = Path("run.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "enable_cascade":
                return
        raise AssertionError("run.py 应传递 enable_cascade 参数给 WatermarkConfig")
```

- [ ] **Step 2: 运行，确认 enable_cascade 那个 FAIL**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py::TestRunWatermarkConfigNoFallback -v
```

- [ ] **Step 3: 修改 run.py 的 WatermarkConfig 构建**

删除：
```python
        enable_fallback=wm_cfg.get("enable_fallback", True),
```

添加：
```python
        enable_cascade=wm_cfg.get("enable_cascade", True),
        cascade_max_depth=wm_cfg.get("cascade_max_depth", 1),
```

- [ ] **Step 4: 确认 PASS**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py::TestRunWatermarkConfigNoFallback -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit Chunk 4**

```bash
git add configs/base_config.json run.py tests/test_run_config.py tests/test_run.py
git commit -m "$(cat <<'EOF'
chore: 配置文件和 run.py 移除 enable_fallback，启用 cascade

configs/base_config.json: 删除 enable_fallback，添加 enable_cascade=true/cascade_max_depth=1
run.py: WatermarkConfig 构建移除 enable_fallback，改传 enable_cascade/cascade_max_depth

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 5: 全量测试验证

### Task 11: 全量测试，确认无回归

- [ ] **Step 1: 运行全量测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v 2>&1 | tail -50
```

Expected: 全部 PASS（xfail 标记的 1 个用例为 xfail，不算 failure）

- [ ] **Step 2: 若有 FAIL，逐一修复**

对每个意外 FAIL：
- 若是过时测试（依赖 enable_fallback）：更新测试，去掉该参数
- 若是实现回归：修复实现

---

## Chunk 6: README 更新

### Task 12: 更新 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新系统架构图阶段二描述**

将：
```
阶段二：生成时水印嵌入
  LLM + 编码器 E → 实时 AST 拦截 → 节点熵 → LSH 超平面派生有效区域集 G
  → LSH 签名检验（sign ∈ G 且 min_margin > γ）→ 拒绝采样回滚 → 含水印代码
```

更新为：
```
阶段二：生成时水印嵌入
  LLM + 编码器 E → 实时 AST 拦截（simple block）→ 节点熵 → LSH 超平面派生有效区域集 G
  → LSH 签名检验（sign ∈ G 且 min_margin > γ）→ 拒绝采样回滚
  → 失败超限 → cascade：回滚至 compound block 起始，重生成内部 simple blocks
  → 含水印代码（嵌入端与提取端均基于最终代码的 AST simple blocks，文本严格一致）
```

- [ ] **Step 2: 更新 Phase 2 API 关键 API 部分**

将 `WatermarkConfig` 说明更新为：

```
- `WatermarkConfig` — 水印嵌入参数（secret_key, embed_dim, margin, lsh_d, lsh_gamma,
  enable_cascade, cascade_max_depth 等）
  注：passive fallback（compound 中间态验证）已废弃；cascade（enable_cascade=True，默认开启）
  是唯一的复合语句兜底路径，保证嵌入文本与提取端完全一致
```

- [ ] **Step 3: 在 Phase 2 示例代码后补充机制说明**

```markdown
**水印嵌入机制：**

- interceptor 实时检测 simple 语句块（`return`、`x = ...` 等）
- 每个 simple block 完成后，用编码器计算 LSH 签名，检验是否落入有效区域集 G
  （由 `secret_key + parent_node_type` 派生，parent_node_type 由生成时 AST 解析确定）
- 签名未通过：回滚该 block 的生成起点，重试最多 `max_retries` 次
- 仍未通过：cascade —— 回滚至外层 compound block 起始，重新生成，内部 simple blocks 再次独立验证
- **关键不变量**：提取端只需代码 + 密钥，节点信息由 AST 解析器独立提取；
  嵌入端和提取端对同一 simple block 使用完全相同的文本，LSH 签名天然对齐
```

- [ ] **Step 4: 确认无遗留 enable_fallback 引用**

```bash
grep -n "enable_fallback\|passive fallback" README.md
```

若有，更新或删除。

- [ ] **Step 5: Commit Chunk 6**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: 更新 README，说明 cascade 嵌入机制和核心不变量

明确提取端只需代码+密钥、节点信息由 AST 解析器独立提取；
移除废弃的 passive fallback 描述；说明 cascade 是唯一兜底路径。

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 附录：关键不变量速查

| 不变量 | 验证测试 |
|--------|---------|
| `_try_passive_fallback` 已从 WatermarkGenerator 删除 | `TestFallbackDeprecated::test_generator_has_no_try_passive_fallback` |
| `enable_fallback` 已从 WatermarkConfig 删除 | `TestFallbackDeprecated::test_enable_fallback_field_removed_from_config` |
| `enable_cascade` 默认 True | `TestFallbackDeprecated::test_enable_cascade_defaults_true` |
| compound event 在 body 未生成时触发（中间态） | `TestCascadeTextConsistency::test_compound_event_fires_before_body_complete` |
| cascade rollback 后 simple block 文本与 ast_parser 一致 | `TestCascadeTextConsistency::test_after_cascade_rollback_simple_blocks_match_final_ast` |
| cascade rollback 后 parent_node_type 与 ast_parser 一致 | `TestCascadeTextConsistency::test_after_cascade_rollback_parent_type_matches_ast` |
| 集成：全链路文本一致性 | `TestEmbedExtractTextAlignment::test_after_cascade_rollback_simple_blocks_match_ast` |
| _try_cascade 中间态问题（已知缺陷） | `TestCascadeTextConsistency::test_try_cascade_compound_text_matches_final_ast` (xfail) |
