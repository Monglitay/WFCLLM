"""Tests for wfcllm.watermark.cascade — compound block cascade fallback."""

import json
import pytest
from unittest.mock import MagicMock

from wfcllm.watermark.cascade import CascadeManager, CascadeCheckpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.interceptor import InterceptEvent


class TestCascadeDisabled:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="k", enable_cascade=False)

    @pytest.fixture
    def mgr(self, config):
        return CascadeManager(config)

    def test_disabled_by_default(self, mgr):
        assert not mgr._enabled

    def test_on_compound_no_op(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        ctx.checkpoint.assert_not_called()

    def test_should_cascade_false(self, mgr):
        assert mgr.should_cascade() is False

    def test_cascade_returns_none(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        assert mgr.cascade(ctx) is None


class TestCascadeEnabled:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="k", enable_cascade=True, cascade_max_depth=2,
        )

    @pytest.fixture
    def mgr(self, config):
        return CascadeManager(config)

    def test_stores_compound_checkpoint(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.last_block_checkpoint = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        assert len(mgr._stack) == 1

    def test_skips_when_no_block_checkpoint(self, mgr):
        """on_compound_block_start does nothing if last_block_checkpoint is None."""
        ctx = MagicMock(spec=GenerationContext)
        ctx.last_block_checkpoint = None
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        assert len(mgr._stack) == 0

    def test_records_failed_simple_block(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.last_block_checkpoint = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        assert mgr._stack[-1].failed_simple_blocks == [
            {
                "block_text": "x = 1",
                "block_ordinal": None,
            }
        ]

    def test_should_cascade_true_after_failure(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.last_block_checkpoint = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        assert mgr.should_cascade() is True

    def test_should_cascade_false_no_failures(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.last_block_checkpoint = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        assert mgr.should_cascade() is False

    def test_cascade_pops_stack_and_rollbacks(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        cp = MagicMock(spec=Checkpoint)
        ctx.last_block_checkpoint = cp
        event = MagicMock(spec=InterceptEvent)
        event.node_type = "if_statement"
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        result = mgr.cascade(ctx)
        assert result is not None
        assert isinstance(result, CascadeCheckpoint)
        ctx.rollback.assert_called_once_with(cp)
        assert len(mgr._stack) == 0

    def test_cascade_exposes_serializable_replacement_metadata(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        cp = MagicMock(spec=Checkpoint)
        ctx.last_block_checkpoint = cp
        event = MagicMock(spec=InterceptEvent)
        event.node_type = "if_statement"
        event.parent_node_type = None
        mgr.on_compound_block_start(
            ctx,
            event,
            stats_snapshot={
                "total_blocks": 1,
                "embedded_blocks": 0,
                "failed_blocks": 1,
            },
        )
        mgr.on_simple_block_failed("x = 1", block_ordinal=3)

        result = mgr.cascade(ctx)

        metadata = result.build_diagnostic_metadata(
            restored_stats={
                "total_blocks": 0,
                "embedded_blocks": 0,
                "failed_blocks": 0,
            }
        )

        assert metadata == {
            "triggered": True,
            "compound_node_type": "if_statement",
            "failed_simple_count_before_cascade": 1,
            "replaced_block_ordinals": [3],
            "restored_total_blocks": 0,
            "restored_embedded_blocks": 0,
            "restored_failed_blocks": 0,
        }
        json.dumps(metadata)

        scope = result.build_replacement_scope()
        assert scope["compound_node_type"] == "if_statement"
        assert scope["compound_parent_node_type"] == "module"
        assert scope["replaced_block_ordinals"] == [3]

    def test_max_depth_evicts_oldest(self, mgr):
        """Exceeding max_depth drops the oldest checkpoint."""
        ctx = MagicMock(spec=GenerationContext)
        for i in range(3):
            ctx.last_block_checkpoint = MagicMock(spec=Checkpoint, name=f"cp{i}")
            event = MagicMock(spec=InterceptEvent)
            mgr.on_compound_block_start(ctx, event)
        # max_depth=2, so only last 2 should remain
        assert len(mgr._stack) == 2

    def test_uses_block_start_checkpoint_not_current(self, mgr):
        """cascade checkpoint must be at block START (last_block_checkpoint),
        not at the current context position. This prevents the stale-snapshot
        ValueError when the retry loop rolls back ctx to an earlier position."""
        ctx = MagicMock(spec=GenerationContext)
        block_start_cp = MagicMock(spec=Checkpoint, name="block_start")
        current_cp = MagicMock(spec=Checkpoint, name="current_end")
        ctx.last_block_checkpoint = block_start_cp
        # ctx.checkpoint() would return a different (later) checkpoint
        ctx.checkpoint.return_value = current_cp
        event = MagicMock(spec=InterceptEvent)
        event.node_type = "for_statement"
        mgr.on_compound_block_start(ctx, event)
        assert len(mgr._stack) == 1
        # The stored checkpoint must be block_start_cp, not current_cp
        assert mgr._stack[0].checkpoint is block_start_cp
        ctx.checkpoint.assert_not_called()

    def test_cascade_empty_stack(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        assert mgr.cascade(ctx) is None

    def test_no_compound_blocks_seen(self, mgr):
        assert mgr.should_cascade() is False


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
