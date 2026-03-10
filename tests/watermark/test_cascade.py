"""Tests for wfcllm.watermark.cascade — compound block cascade fallback."""

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
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        assert len(mgr._stack) == 1

    def test_records_failed_simple_block(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        assert mgr._stack[-1].failed_simple_blocks == ["x = 1"]

    def test_should_cascade_true_after_failure(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        assert mgr.should_cascade() is True

    def test_should_cascade_false_no_failures(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        assert mgr.should_cascade() is False

    def test_cascade_pops_stack_and_rollbacks(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        cp = MagicMock(spec=Checkpoint)
        ctx.checkpoint.return_value = cp
        event = MagicMock(spec=InterceptEvent)
        event.node_type = "if_statement"
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        result = mgr.cascade(ctx)
        assert result is not None
        assert isinstance(result, CascadeCheckpoint)
        ctx.rollback.assert_called_once_with(cp)
        assert len(mgr._stack) == 0

    def test_max_depth_evicts_oldest(self, mgr):
        """Exceeding max_depth drops the oldest checkpoint."""
        ctx = MagicMock(spec=GenerationContext)
        for i in range(3):
            ctx.checkpoint.return_value = MagicMock(spec=Checkpoint, name=f"cp{i}")
            event = MagicMock(spec=InterceptEvent)
            mgr.on_compound_block_start(ctx, event)
        # max_depth=2, so only last 2 should remain
        assert len(mgr._stack) == 2

    def test_cascade_empty_stack(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        assert mgr.cascade(ctx) is None

    def test_no_compound_blocks_seen(self, mgr):
        assert mgr.should_cascade() is False
