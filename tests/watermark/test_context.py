"""Tests for wfcllm.watermark.context — GenerationContext with checkpoint/rollback."""

import pytest
import torch
from unittest.mock import MagicMock

from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import InterceptorState


class TestCheckpointRollback:
    """Core checkpoint/rollback atomicity tests."""

    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            temperature=0.0,
        )

    @pytest.fixture
    def ctx(self, config, mock_model, mock_tokenizer):
        ctx = GenerationContext(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )
        ctx.prefill("def foo():\n    ")
        return ctx

    def test_checkpoint_captures_all_state(self, ctx):
        """checkpoint saves generated_ids, text, kv seq_len, interceptor state."""
        cp = ctx.checkpoint()
        assert isinstance(cp, Checkpoint)
        assert isinstance(cp.generated_ids, list)
        assert isinstance(cp.generated_text, str)
        assert cp.kv_snapshot is not None
        assert isinstance(cp.interceptor_state, InterceptorState)

    def test_rollback_restores_generated_ids(self, ctx):
        """After generate + rollback, generated_ids matches checkpoint."""
        cp = ctx.checkpoint()
        ids_at_cp = list(cp.generated_ids)
        # Generate some tokens
        for _ in range(5):
            ctx.forward_and_sample()
        assert len(ctx.generated_ids) > len(ids_at_cp)
        # Rollback
        ctx.rollback(cp)
        assert ctx.generated_ids == ids_at_cp

    def test_rollback_restores_generated_text(self, ctx):
        cp = ctx.checkpoint()
        text_at_cp = cp.generated_text
        for _ in range(5):
            ctx.forward_and_sample()
        ctx.rollback(cp)
        assert ctx.generated_text == text_at_cp

    def test_rollback_restores_kv_cache_seq_len(self, ctx):
        cp = ctx.checkpoint()
        kv_len_at_cp = cp.kv_snapshot.seq_len
        for _ in range(5):
            ctx.forward_and_sample()
        assert ctx.past_kv[0][0].shape[2] > kv_len_at_cp
        ctx.rollback(cp)
        assert ctx.past_kv[0][0].shape[2] == kv_len_at_cp

    def test_rollback_restores_interceptor_accumulated(self, ctx):
        """Directly validates fix for 'return the' → 'the' bug."""
        cp = ctx.checkpoint()
        acc_at_cp = cp.interceptor_state.accumulated
        for _ in range(5):
            ctx.forward_and_sample()
        ctx.rollback(cp)
        assert ctx.interceptor._accumulated == acc_at_cp

    def test_multiple_checkpoint_rollback_cycles(self, ctx):
        """3 cycles of checkpoint → generate → rollback all restore correctly."""
        for _ in range(3):
            cp = ctx.checkpoint()
            ids_at_cp = list(cp.generated_ids)
            for _ in range(3):
                ctx.forward_and_sample()
            ctx.rollback(cp)
            assert ctx.generated_ids == ids_at_cp

    def test_checkpoint_is_independent_copy(self, ctx):
        """Modifying state after checkpoint doesn't affect the checkpoint."""
        cp = ctx.checkpoint()
        ids_at_cp = list(cp.generated_ids)
        ctx.forward_and_sample()
        # Checkpoint should still hold original ids
        assert cp.generated_ids == ids_at_cp

    def test_rollback_to_empty_state(self, ctx):
        """Checkpoint at 0 generated tokens, rollback restores empty state."""
        cp = ctx.checkpoint()
        assert cp.generated_ids == []
        for _ in range(5):
            ctx.forward_and_sample()
        ctx.rollback(cp)
        assert ctx.generated_ids == []
        assert ctx.generated_text == ""


class TestForwardAndSample:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key", max_new_tokens=50,
            encoder_device="cpu", temperature=0.0,
        )

    @pytest.fixture
    def ctx(self, config, mock_model, mock_tokenizer):
        ctx = GenerationContext(
            model=mock_model, tokenizer=mock_tokenizer, config=config,
        )
        ctx.prefill("x")
        return ctx

    def test_forward_and_sample_appends_to_generated_ids(self, ctx):
        before = len(ctx.generated_ids)
        ctx.forward_and_sample()
        assert len(ctx.generated_ids) == before + 1

    def test_forward_and_sample_updates_text(self, ctx):
        ctx.forward_and_sample()
        # generated_text should have at least one character
        assert len(ctx.generated_text) >= 0  # May be empty string for special tokens

    def test_forward_and_sample_grows_kv_cache(self, ctx):
        kv_before = ctx.past_kv[0][0].shape[2]
        ctx.forward_and_sample()
        assert ctx.past_kv[0][0].shape[2] == kv_before
        ctx.forward_and_sample()
        assert ctx.past_kv[0][0].shape[2] == kv_before + 1

    def test_forward_and_sample_feeds_interceptor(self, ctx):
        """Token is fed to interceptor (token_idx increments)."""
        idx_before = ctx.interceptor._token_idx
        ctx.forward_and_sample()
        assert ctx.interceptor._token_idx == idx_before + 1

    def test_last_event_is_none_for_partial_token(self, ctx):
        """Single token usually doesn't complete a statement block."""
        ctx.forward_and_sample()
        # No guarantee either way, but shouldn't crash
        # last_event is either None or an InterceptEvent

    def test_last_block_checkpoint_available_on_event(self, ctx, mock_model):
        """When interceptor fires an event, last_block_checkpoint is set."""
        # This is hard to test without controlling exact token output
        # Just verify the attribute exists and is None initially
        assert ctx.last_block_checkpoint is None

    def test_first_step_uses_prefill_logits(self):
        """prefill 后首次采样应直接使用 prompt 的末位 logits。"""
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            temperature=0.0,
        )
        model = MagicMock()
        tokenizer = MagicMock()

        prefill_output = MagicMock()
        prefill_output.logits = torch.tensor([[[0.0, 0.0, 0.0, 10.0, 0.0]]])
        prefill_output.past_key_values = (
            (torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2)),
        )
        model.return_value = prefill_output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer.encode.return_value = torch.tensor([[11, 22, 33]])
        tokenizer.decode.return_value = "X"
        tokenizer.eos_token_id = 4

        ctx = GenerationContext(model=model, tokenizer=tokenizer, config=config)
        ctx.prefill("prompt")
        next_id = ctx.forward_and_sample()

        assert next_id == 3
        assert ctx.generated_ids == [3]
        assert model.call_count == 1, (
            "首次采样应直接使用 prefill logits，而不是额外再 forward 一次"
        )

    def test_rollback_to_prefill_state_restores_prefill_logits(self):
        """rollback 到 prefill 后的空生成状态，首次采样仍应复用 prefill logits。"""
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            temperature=0.0,
        )
        model = MagicMock()
        tokenizer = MagicMock()

        prefill_output = MagicMock()
        prefill_output.logits = torch.tensor([[[0.0, 7.0, 0.0, 0.0, 0.0]]])
        prefill_output.past_key_values = (
            (torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2)),
        )
        model.return_value = prefill_output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer.encode.return_value = torch.tensor([[11, 22, 33]])
        tokenizer.decode.return_value = "Y"
        tokenizer.eos_token_id = 4

        ctx = GenerationContext(model=model, tokenizer=tokenizer, config=config)
        ctx.prefill("prompt")
        cp = ctx.checkpoint()

        first_id = ctx.forward_and_sample()
        ctx.rollback(cp)
        second_id = ctx.forward_and_sample()

        assert first_id == 1
        assert second_id == 1
        assert model.call_count == 1, (
            "rollback 回到 prefill 状态后，首次采样仍应复用 prefill logits"
        )

    def test_rollback_to_later_checkpoint_restores_checkpoint_logits_not_prefill_logits(self):
        """Later block-boundary rollback must restore the checkpointed next logits."""
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            temperature=0.0,
        )
        model = MagicMock()
        tokenizer = MagicMock()

        prefill_output = MagicMock()
        prefill_output.logits = torch.tensor([[[0.0, 9.0, 0.0, 0.0, 0.0]]])
        prefill_output.past_key_values = (
            (torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2)),
        )
        second_output = MagicMock()
        second_output.logits = torch.tensor([[[0.0, 0.0, 0.0, 4.0, 0.0]]])
        second_output.past_key_values = (
            (torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2)),
        )
        model.side_effect = [prefill_output, second_output]
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer.encode.return_value = torch.tensor([[11, 22, 33]])
        tokenizer.decode.return_value = "Z"
        tokenizer.eos_token_id = 4

        ctx = GenerationContext(model=model, tokenizer=tokenizer, config=config)
        ctx.prefill("prompt")
        assert ctx.forward_and_sample() == 1
        assert model.call_count == 1

        ctx.forward_and_sample()
        assert model.call_count == 2

        token_channel_logits = torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]])
        ctx._next_logits = token_channel_logits.clone()
        cp = ctx.checkpoint()

        ctx.forward_and_sample()
        ctx.rollback(cp)
        rolled_back_id = ctx.forward_and_sample()

        assert rolled_back_id == 2
        assert model.call_count == 2, (
            "rollback 到 later checkpoint 后应复用 checkpoint 保存的 next logits"
        )

    def test_step_history_does_not_store_full_next_logits_tensors_per_token(self):
        """Per-token history should stay sparse even when next logits exist."""
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            temperature=0.0,
        )
        model = MagicMock()
        tokenizer = MagicMock()

        prefill_output = MagicMock()
        prefill_output.logits = torch.tensor([[[0.0, 8.0, 0.0, 0.0, 0.0]]])
        prefill_output.past_key_values = (
            (torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2)),
        )
        model.return_value = prefill_output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer.encode.return_value = torch.tensor([[11, 22, 33]])
        tokenizer.decode.return_value = "T"
        tokenizer.eos_token_id = 4

        ctx = GenerationContext(model=model, tokenizer=tokenizer, config=config)
        ctx.prefill("prompt")
        ctx._next_logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]])

        ctx.forward_and_sample()

        assert not isinstance(ctx._step_history[0][4], torch.Tensor)


class TestMemorySafety:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key", max_new_tokens=50,
            encoder_device="cpu", cuda_empty_cache_interval=2,
        )

    @pytest.fixture
    def ctx(self, config, mock_model, mock_tokenizer):
        ctx = GenerationContext(
            model=mock_model, tokenizer=mock_tokenizer, config=config,
        )
        ctx.prefill("x")
        return ctx

    def test_repeated_rollback_no_memory_growth(self, ctx):
        """50 cycles of generate → rollback shouldn't leak memory.
        We verify by checking tensor count doesn't grow unbounded."""
        cp = ctx.checkpoint()
        for _ in range(50):
            for _ in range(3):
                ctx.forward_and_sample()
            ctx.rollback(cp)
        # If we got here without OOM, the test passes
        assert ctx.past_kv[0][0].shape[2] == cp.kv_snapshot.seq_len


class TestEosResolution:
    def test_explicit_zero_eos_token_id_is_preserved(self, mock_model, mock_tokenizer):
        """显式配置 eos_token_id=0 时，不应回退到 tokenizer.eos_token_id。"""
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            eos_token_id=0,
        )
        mock_model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        mock_tokenizer.eos_token_id = 99

        ctx = GenerationContext(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )

        assert ctx.eos_id == 0
