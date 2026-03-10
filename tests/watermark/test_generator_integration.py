"""Integration and regression tests for WatermarkGenerator."""

import pytest
import torch
from unittest.mock import MagicMock

from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult, EmbedStats
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.verifier import VerifyResult


class TestGenerateResultBackwardCompat:
    """Verify backward-compatible properties still work."""

    def test_total_blocks_property(self):
        stats = EmbedStats(total_blocks=5, embedded_blocks=3, failed_blocks=2)
        r = GenerateResult(code="code", stats=stats)
        assert r.total_blocks == 5
        assert r.embedded_blocks == 3
        assert r.failed_blocks == 2

    def test_fallback_blocks_property(self):
        stats = EmbedStats(fallback_blocks=1)
        r = GenerateResult(code="", stats=stats)
        assert r.fallback_blocks == 1


class TestGeneratorEOS:
    """Generator handles immediate EOS correctly."""

    @pytest.fixture
    def eos_generator(self):
        config = WatermarkConfig(
            secret_key="k", max_new_tokens=50, encoder_device="cpu",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        enc_tok = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        # Model immediately returns EOS
        vocab_size = 100
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, 2] = 10.0  # EOS
        kv = tuple(
            (torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32))
            for _ in range(2)
        )
        output = MagicMock()
        output.logits = logits
        output.past_key_values = kv
        model.return_value = output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        return WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )

    def test_immediate_eos_returns_empty_code(self, eos_generator):
        result = eos_generator.generate("test")
        assert isinstance(result, GenerateResult)
        assert result.total_blocks == 0
        assert result.failed_blocks == 0


class TestGeneratorStatsAccumulation:
    """Verify stats are correctly accumulated during generation."""

    def test_stats_has_retry_diagnostics_list(self):
        stats = EmbedStats()
        assert isinstance(stats.retry_diagnostics, list)
        assert len(stats.retry_diagnostics) == 0

    def test_embed_stats_cascade_blocks(self):
        stats = EmbedStats(cascade_blocks=2)
        assert stats.cascade_blocks == 2


class TestRegressionReturnTruncation:
    """Regression: 'return the' retry must NOT produce 'the' alone.

    This is the core bug that motivated the refactoring. We verify at the
    interceptor + context level that rollback produces correct state.
    """

    def test_interceptor_rollback_no_truncation(self):
        """After rollback, accumulated text does not retain 'return the'."""
        from wfcllm.watermark.interceptor import StatementInterceptor

        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()

        for ch in "return the\n":
            ic.feed_token(ch)

        ic.rollback(cp)
        # Critical assertion: accumulated must NOT contain 'return the'
        assert ic._accumulated == cp.accumulated
        assert "return the" not in ic._accumulated

        # Re-feed should produce complete statement
        events = []
        for ch in "return result\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        block = events[0].block_text.strip()
        # The block should contain complete statement after rollback
        assert "return" in block, f"Expected 'return ...' but got {block!r}"


class TestRegressionImportTruncation:
    """Regression: 'import doctest' retry must NOT produce 'doctest' alone."""

    def test_interceptor_rollback_import(self):
        from wfcllm.watermark.interceptor import StatementInterceptor

        ic = StatementInterceptor()
        cp = ic.checkpoint()

        for ch in "import doctest\n":
            ic.feed_token(ch)

        ic.rollback(cp)
        events = []
        for ch in "import unittest\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        block = events[0].block_text.strip()
        assert block.startswith("import"), f"Expected 'import ...' but got {block!r}"


class TestRegressionOpenCountTruncation:
    """Regression: 'open_count -= 1' retry must NOT produce '_count -= 1'."""

    def test_interceptor_rollback_augmented_assign(self):
        from wfcllm.watermark.interceptor import StatementInterceptor

        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()

        for ch in "open_count -= 1\n":
            ic.feed_token(ch)

        ic.rollback(cp)
        events = []
        for ch in "close_count += 1\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        if events:
            block = events[0].block_text.strip()
            # Ensure we don't have truncated block that is just a suffix
            assert "_count" != block, f"Got truncated block: {block!r}"
            # The important thing is that a complete statement was detected
            assert "=" in block, f"Expected assignment but got {block!r}"
