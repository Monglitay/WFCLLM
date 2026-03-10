"""Tests for wfcllm.watermark.generator — thin orchestrator."""

import pytest
import torch
from unittest.mock import MagicMock

from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult, EmbedStats
from wfcllm.watermark.config import WatermarkConfig


class TestEmbedStats:
    def test_default_values(self):
        s = EmbedStats()
        assert s.total_blocks == 0
        assert s.embedded_blocks == 0
        assert s.failed_blocks == 0
        assert s.fallback_blocks == 0
        assert s.cascade_blocks == 0
        assert s.retry_diagnostics == []


class TestGenerateResult:
    def test_result_with_stats(self):
        stats = EmbedStats(total_blocks=3, embedded_blocks=2, failed_blocks=1)
        r = GenerateResult(code="x = 1", stats=stats)
        assert r.code == "x = 1"
        assert r.stats.total_blocks == 3

    def test_backward_compat_properties(self):
        stats = EmbedStats(total_blocks=5, embedded_blocks=3, failed_blocks=1, fallback_blocks=1)
        r = GenerateResult(code="", stats=stats)
        assert r.total_blocks == 5
        assert r.embedded_blocks == 3
        assert r.failed_blocks == 1
        assert r.fallback_blocks == 1


class TestWatermarkGeneratorInit:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="test-key", max_new_tokens=50, encoder_device="cpu")

    @pytest.fixture
    def mock_components(self):
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        encoder_tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        return model, tokenizer, encoder, encoder_tokenizer

    def test_generator_init(self, config, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert gen._config == config
        assert hasattr(gen, "_structural_token_ids")
        assert isinstance(gen._structural_token_ids, set)

    def test_sample_token_method_removed(self, config, mock_components):
        """_sample_token moved to GenerationContext._sample."""
        model, tokenizer, encoder, enc_tok = mock_components
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert not hasattr(gen, "_sample_token")

    def test_generate_returns_generate_result(self, config, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        vocab_size = 100
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, tokenizer.eos_token_id] = 10.0
        past_kv = tuple(
            (torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32))
            for _ in range(2)
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.past_key_values = past_kv
        model.return_value = mock_output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        result = gen.generate("test")
        assert isinstance(result, GenerateResult)
        assert isinstance(result.stats, EmbedStats)
        assert isinstance(result.code, str)


class TestStructuralTokenFiltering:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="test-key", max_new_tokens=50, encoder_device="cpu")

    def test_generator_has_structural_token_ids(self, config):
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        enc_tok = MagicMock()
        call_map = {"import": [10], "return": [11], "def": [12]}
        def encode_side_effect(text, **kw):
            return call_map.get(text, [99])
        tokenizer.encode = MagicMock(side_effect=encode_side_effect)
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert 10 in gen._structural_token_ids
        assert 11 in gen._structural_token_ids
