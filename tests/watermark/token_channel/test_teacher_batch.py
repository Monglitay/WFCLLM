"""Tests for batch inference optimization in teacher model."""

from __future__ import annotations

import torch
import pytest


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, bos_token_id: int | None = None):
        self.bos_token_id = bos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2, 3, 4, 5]


class MockModel(torch.nn.Module):
    """Mock model for testing."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = torch.nn.Linear(1, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return {"logits": logits}


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer(bos_token_id=0)


@pytest.fixture
def mock_model():
    return MockModel(vocab_size=100)


def test_extract_single_text_all_positions_shape(mock_model, mock_tokenizer):
    """Test that _extract_single_text_all_positions returns correct shape."""
    from wfcllm.watermark.token_channel.teacher import _extract_single_text_all_positions

    token_ids = [1, 2, 3, 4, 5]
    result = _extract_single_text_all_positions(mock_model, token_ids, mock_tokenizer)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (len(token_ids), mock_model.vocab_size)
    assert result.dtype == torch.float32


def test_extract_single_text_all_positions_empty(mock_model, mock_tokenizer):
    """Test that _extract_single_text_all_positions handles empty token_ids."""
    from wfcllm.watermark.token_channel.teacher import _extract_single_text_all_positions

    token_ids = []
    result = _extract_single_text_all_positions(mock_model, token_ids, mock_tokenizer)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (0, mock_model.vocab_size)
    assert result.dtype == torch.float32
