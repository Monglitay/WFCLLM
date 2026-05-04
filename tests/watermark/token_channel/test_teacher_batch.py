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


def test_extract_single_text_all_positions_prepends_bos(mock_model):
    """Test that BOS token is prepended when tokenizer has one."""
    from wfcllm.watermark.token_channel.teacher import _extract_single_text_all_positions

    tokenizer = MockTokenizer(bos_token_id=0)
    token_ids = [1, 2, 3]

    # Track what input_ids the model receives
    received_input_ids = []
    original_forward = mock_model.forward

    def tracking_forward(input_ids):
        received_input_ids.append(input_ids.tolist())
        return original_forward(input_ids)

    mock_model.forward = tracking_forward

    result = _extract_single_text_all_positions(mock_model, token_ids, tokenizer)

    # Verify BOS token was prepended
    assert len(received_input_ids) == 1
    assert received_input_ids[0][0][0] == 0  # BOS token at start
    assert received_input_ids[0][0][1:] == token_ids  # Original tokens follow


def test_extract_single_text_all_positions_no_bos_raises():
    """Test that missing BOS token raises error for non-empty token_ids."""
    from wfcllm.watermark.token_channel.teacher import _extract_single_text_all_positions

    tokenizer = MockTokenizer(bos_token_id=None)
    model = MockModel(vocab_size=100)
    token_ids = [1, 2, 3]

    with pytest.raises(ValueError, match="require tokenizer.bos_token_id"):
        _extract_single_text_all_positions(model, token_ids, tokenizer)


def test_extract_single_text_all_positions_eval_mode(mock_model, mock_tokenizer):
    """Test that model.eval() is called during extraction."""
    from wfcllm.watermark.token_channel.teacher import _extract_single_text_all_positions

    # Set model to training mode
    mock_model.train()
    assert mock_model.training is True

    token_ids = [1, 2, 3]

    # Track eval calls
    eval_called = []
    original_eval = mock_model.eval

    def tracking_eval():
        eval_called.append(True)
        return original_eval()

    mock_model.eval = tracking_eval

    result = _extract_single_text_all_positions(mock_model, token_ids, mock_tokenizer)

    # Verify eval was called
    assert len(eval_called) == 1
    # Verify training mode was restored
    assert mock_model.training is True


def test_extract_single_text_all_positions_no_grad(mock_model, mock_tokenizer):
    """Test that gradient tracking is disabled during extraction."""
    from wfcllm.watermark.token_channel.teacher import _extract_single_text_all_positions

    token_ids = [1, 2, 3]

    # Track whether forward pass happens under no_grad
    grad_enabled_during_forward = []
    original_forward = mock_model.forward

    def tracking_forward(input_ids):
        grad_enabled_during_forward.append(torch.is_grad_enabled())
        return original_forward(input_ids)

    mock_model.forward = tracking_forward

    result = _extract_single_text_all_positions(mock_model, token_ids, mock_tokenizer)

    # Verify forward pass happened with gradients disabled
    assert len(grad_enabled_during_forward) == 1
    assert grad_enabled_during_forward[0] is False


def test_get_vocab_size():
    """Test _get_vocab_size helper function."""
    from wfcllm.watermark.token_channel.teacher import _get_vocab_size

    # Test with config.vocab_size
    class ModelWithConfig:
        class Config:
            vocab_size = 50000
        config = Config()

    assert _get_vocab_size(ModelWithConfig()) == 50000

    # Test with direct vocab_size attribute
    class ModelWithVocabSize:
        vocab_size = 32000

    assert _get_vocab_size(ModelWithVocabSize()) == 32000

    # Test with both (config takes precedence)
    class ModelWithBoth:
        class Config:
            vocab_size = 50000
        config = Config()
        vocab_size = 32000

    assert _get_vocab_size(ModelWithBoth()) == 50000

    # Test with neither
    class ModelWithNeither:
        pass

    with pytest.raises(ValueError, match="must expose vocab_size"):
        _get_vocab_size(ModelWithNeither())
