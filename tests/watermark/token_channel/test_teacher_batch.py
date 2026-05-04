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


# ============================================================================
# Task 2: Inter-Text Parallel Inference (Batch Multiple Texts)
# ============================================================================


def test_batch_forward_all_positions_returns_list(mock_model, mock_tokenizer):
    """Test that _batch_forward_all_positions returns a list of tensors."""
    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    batch_token_ids = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]

    result = _batch_forward_all_positions(mock_model, batch_token_ids, mock_tokenizer)

    assert isinstance(result, list)
    assert len(result) == len(batch_token_ids)


def test_batch_forward_all_positions_correct_shapes(mock_model, mock_tokenizer):
    """Test that each output tensor has correct shape [seq_len, vocab_size]."""
    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    batch_token_ids = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]

    result = _batch_forward_all_positions(mock_model, batch_token_ids, mock_tokenizer)

    for i, token_ids in enumerate(batch_token_ids):
        assert isinstance(result[i], torch.Tensor)
        assert result[i].shape == (len(token_ids), mock_model.vocab_size)
        assert result[i].dtype == torch.float32


def test_batch_forward_all_positions_empty_batch(mock_model, mock_tokenizer):
    """Test that empty batch returns empty list."""
    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    batch_token_ids = []
    result = _batch_forward_all_positions(mock_model, batch_token_ids, mock_tokenizer)

    assert isinstance(result, list)
    assert len(result) == 0


def test_batch_forward_all_positions_single_empty_text(mock_model, mock_tokenizer):
    """Test that batch with single empty text returns correct shape."""
    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    batch_token_ids = [[]]
    result = _batch_forward_all_positions(mock_model, batch_token_ids, mock_tokenizer)

    assert len(result) == 1
    assert result[0].shape == (0, mock_model.vocab_size)


def test_batch_forward_all_positions_matches_single_inference(mock_tokenizer):
    """Test that batch inference produces same results as single inference."""
    from wfcllm.watermark.token_channel.teacher import (
        _batch_forward_all_positions,
        _extract_single_text_all_positions,
    )

    # Use a deterministic model that respects attention_mask
    class DeterministicModel(torch.nn.Module):
        def __init__(self, vocab_size: int = 100):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding = torch.nn.Embedding(vocab_size, 16)
            self.linear = torch.nn.Linear(16, vocab_size)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> dict[str, torch.Tensor]:
            # Deterministic forward pass based on input_ids
            # Use position-wise computation to avoid padding effects
            embeddings = self.embedding(input_ids)
            # Simple position-wise transformation
            logits = self.linear(embeddings)
            return {"logits": logits}

    torch.manual_seed(42)
    model = DeterministicModel(vocab_size=100)
    model.eval()

    batch_token_ids = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]

    # Get batch results
    batch_results = _batch_forward_all_positions(model, batch_token_ids, mock_tokenizer)

    # Get single results
    single_results = [
        _extract_single_text_all_positions(model, token_ids, mock_tokenizer)
        for token_ids in batch_token_ids
    ]

    # Compare results
    for i, (batch_result, single_result) in enumerate(zip(batch_results, single_results)):
        assert batch_result.shape == single_result.shape, f"Shape mismatch for text {i}"
        # Allow small numerical differences due to padding
        assert torch.allclose(batch_result, single_result, atol=1e-5), f"Result mismatch for text {i}"


def test_batch_forward_all_positions_uses_left_padding(mock_tokenizer):
    """Test that left padding is used (preserves causal structure)."""
    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    # Create a model that tracks input_ids
    class TrackingModel(torch.nn.Module):
        def __init__(self, vocab_size: int = 100):
            super().__init__()
            self.vocab_size = vocab_size
            self.received_input_ids = None
            self.received_attention_mask = None

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> dict[str, torch.Tensor]:
            self.received_input_ids = input_ids
            self.received_attention_mask = attention_mask
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
            return {"logits": logits}

    model = TrackingModel(vocab_size=100)

    batch_token_ids = [
        [1, 2, 3, 4, 5],  # Length 5
        [6, 7],           # Length 2
    ]

    result = _batch_forward_all_positions(model, batch_token_ids, mock_tokenizer)

    # Verify left padding was used
    # With BOS prepended: [0, 1, 2, 3, 4, 5] (length 6) and [0, 6, 7] (length 3)
    # After left padding to length 6: [0, 1, 2, 3, 4, 5] and [pad, pad, pad, 0, 6, 7]
    assert model.received_input_ids is not None
    assert model.received_attention_mask is not None

    # Check that shorter sequence has padding on the left
    # The second sequence should have padding tokens at the beginning
    input_ids = model.received_input_ids
    attention_mask = model.received_attention_mask

    # Second sequence should have 0s in attention mask at the beginning
    assert attention_mask[1, 0] == 0  # First position is padding
    assert attention_mask[1, -1] == 1  # Last position is real token


def test_batch_forward_all_positions_eval_mode(mock_tokenizer):
    """Test that model.eval() is called during batch extraction."""
    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    model = MockModel(vocab_size=100)
    model.train()
    assert model.training is True

    batch_token_ids = [[1, 2, 3], [4, 5]]

    # Track eval calls
    eval_called = []
    original_eval = model.eval

    def tracking_eval():
        eval_called.append(True)
        return original_eval()

    model.eval = tracking_eval

    result = _batch_forward_all_positions(model, batch_token_ids, mock_tokenizer)

    # Verify eval was called
    assert len(eval_called) == 1
    # Verify training mode was restored
    assert model.training is True


def test_batch_forward_all_positions_no_grad(mock_tokenizer):
    """Test that gradient tracking is disabled during batch extraction."""
    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    model = MockModel(vocab_size=100)
    batch_token_ids = [[1, 2, 3], [4, 5]]

    # Track whether forward pass happens under no_grad
    grad_enabled_during_forward = []
    original_forward = model.forward

    def tracking_forward(input_ids, attention_mask=None):
        grad_enabled_during_forward.append(torch.is_grad_enabled())
        return original_forward(input_ids)

    model.forward = tracking_forward

    result = _batch_forward_all_positions(model, batch_token_ids, mock_tokenizer)

    # Verify forward pass happened with gradients disabled
    assert len(grad_enabled_during_forward) == 1
    assert grad_enabled_during_forward[0] is False


# ============================================================================
# Task 3: Dynamic Batch Sizing and Length Grouping
# ============================================================================


def test_group_texts_by_length_returns_groups(mock_tokenizer):
    """Test that _group_texts_by_length returns list of groups."""
    from wfcllm.watermark.token_channel.teacher import _group_texts_by_length

    texts = ["short", "medium text", "another short", "very long text here"]

    groups = _group_texts_by_length(texts, mock_tokenizer, tolerance=0.2)

    assert isinstance(groups, list)
    assert len(groups) > 0
    assert all(isinstance(group, list) for group in groups)


def test_group_texts_by_length_groups_similar_lengths(mock_tokenizer):
    """Test that texts with similar lengths are grouped together."""
    from wfcllm.watermark.token_channel.teacher import _group_texts_by_length

    # Create a tokenizer that returns different lengths
    class LengthTokenizer:
        def __init__(self):
            self.bos_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Return length proportional to text length
            return list(range(len(text)))

    tokenizer = LengthTokenizer()

    # Texts with clearly different lengths
    texts = ["a", "bb", "ccc", "d", "ee", "ffff"]  # lengths: 1, 2, 3, 1, 2, 4

    groups = _group_texts_by_length(texts, tokenizer, tolerance=0.3)

    # Should have multiple groups
    assert len(groups) >= 2

    # Each group should contain (index, text) tuples
    for group in groups:
        assert all(isinstance(item, tuple) and len(item) == 2 for item in group)
        assert all(isinstance(item[0], int) and isinstance(item[1], str) for item in group)

    # Verify texts with length 1 are grouped together
    length_1_indices = {idx for group in groups for idx, text in group if len(text) == 1}
    assert 0 in length_1_indices and 3 in length_1_indices


def test_group_texts_by_length_respects_tolerance(mock_tokenizer):
    """Test that tolerance parameter controls grouping."""
    from wfcllm.watermark.token_channel.teacher import _group_texts_by_length

    class LengthTokenizer:
        def __init__(self):
            self.bos_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(range(len(text)))

    tokenizer = LengthTokenizer()
    texts = ["a" * 10, "b" * 11, "c" * 20]  # lengths: 10, 11, 20

    # With high tolerance, should group 10 and 11 together
    groups_high = _group_texts_by_length(texts, tokenizer, tolerance=0.5)

    # With low tolerance, 10 and 11 might be separate
    groups_low = _group_texts_by_length(texts, tokenizer, tolerance=0.05)

    # High tolerance should produce fewer or equal groups
    assert len(groups_high) <= len(groups_low)


def test_group_texts_by_length_empty_input(mock_tokenizer):
    """Test that empty input returns empty list."""
    from wfcllm.watermark.token_channel.teacher import _group_texts_by_length

    texts = []
    groups = _group_texts_by_length(texts, mock_tokenizer, tolerance=0.2)

    assert groups == []


def test_group_texts_by_length_single_text(mock_tokenizer):
    """Test that single text returns one group."""
    from wfcllm.watermark.token_channel.teacher import _group_texts_by_length

    texts = ["single text"]
    groups = _group_texts_by_length(texts, mock_tokenizer, tolerance=0.2)

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert groups[0][0] == (0, "single text")


def test_compute_dynamic_batch_size_scales_with_length():
    """Test that batch size decreases as sequence length increases."""
    from wfcllm.watermark.token_channel.teacher import _compute_dynamic_batch_size

    available_memory = 20.0  # 20 GB
    base_batch_size = 32

    # Short sequences should allow larger batch
    batch_short = _compute_dynamic_batch_size(
        max_length=50,
        base_batch_size=base_batch_size,
        available_memory_gb=available_memory
    )

    # Long sequences should use smaller batch
    batch_long = _compute_dynamic_batch_size(
        max_length=500,
        base_batch_size=base_batch_size,
        available_memory_gb=available_memory
    )

    assert batch_short > batch_long
    assert isinstance(batch_short, int)
    assert isinstance(batch_long, int)


def test_compute_dynamic_batch_size_respects_min_constraint():
    """Test that batch size never goes below minimum."""
    from wfcllm.watermark.token_channel.teacher import _compute_dynamic_batch_size

    # Very long sequence with limited memory
    batch_size = _compute_dynamic_batch_size(
        max_length=10000,
        base_batch_size=32,
        available_memory_gb=1.0
    )

    # Should be at least 4
    assert batch_size >= 4


def test_compute_dynamic_batch_size_respects_max_constraint():
    """Test that batch size never exceeds maximum."""
    from wfcllm.watermark.token_channel.teacher import _compute_dynamic_batch_size

    base_batch_size = 32

    # Very short sequence with lots of memory
    batch_size = _compute_dynamic_batch_size(
        max_length=10,
        base_batch_size=base_batch_size,
        available_memory_gb=100.0
    )

    # Should not exceed base_batch_size * 2
    assert batch_size <= base_batch_size * 2


def test_get_available_memory_gb_returns_float(mock_model):
    """Test that _get_available_memory_gb returns a float."""
    from wfcllm.watermark.token_channel.teacher import _get_available_memory_gb

    memory = _get_available_memory_gb(mock_model)

    assert isinstance(memory, float)
    assert memory > 0


def test_get_available_memory_gb_fallback():
    """Test that _get_available_memory_gb falls back to default when CUDA unavailable."""
    from wfcllm.watermark.token_channel.teacher import _get_available_memory_gb

    # Mock model without CUDA
    class CPUModel:
        pass

    model = CPUModel()
    memory = _get_available_memory_gb(model)

    # Should return fallback value
    assert memory == 20.0
