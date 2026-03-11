"""Tests for WatermarkDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from wfcllm.extract.config import DetectionResult, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector


class TestWatermarkDetector:
    @pytest.fixture
    def config(self):
        return ExtractConfig(secret_key="test-key", embed_dim=128, fpr_threshold=3.0)

    @pytest.fixture
    def mock_encoder(self):
        encoder = MagicMock()
        vec = torch.randn(1, 128)
        vec = vec / vec.norm()
        encoder.return_value = vec
        encoder.eval = MagicMock(return_value=encoder)
        return encoder

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        return tokenizer

    def test_detect_returns_detection_result(
        self, config, mock_encoder, mock_tokenizer
    ):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        # Use compound+simple input to verify compound block is excluded
        code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
        result = detector.detect(code)
        assert isinstance(result, DetectionResult)
        # Only simple blocks counted (x = i + 1, y = i * 2), not the for compound block
        assert result.total_blocks == 2
        assert result.independent_blocks == result.total_blocks  # all simple blocks selected
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value, float)

    def test_detect_empty_code(self, config, mock_encoder, mock_tokenizer):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        result = detector.detect("")

        assert result.is_watermarked is False
        assert result.total_blocks == 0
        assert result.independent_blocks == 0

    def test_block_details_include_simple_blocks(
        self, config, mock_encoder, mock_tokenizer
    ):
        """block_details should contain only simple blocks, not compound blocks."""
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
        result = detector.detect(code)
        assert len(result.block_details) == result.total_blocks
        # total_blocks should be 2 (only the simple blocks), not 3
        assert result.total_blocks == 2

    def test_selected_flag_set(self, config, mock_encoder, mock_tokenizer):
        """All simple blocks should have selected=True."""
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
        result = detector.detect(code)
        # All simple blocks are selected (no DP filtering)
        assert all(s.selected for s in result.block_details)
        assert result.independent_blocks == result.total_blocks
