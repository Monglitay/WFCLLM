"""Tests for wfcllm.encoder.train entry point."""

import pytest
from unittest.mock import patch, MagicMock

from wfcllm.encoder.train import load_code_samples, prepare_blocks_with_variants


class TestLoadCodeSamples:
    @patch("wfcllm.encoder.train.load_dataset")
    def test_loads_mbpp(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"code": "x = 1", "task_id": 1, "text": "test"},
        ]))
        mock_load.return_value = {"train": mock_ds}
        samples = load_code_samples(["mbpp"])
        assert len(samples) > 0
        assert "code" in samples[0]


class TestPrepareBlocksWithVariants:
    def test_basic(self):
        code_samples = [{"code": "x = 1\ny = 2"}]
        blocks = prepare_blocks_with_variants(code_samples, max_variants=5)
        assert len(blocks) > 0
        for b in blocks:
            assert "source" in b
            assert "positive_variants" in b
            assert "negative_variants" in b
