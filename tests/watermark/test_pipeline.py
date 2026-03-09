"""Tests for wfcllm.watermark.pipeline."""
from __future__ import annotations

import pytest
from wfcllm.watermark.pipeline import WatermarkPipelineConfig


class TestWatermarkPipelineConfig:
    def test_default_fields(self):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir="data/watermarked",
            dataset_path="data/datasets",
        )
        assert cfg.dataset == "humaneval"
        assert cfg.output_dir == "data/watermarked"
        assert cfg.dataset_path == "data/datasets"

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must be"):
            WatermarkPipelineConfig(
                dataset="unknown",
                output_dir="data/watermarked",
                dataset_path="data/datasets",
            )


from unittest.mock import patch, MagicMock
from wfcllm.watermark.pipeline import WatermarkPipeline


class TestWatermarkPipelineLoadPrompts:
    """Tests for _load_prompts() — uses mocked datasets library."""

    @pytest.fixture
    def pipeline(self):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir="data/watermarked",
            dataset_path="data/datasets",
        )
        generator = MagicMock()
        return WatermarkPipeline(generator=generator, config=cfg)

    def test_load_humaneval_returns_list_of_dicts(self, pipeline):
        mock_ds = {
            "test": [
                {"task_id": "HumanEval/0", "prompt": "def foo():\n    pass\n"},
            ]
        }
        with patch("wfcllm.watermark.pipeline.load_dataset", return_value=mock_ds):
            prompts = pipeline._load_prompts()
        assert len(prompts) == 1
        assert prompts[0]["id"] == "HumanEval/0"
        assert "def foo():" in prompts[0]["prompt"]

    def test_load_mbpp_returns_list_of_dicts(self):
        cfg = WatermarkPipelineConfig(
            dataset="mbpp",
            output_dir="data/watermarked",
            dataset_path="data/datasets",
        )
        pipeline = WatermarkPipeline(generator=MagicMock(), config=cfg)
        mock_ds = {
            "train": [
                {"task_id": 1, "text": "Write a function", "code": "def f(): pass"},
            ]
        }
        with patch("wfcllm.watermark.pipeline.load_dataset", return_value=mock_ds):
            prompts = pipeline._load_prompts()
        assert len(prompts) == 1
        assert prompts[0]["id"] == "mbpp/1"
        assert "Write a function" in prompts[0]["prompt"]
