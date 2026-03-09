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


import json
import tempfile
from pathlib import Path
from wfcllm.watermark.generator import GenerateResult


class TestWatermarkPipelineRun:
    """Tests for WatermarkPipeline.run() — mocks generator and dataset."""

    @pytest.fixture
    def mock_result(self):
        return GenerateResult(
            code="def foo():\n    return 1\n",
            total_blocks=3,
            embedded_blocks=2,
            failed_blocks=1,
            fallback_blocks=0,
        )

    def test_run_creates_jsonl(self, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = MagicMock()
            generator.generate.return_value = mock_result

            pipeline = WatermarkPipeline(generator=generator, config=cfg)

            mock_prompts = [
                {"id": "HumanEval/0", "prompt": "def foo():\n"},
                {"id": "HumanEval/1", "prompt": "def bar():\n"},
            ]
            with patch.object(pipeline, "_load_prompts", return_value=mock_prompts):
                output_path = pipeline.run()

            # File exists
            assert Path(output_path).exists()
            assert output_path.endswith(".jsonl")

            # Parse JSONL
            lines = Path(output_path).read_text().strip().splitlines()
            assert len(lines) == 2

            record = json.loads(lines[0])
            assert record["id"] == "HumanEval/0"
            assert record["dataset"] == "humaneval"
            assert record["prompt"] == "def foo():\n"
            assert record["generated_code"] == mock_result.code
            assert record["total_blocks"] == 3
            assert record["embedded_blocks"] == 2
            assert record["failed_blocks"] == 1
            assert record["fallback_blocks"] == 0
            assert abs(record["embed_rate"] - 2/3) < 1e-6

    def test_run_returns_output_path(self, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="mbpp",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = MagicMock()
            generator.generate.return_value = mock_result
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "mbpp/1", "prompt": "Write a function"}
            ]):
                output_path = pipeline.run()
            assert "mbpp" in output_path
            assert output_path.endswith(".jsonl")

    def test_embed_rate_zero_blocks(self):
        """embed_rate is 0.0 when total_blocks is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = MagicMock()
            generator.generate.return_value = GenerateResult(
                code="", total_blocks=0, embedded_blocks=0,
                failed_blocks=0, fallback_blocks=0,
            )
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():"}
            ]):
                output_path = pipeline.run()
            record = json.loads(Path(output_path).read_text().strip())
            assert record["embed_rate"] == 0.0
