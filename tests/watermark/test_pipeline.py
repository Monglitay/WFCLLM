"""Tests for wfcllm.watermark.pipeline."""
from __future__ import annotations

from dataclasses import asdict, replace
import pytest
from wfcllm.common.block_contract import build_block_contracts
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
        assert cfg.resume is None

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
        with patch("wfcllm.common.dataset_loader.load_dataset", return_value=mock_ds):
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
        with patch("wfcllm.common.dataset_loader.load_dataset", return_value=mock_ds):
            prompts = pipeline._load_prompts()
        assert len(prompts) == 1
        assert prompts[0]["id"] == "mbpp/1"
        assert "Write a function" in prompts[0]["prompt"]


import json
import tempfile
from pathlib import Path
from wfcllm.watermark.generator import GenerateResult, EmbedStats


class TestWatermarkPipelineRun:
    """Tests for WatermarkPipeline.run() — mocks generator and dataset."""

    @pytest.fixture
    def mock_result(self):
        return GenerateResult(
            code="def foo():\n    return 1\n",
            stats=EmbedStats(
                total_blocks=3,
                embedded_blocks=2,
                failed_blocks=1,
                fallback_blocks=0,
            ),
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
                code="", stats=EmbedStats(total_blocks=0, embedded_blocks=0,
                    failed_blocks=0, fallback_blocks=0),
            )
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():"}
            ]):
                output_path = pipeline.run()
            record = json.loads(Path(output_path).read_text().strip())
            assert record["embed_rate"] == 0.0

    def test_run_serializes_adaptive_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            contracts = build_block_contracts("x = 1\n")
            generator = MagicMock()
            generator.generate.return_value = GenerateResult(
                code="x = 1\n",
                stats=EmbedStats(
                    total_blocks=1,
                    embedded_blocks=1,
                    failed_blocks=0,
                    fallback_blocks=0,
                ),
                block_contracts=contracts,
                adaptive_mode="fixed",
                profile_id=None,
                alignment_summary={
                    "final_block_count": 1,
                    "generator_total_blocks": 1,
                    "block_count_matches_total_blocks": True,
                },
            )
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():"}
            ]):
                output_path = pipeline.run()

            record = json.loads(Path(output_path).read_text().strip())
            assert record["blocks"] == [asdict(contract) for contract in contracts]
            assert record["adaptive_mode"] == "fixed"
            assert record["profile_id"] is None
            assert record["alignment_summary"] == {
                "final_block_count": 1,
                "generator_total_blocks": 1,
                "block_count_matches_total_blocks": True,
            }

    def test_run_serializes_fixed_metadata_when_adaptive_config_is_enabled_but_runtime_is_not(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            from wfcllm.watermark.config import WatermarkConfig
            from wfcllm.watermark.generator import WatermarkGenerator
            from wfcllm.watermark.interceptor import InterceptEvent
            from wfcllm.watermark.verifier import VerifyResult

            runtime_cfg = WatermarkConfig(
                secret_key="test-key",
                max_new_tokens=50,
                encoder_device="cpu",
            )
            runtime_cfg.adaptive_gamma.enabled = True

            class FakeContext:
                def __init__(self, model, tokenizer, config):
                    self.generated_text = ""
                    self.generated_ids = []
                    self.last_event = None
                    self.last_block_checkpoint = None
                    self._steps = 0

                @property
                def eos_id(self):
                    return -1

                def prefill(self, prompt):
                    return None

                def forward_and_sample(self, penalty_ids=None):
                    self._steps += 1
                    if self._steps == 1:
                        self.generated_text = "x = 1\n"
                        self.generated_ids.append(101)
                        self.last_event = InterceptEvent(
                            block_text="x = 1",
                            block_type="simple",
                            node_type="expression_statement",
                            parent_node_type="module",
                            token_start_idx=0,
                            token_count=1,
                        )
                        self.last_block_checkpoint = MagicMock(
                            generated_ids=[],
                            generated_text="",
                        )
                        return 101
                    self.last_event = None
                    return self.eos_id

                def is_finished(self):
                    return self._steps >= 2

            monkeypatch.setattr(
                "wfcllm.watermark.generator.GenerationContext",
                FakeContext,
            )

            model = MagicMock()
            tokenizer = MagicMock()
            encoder = MagicMock()
            enc_tok = MagicMock()
            tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            tokenizer.decode = MagicMock(return_value="")
            tokenizer.eos_token_id = 2

            generator = WatermarkGenerator(
                model=model,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=enc_tok,
                config=runtime_cfg,
            )
            generator._verify_block = MagicMock(
                return_value=VerifyResult(
                    passed=True,
                    min_margin=0.1,
                    lsh_signature=(1, 1, 1),
                )
            )

            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():"}
            ]):
                output_path = pipeline.run()

            record = json.loads(Path(output_path).read_text().strip())
            assert record["adaptive_mode"] == "fixed"
            assert record["profile_id"] is None

    def test_run_serializes_non_fixed_adaptive_metadata_without_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            contracts = build_block_contracts("x = 1\n")
            contracts = [
                replace(
                    contracts[0],
                    gamma_target=0.75,
                    k=3,
                    gamma_effective=0.75,
                )
            ]

            generator = MagicMock()
            generator.generate.return_value = GenerateResult(
                code="x = 1\n",
                stats=EmbedStats(
                    total_blocks=1,
                    embedded_blocks=1,
                    failed_blocks=0,
                    fallback_blocks=0,
                ),
                block_contracts=contracts,
                adaptive_mode="piecewise_quantile",
                profile_id="entropy-profile-v1",
                alignment_summary={
                    "final_block_count": 1,
                    "generator_total_blocks": 1,
                    "block_count_matches_total_blocks": True,
                },
            )
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():"}
            ]):
                output_path = pipeline.run()

            record = json.loads(Path(output_path).read_text(encoding="utf-8").strip())
            assert record["adaptive_mode"] == "piecewise_quantile"
            assert record["profile_id"] == "entropy-profile-v1"
            assert record["blocks"][0]["gamma_effective"] == 0.75
