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

    def test_sample_limit_default_none(self):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir="data/watermarked",
            dataset_path="data/datasets",
        )
        assert cfg.sample_limit is None


from unittest.mock import patch, MagicMock
from wfcllm.watermark.diagnostics import (
    BlockLifecycleRecord,
    summarize_sample_diagnostics,
)
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
from types import SimpleNamespace
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

    @staticmethod
    def _build_generator(mock_result):
        generator = MagicMock()
        generator.generate.return_value = mock_result
        generator.config = SimpleNamespace(
            lsh_d=4,
            lsh_gamma=0.75,
            margin_base=0.1,
            margin_alpha=0.05,
            secret_key="test-secret",
        )
        return generator

    def test_run_creates_jsonl(self, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = self._build_generator(mock_result)

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

    def test_pipeline_writes_watermark_params(self, tmp_path, mock_result):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(tmp_path),
            dataset_path="data/datasets",
        )
        generator = self._build_generator(mock_result)
        pipeline = WatermarkPipeline(generator=generator, config=cfg)

        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"},
        ]):
            output_path = pipeline.run()

        row = json.loads(Path(output_path).read_text(encoding="utf-8").splitlines()[0])
        assert row["watermark_params"] == {
            "lsh_d": 4,
            "lsh_gamma": 0.75,
            "margin_base": 0.1,
            "margin_alpha": 0.05,
        }

    def test_pipeline_writes_public_adaptive_gamma_metadata(self, tmp_path):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(tmp_path),
            dataset_path="data/datasets",
        )
        contracts = build_block_contracts("x = 1\n")
        generator = self._build_generator(
            GenerateResult(
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
        )
        generator.config.adaptive_gamma = SimpleNamespace(
            enabled=True,
            strategy="piecewise_quantile",
            profile_id="entropy-profile-v1",
            anchors={
                "p10": 0.95,
                "p50": 0.75,
                "p75": 0.55,
                "p90": 0.35,
                "p95": 0.25,
            },
        )
        generator._entropy_profile = SimpleNamespace(
            language="python",
            model_family="demo-model",
            quantiles_units_map={
                "p10": 1,
                "p50": 2,
                "p75": 3,
                "p90": 4,
                "p95": 5,
            },
            strategy="piecewise_quantile",
        )
        pipeline = WatermarkPipeline(generator=generator, config=cfg)

        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"},
        ]):
            output_path = pipeline.run()

        row = json.loads(Path(output_path).read_text(encoding="utf-8").splitlines()[0])
        assert row["watermark_params"]["adaptive_gamma"] == {
            "strategy": "piecewise_quantile",
            "profile_id": "entropy-profile-v1",
            "anchors": {
                "p10": 0.95,
                "p50": 0.75,
                "p75": 0.55,
                "p90": 0.35,
                "p95": 0.25,
            },
            "profile": {
                "language": "python",
                "model_family": "demo-model",
                "quantiles_units": {
                    "p10": 1,
                    "p50": 2,
                    "p75": 3,
                    "p90": 4,
                    "p95": 5,
                },
                "strategy": "piecewise_quantile",
            },
        }
        assert "profile_path" not in row["watermark_params"]["adaptive_gamma"]

    def test_run_persists_diagnostic_summary_and_block_ledger(self, tmp_path):
        watermarked_dir = tmp_path / "watermarked"
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(watermarked_dir),
            dataset_path="data/datasets",
        )
        diagnostic_summary = {
            "diagnostics_version": 1,
            "retry_summary": {
                "blocks_with_retry": 1,
                "attempts_total": 1,
                "attempts_no_block": 0,
                "retry_rescued_blocks": 1,
                "retry_exhausted_blocks": 0,
            },
            "cascade_summary": {
                "cascade_triggers": 0,
                "cascade_rollbacks": 0,
                "cascade_rescued_blocks": 0,
            },
            "failure_reason_counts": {
                "signature_miss": 1,
            },
            "rescued_blocks": 1,
            "unrescued_blocks": 0,
        }
        block_ledgers = [
            {
                "sample_id": "HumanEval/0",
                "block_ordinal": 0,
                "initial_verify": {"passed": False, "failure_reason": "signature_miss"},
                "retry_attempts": [{"attempt_index": 1, "produced_block": True}],
                "cascade_events": [],
                "final_outcome": {"embedded": True, "rescued_by_retry": True},
            },
            {
                "sample_id": "HumanEval/0",
                "block_ordinal": 1,
                "initial_verify": {"passed": True},
                "retry_attempts": [],
                "cascade_events": [],
                "final_outcome": {"embedded": True},
            },
        ]
        generator = self._build_generator(GenerateResult(
            code="def foo():\n    return 1\n",
            stats=EmbedStats(
                total_blocks=2,
                embedded_blocks=2,
                failed_blocks=0,
                fallback_blocks=0,
            ),
            diagnostic_summary=diagnostic_summary,
            block_ledgers=block_ledgers,
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"}
        ]):
            output_path = Path(pipeline.run())

        row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["diagnostics_version"] == 1
        assert row["retry_summary"] == diagnostic_summary["retry_summary"]
        assert row["cascade_summary"] == diagnostic_summary["cascade_summary"]

        diagnostics_path = (
            output_path.parent.parent
            / "diagnostics"
            / f"{output_path.stem}_block_ledger.jsonl"
        )
        assert diagnostics_path.exists()
        ledger_rows = [
            json.loads(line)
            for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        ]
        assert ledger_rows == block_ledgers

    def test_run_keeps_retry_rescue_rollups_in_sync_with_block_ledger(self, tmp_path):
        watermarked_dir = tmp_path / "watermarked"
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(watermarked_dir),
            dataset_path="data/datasets",
        )
        diagnostic_summary = {
            "diagnostics_version": 1,
            "retry_summary": {
                "blocks_with_retry": 2,
                "attempts_total": 2,
                "attempts_no_block": 1,
                "retry_rescued_blocks": 1,
                "retry_exhausted_blocks": 1,
            },
            "cascade_summary": {
                "cascade_triggers": 0,
                "cascade_rollbacks": 0,
                "cascade_rescued_blocks": 0,
            },
            "failure_reason_counts": {"signature_miss": 2},
            "rescued_blocks": 1,
            "unrescued_blocks": 1,
        }
        block_ledgers = [
            {
                "sample_id": "HumanEval/0",
                "block_ordinal": 0,
                "initial_verify": {"passed": False, "failure_reason": "signature_miss"},
                "retry_attempts": [{"attempt_index": 1, "produced_block": True}],
                "cascade_events": [],
                "final_outcome": {"embedded": True, "rescued_by_retry": True},
            },
            {
                "sample_id": "HumanEval/0",
                "block_ordinal": 1,
                "initial_verify": {"passed": False, "failure_reason": "signature_miss"},
                "retry_attempts": [{"attempt_index": 1, "produced_block": False}],
                "cascade_events": [],
                "final_outcome": {
                    "embedded": False,
                    "exhausted_retries": True,
                    "failure_reason": "signature_miss",
                },
            },
            {
                "sample_id": "HumanEval/0",
                "block_ordinal": 2,
                "initial_verify": {"passed": True},
                "retry_attempts": [],
                "cascade_events": [],
                "final_outcome": {"embedded": True},
            },
        ]
        generator = self._build_generator(GenerateResult(
            code="def foo():\n    return 1\n",
            stats=EmbedStats(
                total_blocks=3,
                embedded_blocks=2,
                failed_blocks=1,
                fallback_blocks=0,
            ),
            diagnostic_summary=diagnostic_summary,
            block_ledgers=block_ledgers,
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"}
        ]):
            output_path = Path(pipeline.run())

        row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        diagnostics_path = (
            output_path.parent.parent
            / "diagnostics"
            / f"{output_path.stem}_block_ledger.jsonl"
        )
        ledger_rows = [
            json.loads(line)
            for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        ]
        expected_summary = summarize_sample_diagnostics(
            BlockLifecycleRecord(
                sample_id=ledger["sample_id"],
                block_ordinal=ledger["block_ordinal"],
                initial_verify=ledger.get("initial_verify", {}),
                retry_attempts=ledger.get("retry_attempts", []),
                cascade_events=ledger.get("cascade_events", []),
                final_outcome=ledger.get("final_outcome", {}),
            )
            for ledger in ledger_rows
        )

        assert any(
            ledger.get("final_outcome", {}).get("exhausted_retries") is True
            for ledger in ledger_rows
        )
        assert expected_summary["retry_summary"]["retry_rescued_blocks"] > 0
        assert expected_summary["retry_summary"]["retry_exhausted_blocks"] > 0
        assert (
            row["retry_summary"]["retry_rescued_blocks"]
            == expected_summary["retry_summary"]["retry_rescued_blocks"]
        )
        assert (
            row["retry_summary"]["retry_exhausted_blocks"]
            == expected_summary["retry_summary"]["retry_exhausted_blocks"]
        )
        assert row["rescued_blocks"] == expected_summary["rescued_blocks"]
        assert row["unrescued_blocks"] == expected_summary["unrescued_blocks"]

    def test_run_persists_cascade_visibility_without_fallback_blocks(self, tmp_path):
        watermarked_dir = tmp_path / "watermarked"
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(watermarked_dir),
            dataset_path="data/datasets",
        )
        generator = self._build_generator(GenerateResult(
            code="if x:\n    return 1\n",
            stats=EmbedStats(
                total_blocks=1,
                embedded_blocks=1,
                failed_blocks=0,
                fallback_blocks=0,
            ),
            diagnostic_summary={
                "diagnostics_version": 1,
                "retry_summary": {
                    "blocks_with_retry": 0,
                    "attempts_total": 0,
                    "attempts_no_block": 0,
                    "retry_rescued_blocks": 0,
                    "retry_exhausted_blocks": 0,
                },
                "cascade_summary": {
                    "cascade_triggers": 1,
                    "cascade_rollbacks": 1,
                    "cascade_rescued_blocks": 0,
                },
                "failure_reason_counts": {},
                "rescued_blocks": 0,
                "unrescued_blocks": 0,
            },
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"}
        ]):
            output_path = pipeline.run()

        row = json.loads(Path(output_path).read_text(encoding="utf-8").splitlines()[0])
        assert row["fallback_blocks"] == 0
        assert row["cascade_summary"]["cascade_triggers"] == 1

    def test_run_writes_ledger_beside_explicit_resume_output_path(self, tmp_path):
        configured_output_dir = tmp_path / "configured" / "watermarked"
        actual_resume_dir = tmp_path / "actual" / "watermarked"
        actual_resume_dir.mkdir(parents=True)
        resume_path = actual_resume_dir / "humaneval_20260101_010101.jsonl"
        resume_path.write_text(
            json.dumps({"id": "HumanEval/0", "total_blocks": 1}) + "\n",
            encoding="utf-8",
        )
        expected_ledger_path = (
            resume_path.parent.parent
            / "diagnostics"
            / f"{resume_path.stem}_block_ledger.jsonl"
        )
        expected_ledger_path.parent.mkdir(parents=True)
        existing_ledger_row = {
            "sample_id": "HumanEval/0",
            "block_ordinal": 0,
            "initial_verify": {"passed": True},
            "retry_attempts": [],
            "cascade_events": [],
            "final_outcome": {"embedded": True},
        }
        expected_ledger_path.write_text(
            json.dumps(existing_ledger_row) + "\n",
            encoding="utf-8",
        )
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(configured_output_dir),
            dataset_path="data/datasets",
            resume=str(resume_path),
        )
        block_ledgers = [
            {
                "sample_id": "HumanEval/1",
                "block_ordinal": 0,
                "initial_verify": {"passed": True},
                "retry_attempts": [],
                "cascade_events": [],
                "final_outcome": {"embedded": True},
            }
        ]
        generator = self._build_generator(GenerateResult(
            code="def bar():\n    return 2\n",
            stats=EmbedStats(
                total_blocks=1,
                embedded_blocks=1,
                failed_blocks=0,
                fallback_blocks=0,
            ),
            diagnostic_summary={
                "diagnostics_version": 1,
                "retry_summary": {},
                "cascade_summary": {},
            },
            block_ledgers=block_ledgers,
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"},
            {"id": "HumanEval/1", "prompt": "def bar():\n"},
        ]):
            output_path = Path(pipeline.run())

        assert output_path == resume_path
        assert expected_ledger_path.exists()
        ledger_rows = [
            json.loads(line)
            for line in expected_ledger_path.read_text(encoding="utf-8").splitlines()
        ]
        assert ledger_rows == [existing_ledger_row, *block_ledgers]

        wrong_ledger_path = (
            configured_output_dir.parent
            / "diagnostics"
            / f"{output_path.stem}_block_ledger.jsonl"
        )
        assert not wrong_ledger_path.exists()

    def test_run_legacy_resume_without_route_one_fields_skips_sidecar_validation(
        self,
        tmp_path,
    ):
        configured_output_dir = tmp_path / "configured" / "watermarked"
        actual_resume_dir = tmp_path / "actual" / "watermarked"
        actual_resume_dir.mkdir(parents=True)
        resume_path = actual_resume_dir / "humaneval_20260101_010101.jsonl"
        resume_path.write_text(
            json.dumps(
                {
                    "id": "HumanEval/0",
                    "total_blocks": 1,
                    "embedded_blocks": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(configured_output_dir),
            dataset_path="data/datasets",
            resume=str(resume_path),
        )
        block_ledgers = [
            {
                "sample_id": "HumanEval/1",
                "block_ordinal": 0,
                "initial_verify": {"passed": True},
                "retry_attempts": [],
                "cascade_events": [],
                "final_outcome": {"embedded": True},
            }
        ]
        generator = self._build_generator(GenerateResult(
            code="def bar():\n    return 2\n",
            stats=EmbedStats(
                total_blocks=1,
                embedded_blocks=1,
                failed_blocks=0,
                fallback_blocks=0,
            ),
            diagnostic_summary={
                "diagnostics_version": 1,
                "retry_summary": {},
                "cascade_summary": {},
            },
            block_ledgers=block_ledgers,
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"},
            {"id": "HumanEval/1", "prompt": "def bar():\n"},
        ]):
            output_path = Path(pipeline.run())

        assert output_path == resume_path
        diagnostics_path = (
            resume_path.parent.parent
            / "diagnostics"
            / f"{resume_path.stem}_block_ledger.jsonl"
        )
        assert diagnostics_path.exists()
        ledger_rows = [
            json.loads(line)
            for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        ]
        assert ledger_rows == block_ledgers

    def test_run_resume_requires_aligned_diagnostics_sidecar(self, tmp_path):
        configured_output_dir = tmp_path / "configured" / "watermarked"
        actual_resume_dir = tmp_path / "actual" / "watermarked"
        actual_resume_dir.mkdir(parents=True)
        resume_path = actual_resume_dir / "humaneval_20260101_010101.jsonl"
        resume_path.write_text(
            json.dumps(
                {
                    "id": "HumanEval/0",
                    "total_blocks": 1,
                    "embedded_blocks": 1,
                    "diagnostics_version": 1,
                    "retry_summary": {},
                    "cascade_summary": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(configured_output_dir),
            dataset_path="data/datasets",
            resume=str(resume_path),
        )
        generator = self._build_generator(GenerateResult(
            code="def bar():\n    return 2\n",
            stats=EmbedStats(
                total_blocks=1,
                embedded_blocks=1,
                failed_blocks=0,
                fallback_blocks=0,
            ),
            diagnostic_summary={
                "diagnostics_version": 1,
                "retry_summary": {},
                "cascade_summary": {},
            },
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"},
            {"id": "HumanEval/1", "prompt": "def bar():\n"},
        ]):
            with pytest.raises(ValueError, match="diagnostics"):
                pipeline.run()

    def test_run_resume_detects_truncated_diagnostics_sidecar(self, tmp_path):
        configured_output_dir = tmp_path / "configured" / "watermarked"
        actual_resume_dir = tmp_path / "actual" / "watermarked"
        actual_resume_dir.mkdir(parents=True)
        resume_path = actual_resume_dir / "humaneval_20260101_010101.jsonl"
        resume_path.write_text(
            json.dumps(
                {
                    "id": "HumanEval/0",
                    "total_blocks": 2,
                    "embedded_blocks": 1,
                    "diagnostics_version": 1,
                    "retry_summary": {},
                    "cascade_summary": {},
                    "alignment_summary": {"final_block_count": 2},
                    "blocks": [{"ordinal": 0}, {"ordinal": 1}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        diagnostics_dir = (resume_path.parent.parent / "diagnostics")
        diagnostics_dir.mkdir(parents=True)
        diagnostics_path = diagnostics_dir / f"{resume_path.stem}_block_ledger.jsonl"
        diagnostics_path.write_text(
            json.dumps(
                {
                    "sample_id": "HumanEval/0",
                    "block_ordinal": 0,
                    "initial_verify": {"passed": True},
                    "retry_attempts": [],
                    "cascade_events": [],
                    "final_outcome": {"embedded": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(configured_output_dir),
            dataset_path="data/datasets",
            resume=str(resume_path),
        )
        generator = self._build_generator(GenerateResult(
            code="def bar():\n    return 2\n",
            stats=EmbedStats(
                total_blocks=1,
                embedded_blocks=1,
                failed_blocks=0,
                fallback_blocks=0,
            ),
            diagnostic_summary={
                "diagnostics_version": 1,
                "retry_summary": {},
                "cascade_summary": {},
            },
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"},
            {"id": "HumanEval/1", "prompt": "def bar():\n"},
        ]):
            with pytest.raises(ValueError, match="diagnostics"):
                pipeline.run()

    def test_run_persists_only_allowlisted_diagnostic_summary_fields(self, tmp_path):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(tmp_path / "watermarked"),
            dataset_path="data/datasets",
        )
        generator = self._build_generator(GenerateResult(
            code="def foo():\n    return 1\n",
            stats=EmbedStats(
                total_blocks=1,
                embedded_blocks=1,
                failed_blocks=0,
                fallback_blocks=0,
            ),
            diagnostic_summary={
                "diagnostics_version": 1,
                "retry_summary": {"attempts_total": 0},
                "cascade_summary": {"cascade_triggers": 0},
                "failure_reason_counts": {"unknown": 0},
                "rescued_blocks": 0,
                "unrescued_blocks": 0,
                "debug_blob": {"raw": [1, 2, 3]},
                "trace_id": "abc123",
            },
        ))
        pipeline = WatermarkPipeline(generator=generator, config=cfg)
        with patch.object(pipeline, "_load_prompts", return_value=[
            {"id": "HumanEval/0", "prompt": "def foo():\n"},
        ]):
            output_path = Path(pipeline.run())

        row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["diagnostics_version"] == 1
        assert row["retry_summary"] == {"attempts_total": 0}
        assert row["cascade_summary"] == {"cascade_triggers": 0}
        assert row["failure_reason_counts"] == {"unknown": 0}
        assert row["rescued_blocks"] == 0
        assert row["unrescued_blocks"] == 0
        assert "debug_blob" not in row
        assert "trace_id" not in row
    def test_run_returns_output_path(self, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="mbpp",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = self._build_generator(mock_result)
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "mbpp/1", "prompt": "Write a function"}
            ]):
                output_path = pipeline.run()
            assert "mbpp" in output_path
            assert output_path.endswith(".jsonl")

    def test_run_respects_sample_limit(self, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
                sample_limit=1,
            )
            generator = self._build_generator(mock_result)
            pipeline = WatermarkPipeline(generator=generator, config=cfg)

            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():\n"},
                {"id": "HumanEval/1", "prompt": "def bar():\n"},
            ]):
                output_path = pipeline.run()

            lines = Path(output_path).read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1
            assert generator.generate.call_count == 1

    def test_embed_rate_zero_blocks(self):
        """embed_rate is 0.0 when total_blocks is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            result = GenerateResult(
                code="", stats=EmbedStats(total_blocks=0, embedded_blocks=0,
                    failed_blocks=0, fallback_blocks=0),
            )
            generator = self._build_generator(result)
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
            generator = self._build_generator(GenerateResult(
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
            ))
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

            generator = self._build_generator(GenerateResult(
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
            ))
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():"}
            ]):
                output_path = pipeline.run()

            record = json.loads(Path(output_path).read_text(encoding="utf-8").strip())
            assert record["adaptive_mode"] == "piecewise_quantile"
            assert record["profile_id"] == "entropy-profile-v1"
            assert record["blocks"][0]["gamma_effective"] == 0.75


def test_build_public_watermark_params_requires_public_config():
    generator = SimpleNamespace(_config=SimpleNamespace(
        lsh_d=4,
        lsh_gamma=0.75,
        margin_base=0.1,
        margin_alpha=0.05,
    ))

    with pytest.raises(ValueError, match="via .config"):
        WatermarkPipeline._build_public_watermark_params(generator)
