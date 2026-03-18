"""Tests for wfcllm.extract.pipeline."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig
from wfcllm.extract.config import DetectionResult


class TestExtractPipelineConfig:
    def test_default_fields(self):
        cfg = ExtractPipelineConfig(
            input_file="data/watermarked/humaneval_20260309.jsonl",
            output_dir="data/results",
        )
        assert cfg.input_file == "data/watermarked/humaneval_20260309.jsonl"
        assert cfg.output_dir == "data/results"
        assert cfg.resume is None


def _make_detection_result(is_watermarked: bool, z_score: float) -> DetectionResult:
    return DetectionResult(
        is_watermarked=is_watermarked,
        z_score=z_score,
        p_value=0.001 if is_watermarked else 0.5,
        total_blocks=10,
        independent_blocks=8,
        hit_blocks=7 if is_watermarked else 4,
        block_details=[],
    )


class TestExtractPipelineStatistics:
    """Tests for statistical computation in run()."""

    def _make_jsonl(self, tmpdir: str, n: int = 4) -> str:
        path = Path(tmpdir) / "test.jsonl"
        records = [
            {
                "id": f"HumanEval/{i}",
                "dataset": "humaneval",
                "prompt": f"def f{i}():\n",
                "generated_code": f"def f{i}():\n    return {i}\n",
                "total_blocks": 5,
                "embedded_blocks": 3,
                "failed_blocks": 1,
                "fallback_blocks": 0,
                "embed_rate": 0.6,
            }
            for i in range(n)
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))
        return str(path)

    def test_run_creates_details_and_summary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = self._make_jsonl(tmpdir, n=4)
            cfg = ExtractPipelineConfig(
                input_file=jsonl_path,
                output_dir=tmpdir,
            )
            detector = MagicMock()
            # 3 watermarked, 1 not
            detector.detect.side_effect = [
                _make_detection_result(True, 4.5),
                _make_detection_result(True, 3.8),
                _make_detection_result(True, 5.1),
                _make_detection_result(False, 1.2),
            ]

            pipeline = ExtractPipeline(detector=detector, config=cfg)
            details_path = pipeline.run()
            details = Path(details_path)
            summary = details.parent / "test_summary.json"

            assert details.exists()
            assert details.name == "test_details.jsonl"
            assert summary.exists()

            detail_rows = [
                json.loads(line)
                for line in details.read_text(encoding="utf-8").splitlines()
            ]
            summary_doc = json.loads(summary.read_text(encoding="utf-8"))

            # meta
            assert summary_doc["meta"]["total_samples"] == 4
            assert summary_doc["meta"]["input_file"] == jsonl_path

            # summary
            assert abs(summary_doc["summary"]["watermark_rate"] - 0.75) < 1e-6
            assert len(summary_doc["summary"]["watermark_rate_ci_95"]) == 2
            assert summary_doc["summary"]["mean_z_score"] == pytest.approx(
                (4.5 + 3.8 + 5.1 + 1.2) / 4, abs=1e-4
            )
            assert "std_z_score" in summary_doc["summary"]
            assert "mean_p_value" in summary_doc["summary"]
            assert "mean_blocks" in summary_doc["summary"]
            assert "embed_rate_distribution" in summary_doc["summary"]

            dist = summary_doc["summary"]["embed_rate_distribution"]
            assert "mean" in dist
            assert "std" in dist
            assert "p25" in dist
            assert "p50" in dist
            assert "p75" in dist

            assert len(detail_rows) == 4
            first = detail_rows[0]
            assert first["id"] == "HumanEval/0"
            assert first["is_watermarked"] is True
            assert "z_score" in first
            assert "p_value" in first
            assert "independent_blocks" in first
            assert "hits" in first

    def test_watermark_rate_ci_lower_le_upper(self):
        """CI lower bound should be <= upper bound."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = self._make_jsonl(tmpdir, n=10)
            cfg = ExtractPipelineConfig(input_file=jsonl_path, output_dir=tmpdir)
            detector = MagicMock()
            detector.detect.side_effect = [
                _make_detection_result(i % 2 == 0, float(i)) for i in range(10)
            ]
            pipeline = ExtractPipeline(detector=detector, config=cfg)
            details_path = pipeline.run()
            summary_path = Path(details_path).parent / "test_summary.json"
            summary_doc = json.loads(summary_path.read_text())
            lo, hi = summary_doc["summary"]["watermark_rate_ci_95"]
            assert lo <= hi
