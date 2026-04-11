"""Tests for wfcllm.extract.pipeline."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wfcllm.extract.alignment import compare_block_contracts
from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig
from wfcllm.extract.config import DetectionResult
from wfcllm.extract.calibrator import ThresholdCalibrator
from wfcllm.extract.hypothesis import JointDetectionResult
from wfcllm.extract.hypothesis import LexicalDetectionResult


def _contract(
    *,
    ordinal: int,
    block_id: str,
    entropy_units: int = 100,
) -> dict:
    return {
        "ordinal": ordinal,
        "block_id": block_id,
        "node_type": "expression_statement",
        "parent_node_type": "module",
        "block_text_hash": f"hash-{block_id}",
        "start_line": ordinal + 1,
        "end_line": ordinal + 1,
        "entropy_units": entropy_units,
        "gamma_target": 0.0,
        "k": 0,
        "gamma_effective": 0.0,
    }


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
    result = DetectionResult(
        is_watermarked=is_watermarked,
        z_score=z_score,
        p_value=0.001 if is_watermarked else 0.5,
        total_blocks=10,
        independent_blocks=8,
        hit_blocks=7 if is_watermarked else 4,
        block_details=[],
    )
    result.lexical_result = LexicalDetectionResult(
        num_positions_scored=6,
        num_green_hits=4,
        green_fraction=4 / 6,
        lexical_z_score=1.5,
        lexical_p_value=0.2,
    )
    result.joint_result = JointDetectionResult(
        semantic_z=z_score,
        lexical_z=1.5,
        joint_score=z_score + 0.75,
        p_joint=0.05,
        prediction=is_watermarked,
        confidence=0.95,
        rationale="semantic borderline, lexical supportive",
    )
    result.semantic_result = result
    return result


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
            assert first["num_positions_scored"] == 6
            assert first["num_green_hits"] == 4
            assert first["lexical_z_score"] == 1.5
            assert first["joint_score"] == pytest.approx(5.25)
            assert first["prediction"] is True

    def test_run_uses_spec_required_lexical_count_field_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = self._make_jsonl(tmpdir, n=1)
            cfg = ExtractPipelineConfig(input_file=jsonl_path, output_dir=tmpdir)
            detector = MagicMock()
            detector.detect.return_value = _make_detection_result(True, 4.5)

            pipeline = ExtractPipeline(detector=detector, config=cfg)
            details_path = pipeline.run()
            row = json.loads(Path(details_path).read_text(encoding="utf-8").splitlines()[0])

            assert row["num_positions_scored"] == 6
            assert row["num_green_hits"] == 4
            assert "lexical_num_positions_scored" not in row
            assert "lexical_num_green_hits" not in row

    def test_summary_includes_declared_calibration_regime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = self._make_jsonl(tmpdir, n=1)
            detector = MagicMock()
            detector.detect.return_value = _make_detection_result(True, 4.5)
            pipeline = ExtractPipeline(
                detector=detector,
                config=ExtractPipelineConfig(
                    input_file=jsonl_path,
                    output_dir=tmpdir,
                    summary_metadata={
                        "calibration": {
                            "source": "data/negative_corpus.jsonl",
                            "fpr": 0.05,
                            "threshold": 1.2,
                            "hypothesis_mode": "adaptive",
                            "statistic_definition": "sum(gamma_i), sum(gamma_i*(1-gamma_i))",
                            "decision_rule": "z_score >= threshold",
                        }
                    },
                ),
            )

            details_path = pipeline.run()
            summary_doc = json.loads(
                Path(details_path).with_name("test_summary.json").read_text(encoding="utf-8")
            )

            assert summary_doc["meta"]["calibration"] == {
                "source": "data/negative_corpus.jsonl",
                "fpr": 0.05,
                "threshold": 1.2,
                "hypothesis_mode": "adaptive",
                "statistic_definition": "sum(gamma_i), sum(gamma_i*(1-gamma_i))",
                "decision_rule": "z_score >= threshold",
            }

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

    def test_run_surfaces_contract_alignment_fields_when_metadata_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code = "x = 1\n"
            embedded_contracts = [_contract(ordinal=0, block_id="0"), _contract(ordinal=1, block_id="1")]
            rebuilt_contracts = [_contract(ordinal=0, block_id="0")]
            report = compare_block_contracts(embedded_contracts, rebuilt_contracts)

            input_path = Path(tmpdir) / "test.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "generated_code": code,
                        "blocks": embedded_contracts,
                        "adaptive_mode": "piecewise",
                        "profile_id": "entropy-profile-v1",
                        "embed_rate": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            detector = MagicMock()
            detector.detect.return_value = DetectionResult(
                is_watermarked=False,
                z_score=1.0,
                p_value=0.5,
                total_blocks=1,
                independent_blocks=1,
                hit_blocks=0,
                block_details=[],
                alignment_report=report,
                contract_valid=report.contract_valid,
            )

            pipeline = ExtractPipeline(
                detector=detector,
                config=ExtractPipelineConfig(
                    input_file=str(input_path),
                    output_dir=tmpdir,
                ),
            )

            details_path = pipeline.run()
            detail_rows = Path(details_path).read_text(encoding="utf-8").splitlines()
            row = json.loads(detail_rows[0])
            _, kwargs = detector.detect.call_args

            assert kwargs["watermark_metadata"]["blocks"] == embedded_contracts
            assert kwargs["watermark_metadata"]["adaptive_mode"] == "piecewise"
            assert kwargs["watermark_metadata"]["profile_id"] == "entropy-profile-v1"
            assert row["contract_valid"] is False
            assert row["contract_alignment"]["status"] == "structure_mismatch"
            assert row["contract_alignment"]["block_count_mismatch"] is True

    def test_run_without_metadata_keeps_detail_rows_legacy_shaped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = self._make_jsonl(tmpdir, n=1)
            detector = MagicMock()
            detector.detect.return_value = _make_detection_result(True, 4.5)

            pipeline = ExtractPipeline(
                detector=detector,
                config=ExtractPipelineConfig(
                    input_file=jsonl_path,
                    output_dir=tmpdir,
                ),
            )

            details_path = pipeline.run()
            row = json.loads(Path(details_path).read_text(encoding="utf-8").splitlines()[0])

            _, kwargs = detector.detect.call_args
            assert kwargs == {}
            assert "contract_valid" not in row
            assert "contract_alignment" not in row

    def test_summary_excludes_invalid_samples_when_policy_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "HumanEval/0",
                                "generated_code": "x = 1\n",
                                "blocks": [_contract(ordinal=0, block_id="0")],
                                "embed_rate": 1.0,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "HumanEval/1",
                                "generated_code": "y = 2\n",
                                "blocks": [_contract(ordinal=0, block_id="0")],
                                "embed_rate": 0.0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            detector = MagicMock()
            detector._config.adaptive_detection.exclude_invalid_samples = True
            detector.detect.side_effect = [
                DetectionResult(
                    is_watermarked=True,
                    z_score=4.0,
                    p_value=0.001,
                    total_blocks=1,
                    independent_blocks=1,
                    hit_blocks=1,
                    block_details=[],
                    contract_valid=True,
                    alignment_report=compare_block_contracts(
                        [_contract(ordinal=0, block_id="0")],
                        [_contract(ordinal=0, block_id="0")],
                    ),
                ),
                DetectionResult(
                    is_watermarked=False,
                    z_score=0.5,
                    p_value=0.6,
                    total_blocks=1,
                    independent_blocks=1,
                    hit_blocks=0,
                    block_details=[],
                    contract_valid=False,
                    alignment_report=compare_block_contracts(
                        [_contract(ordinal=0, block_id="0"), _contract(ordinal=1, block_id="1")],
                        [_contract(ordinal=0, block_id="0")],
                    ),
                ),
            ]

            pipeline = ExtractPipeline(
                detector=detector,
                config=ExtractPipelineConfig(
                    input_file=str(input_path),
                    output_dir=tmpdir,
                ),
            )

            details_path = pipeline.run()
            summary_doc = json.loads((Path(details_path).parent / "test_summary.json").read_text(encoding="utf-8"))

            assert summary_doc["meta"]["total_samples"] == 2
            assert summary_doc["meta"]["scored_samples"] == 1
            assert summary_doc["meta"]["invalid_samples"] == 1
            assert summary_doc["summary"]["watermark_rate"] == 1.0
            assert summary_doc["summary"]["mean_z_score"] == 4.0

    def test_summary_distinguishes_modes_and_invalid_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "HumanEval/0",
                                "generated_code": "x = 1\n",
                                "blocks": [_contract(ordinal=0, block_id="0")],
                                "adaptive_mode": "fixed",
                                "embed_rate": 1.0,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "HumanEval/1",
                                "generated_code": "y = 2\n",
                                "blocks": [_contract(ordinal=0, block_id="0", entropy_units=101)],
                                "adaptive_mode": "piecewise_quantile",
                                "embed_rate": 1.0,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "HumanEval/2",
                                "generated_code": "z = 3\n",
                                "blocks": [
                                    _contract(ordinal=0, block_id="0"),
                                    _contract(ordinal=1, block_id="1"),
                                ],
                                "adaptive_mode": "piecewise_quantile",
                                "embed_rate": 0.0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            aligned_report = compare_block_contracts(
                [_contract(ordinal=0, block_id="0")],
                [_contract(ordinal=0, block_id="0")],
            )
            numeric_mismatch_report = compare_block_contracts(
                [_contract(ordinal=0, block_id="0", entropy_units=101)],
                [_contract(ordinal=0, block_id="0", entropy_units=100)],
            )
            structure_mismatch_report = compare_block_contracts(
                [_contract(ordinal=0, block_id="0"), _contract(ordinal=1, block_id="1")],
                [_contract(ordinal=0, block_id="0")],
            )

            detector = MagicMock()
            detector._config.adaptive_detection.exclude_invalid_samples = False
            detector.detect.side_effect = [
                DetectionResult(
                    is_watermarked=True,
                    z_score=4.0,
                    p_value=0.001,
                    total_blocks=1,
                    independent_blocks=1,
                    hit_blocks=1,
                    block_details=[],
                    hypothesis_mode="fixed",
                    contract_valid=True,
                    alignment_report=aligned_report,
                ),
                DetectionResult(
                    is_watermarked=False,
                    z_score=0.5,
                    p_value=0.6,
                    total_blocks=1,
                    independent_blocks=1,
                    hit_blocks=0,
                    block_details=[],
                    hypothesis_mode="adaptive",
                    contract_valid=False,
                    alignment_report=numeric_mismatch_report,
                ),
                DetectionResult(
                    is_watermarked=False,
                    z_score=0.2,
                    p_value=0.8,
                    total_blocks=1,
                    independent_blocks=1,
                    hit_blocks=0,
                    block_details=[],
                    hypothesis_mode="adaptive",
                    contract_valid=False,
                    alignment_report=structure_mismatch_report,
                ),
            ]

            pipeline = ExtractPipeline(
                detector=detector,
                config=ExtractPipelineConfig(
                    input_file=str(input_path),
                    output_dir=tmpdir,
                ),
            )

            details_path = pipeline.run()
            detail_rows = [
                json.loads(line)
                for line in Path(details_path).read_text(encoding="utf-8").splitlines()
            ]
            summary_doc = json.loads(
                (Path(details_path).parent / "test_summary.json").read_text(encoding="utf-8")
            )

            assert detail_rows[0]["mode"] == "fixed"
            assert detail_rows[1]["mode"] == "adaptive"
            assert detail_rows[1]["alignment_ok"] is False
            assert detail_rows[2]["alignment_ok"] is False
            assert summary_doc["summary"]["mode_counts"] == {
                "fixed": 1,
                "adaptive": 2,
            }
            assert summary_doc["summary"]["invalid_reason_counts"] == {
                "alignment_failed": 1,
                "adaptive_contract_invalid": 1,
            }


def test_calibrated_threshold_smoke_range():
    resolved_threshold = ThresholdCalibrator._percentile_threshold(
        z_scores=[0.3, 0.8, 1.1, 1.6, 2.0],
        fpr=0.2,
    )
    assert 0.5 < resolved_threshold < 2.5


def test_debug_extract_alignment_smoke_resolves_embedded_params_without_diag(tmp_path):
    from tools.debug_extract_alignment import build_debug_payload

    input_file = tmp_path / "watermarked.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": "def foo():\n",
                "generated_code": "def foo():\n    return 1\n",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            }
        ) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps({"extract": {"lsh_d": 3, "lsh_gamma": 0.5}}),
        encoding="utf-8",
    )

    payload = build_debug_payload(
        prompt_id="HumanEval/0",
        input_file=str(input_file),
        use_embedded_params=True,
        config_path=str(config_file),
        diag_details=None,
    )

    assert payload["prompt_id"] == "HumanEval/0"
    assert payload["resolved_lsh_d"] == 4
    assert payload["resolved_lsh_gamma"] == 0.75
    assert payload["diagnostic_report_found"] is False


def test_debug_extract_alignment_auto_discovers_diag_details(tmp_path):
    from tools.debug_extract_alignment import build_debug_payload

    watermarked_dir = tmp_path / "data" / "watermarked"
    watermarked_dir.mkdir(parents=True)
    input_file = watermarked_dir / "sample.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": "def foo():\n",
                "generated_code": "def foo():\n    return 1\n",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            }
        ) + "\n",
        encoding="utf-8",
    )

    diag_dir = tmp_path / "data" / "diag_reports"
    diag_dir.mkdir(parents=True)
    details_file = diag_dir / "details_20990101_000000.jsonl"
    details_file.write_text(
        json.dumps(
            {
                "prompt_id": "HumanEval/0",
                "text_mismatch_count": 1,
                "parent_mismatch_count": 0,
                "score_disagree_count": 0,
                "aligned_pairs": [],
            }
        ) + "\n",
        encoding="utf-8",
    )

    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps({"extract": {"lsh_d": 3, "lsh_gamma": 0.5}}),
        encoding="utf-8",
    )

    payload = build_debug_payload(
        prompt_id="HumanEval/0",
        input_file=str(input_file),
        use_embedded_params=True,
        config_path=str(config_file),
        diag_details=None,
    )

    assert payload["diagnostic_report_found"] is True
    assert payload["diagnostic_details_file"] == str(details_file)
    assert payload["text_mismatch_count"] == 1


def test_debug_extract_alignment_cli_runs_from_script_path(tmp_path):
    input_file = tmp_path / "watermarked.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": "def foo():\n",
                "generated_code": "def foo():\n    return 1\n",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            }
        ) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps({"extract": {"lsh_d": 3, "lsh_gamma": 0.5}}),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "tools/debug_extract_alignment.py",
            "--prompt-id",
            "HumanEval/0",
            "--input-file",
            str(input_file),
            "--use-embedded-params",
            "--config",
            str(config_file),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "resolved_lsh_d" in result.stdout
