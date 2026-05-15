"""Tests for wfcllm.extract.calibration.runner.calibrate_threshold_from_corpus."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_calibrate_threshold_from_corpus_writes_result_json(tmp_path):
    from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus

    input_path = tmp_path / "corpus.jsonl"
    output_path = tmp_path / "threshold.json"
    _write_jsonl(input_path, [{"generated_code": "x = 1"} for _ in range(8)])

    fake_calibrator = MagicMock()
    fake_calibrator.calibrate.return_value = {
        "fpr": 0.01,
        "fpr_threshold": 1.234,
        "n_samples": 8,
    }

    with (
        patch("wfcllm.extract.calibration.runner.AutoTokenizer") as mock_tok,
        patch("wfcllm.extract.calibration.runner.AutoModel") as mock_model,
        patch("wfcllm.extract.calibration.runner.LSHSpace"),
        patch("wfcllm.extract.calibration.runner.WatermarkKeying"),
        patch("wfcllm.extract.calibration.runner.ProjectionVerifier"),
        patch("wfcllm.extract.calibration.runner.BlockScorer"),
        patch(
            "wfcllm.extract.calibration.runner.ThresholdCalibrator",
            return_value=fake_calibrator,
        ),
    ):
        mock_tok.from_pretrained.return_value = MagicMock()
        encoder_instance = MagicMock()
        encoder_instance.to.return_value = encoder_instance
        encoder_instance.eval.return_value = encoder_instance
        mock_model.from_pretrained.return_value = encoder_instance

        result_path = calibrate_threshold_from_corpus(
            input=str(input_path),
            output=str(output_path),
            secret_key="k",
            model="data/models/codet5-base",
            device="cpu",
            fpr=0.01,
        )

    assert Path(result_path) == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["fpr"] == 0.01
    assert payload["fpr_threshold"] == pytest.approx(1.234)
    assert payload["n_samples"] == 8
    fake_calibrator.calibrate.assert_called_once()
    args_passed, kwargs_passed = fake_calibrator.calibrate.call_args
    assert kwargs_passed.get("fpr") == 0.01 or (len(args_passed) >= 2 and args_passed[1] == 0.01)
    assert len(args_passed[0]) == 8


def test_calibrate_threshold_from_corpus_returns_path_object(tmp_path):
    from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus

    input_path = tmp_path / "corpus.jsonl"
    output_path = tmp_path / "nested" / "threshold.json"
    _write_jsonl(input_path, [{"generated_code": "x = 1"}])

    fake_calibrator = MagicMock()
    fake_calibrator.calibrate.return_value = {"fpr": 0.05, "fpr_threshold": 0.5, "n_samples": 1}

    with (
        patch("wfcllm.extract.calibration.runner.AutoTokenizer"),
        patch("wfcllm.extract.calibration.runner.AutoModel") as mock_model,
        patch("wfcllm.extract.calibration.runner.LSHSpace"),
        patch("wfcllm.extract.calibration.runner.WatermarkKeying"),
        patch("wfcllm.extract.calibration.runner.ProjectionVerifier"),
        patch("wfcllm.extract.calibration.runner.BlockScorer"),
        patch(
            "wfcllm.extract.calibration.runner.ThresholdCalibrator",
            return_value=fake_calibrator,
        ),
    ):
        encoder_instance = MagicMock()
        encoder_instance.to.return_value = encoder_instance
        encoder_instance.eval.return_value = encoder_instance
        mock_model.from_pretrained.return_value = encoder_instance

        result = calibrate_threshold_from_corpus(
            input=input_path,
            output=output_path,
            secret_key="k",
            model="any",
            device="cpu",
        )

    assert isinstance(result, Path)
    assert result.parent.exists()  # parent dir auto-created
    assert result == output_path
