from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.run_sawr_detect as sawr_detect_cli


def test_calibrate_cli_loads_scorer_and_writes_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "negative.jsonl"
    input_path.write_text('{"id": "neg", "final_code": "def target():\\n    return 0\\n"}\n', encoding="utf-8")
    output_path = tmp_path / "calibration.json"
    encoder_path = tmp_path / "codet5"
    encoder_path.mkdir()
    pipeline = MagicMock()
    artifact = MagicMock()
    pipeline.calibrate.return_value = artifact

    with (
        patch("scripts.run_sawr_detect.load_sawr_window_scorer", return_value=object()) as load_scorer,
        patch("scripts.run_sawr_detect.SawrDetectionPipeline", return_value=pipeline) as pipeline_cls,
        patch("scripts.run_sawr_detect.write_calibration_artifact") as write_artifact,
    ):
        rc = sawr_detect_cli.main([
            "calibrate",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--encoder-model-path",
            str(encoder_path),
            "--secret-key",
            "1010",
            "--device",
            "cpu",
        ])

    assert rc == 0
    load_scorer.assert_called_once()
    pipeline_cls.assert_called_once()
    pipeline.calibrate.assert_called_once()
    write_artifact.assert_called_once_with(output_path, artifact)


def test_detect_cli_loads_calibration_and_writes_details(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text('{"id": "pos", "final_code": "def target():\\n    return 1\\n"}\n', encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    output_path = tmp_path / "details.jsonl"
    encoder_path = tmp_path / "codet5"
    encoder_path.mkdir()
    pipeline = MagicMock()
    artifact = MagicMock()

    with (
        patch("scripts.run_sawr_detect.load_sawr_window_scorer", return_value=object()),
        patch("scripts.run_sawr_detect.SawrDetectionPipeline", return_value=pipeline),
        patch("scripts.run_sawr_detect.load_calibration_artifact", return_value=artifact),
    ):
        rc = sawr_detect_cli.main([
            "detect",
            "--input",
            str(input_path),
            "--calibration",
            str(calibration_path),
            "--output",
            str(output_path),
            "--encoder-model-path",
            str(encoder_path),
            "--secret-key",
            "1010",
            "--device",
            "cpu",
        ])

    assert rc == 0
    pipeline.detect_to_jsonl.assert_called_once()


def test_evaluate_cli_writes_report(tmp_path: Path) -> None:
    positive_path = tmp_path / "positive.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    output_path = tmp_path / "report.json"
    detail = {
        "id": "x",
        "score": 0.5,
        "is_watermarked": True,
        "insufficient_evidence": False,
    }
    positive_path.write_text(json.dumps(detail) + "\n", encoding="utf-8")
    negative_path.write_text(json.dumps({**detail, "is_watermarked": False}) + "\n", encoding="utf-8")

    rc = sawr_detect_cli.main([
        "evaluate",
        "--positive-details",
        str(positive_path),
        "--negative-details",
        str(negative_path),
        "--output",
        str(output_path),
    ])

    assert rc == 0
    assert output_path.exists()


def test_split_cli_writes_task_based_splits(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    output_dir = tmp_path / "splits"
    input_path.write_text(
        '{"id": "HumanEval/0#0"}\n{"id": "HumanEval/0#1"}\n{"id": "HumanEval/1#0"}\n',
        encoding="utf-8",
    )

    rc = sawr_detect_cli.main([
        "split",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--dev-ratio",
        "0.34",
        "--calibration-ratio",
        "0.33",
        "--seed",
        "0",
    ])

    assert rc == 0
    assert (output_dir / "dev.jsonl").exists()
    assert (output_dir / "calibration.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
