from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.wfcllm_v2_detect as detect_cli


def _row() -> dict[str, str]:
    return {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def target():\n",
        "final_code": "def target():\n    return 1\n",
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _detector_args(model_path: Path) -> list[str]:
    return [
        "--encoder-model-path",
        str(model_path),
        "--signature-bits",
        "16",
        "--min-canonical-units",
        "1",
        "--target-fpr",
        "0.05",
        "--encoder-use-lora",
        "--encoder-use-bf16",
    ]


def test_v2_calibrate_cli_uses_environment_secret_and_writes_v2_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "encoder"
    model_path.mkdir()
    input_path = tmp_path / "negative.jsonl"
    output_path = tmp_path / "calibration.json"
    _write_jsonl(input_path, [_row()])
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "env-secret-key")
    scorer = object()
    pipeline = MagicMock()
    artifact = object()
    pipeline.calibrate.return_value = artifact

    with (
        patch(
            "scripts.wfcllm_v2_detect.load_v2_signature_scorer",
            return_value=scorer,
        ) as load_scorer,
        patch(
            "scripts.wfcllm_v2_detect.WFCLLMV2DetectionPipeline",
            return_value=pipeline,
        ) as pipeline_cls,
        patch("scripts.wfcllm_v2_detect.write_v2_calibration_artifact") as write,
    ):
        rc = detect_cli.main(
            [
                "calibrate",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *_detector_args(model_path),
            ]
        )

    assert rc == 0
    config = pipeline_cls.call_args.kwargs["config"]
    assert config.secret_key == "env-secret-key"
    assert config.signature_bits == 16
    assert config.min_canonical_units == 1
    assert load_scorer.call_args.kwargs["secret_key"] == "env-secret-key"
    assert pipeline_cls.call_args.kwargs["scorer"] is scorer
    pipeline.calibrate.assert_called_once_with([_row()])
    write.assert_called_once_with(output_path, artifact)


def test_v2_detect_cli_reads_only_strict_final_code_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "encoder"
    model_path.mkdir()
    input_path = tmp_path / "positive.jsonl"
    calibration_path = tmp_path / "calibration.json"
    output_path = tmp_path / "details.jsonl"
    _write_jsonl(input_path, [_row()])
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "env-secret-key")
    pipeline = MagicMock()
    artifact = object()

    with (
        patch("scripts.wfcllm_v2_detect.load_v2_signature_scorer"),
        patch(
            "scripts.wfcllm_v2_detect.WFCLLMV2DetectionPipeline",
            return_value=pipeline,
        ),
        patch(
            "scripts.wfcllm_v2_detect.load_v2_calibration_artifact",
            return_value=artifact,
        ),
    ):
        rc = detect_cli.main(
            [
                "detect",
                "--input",
                str(input_path),
                "--calibration",
                str(calibration_path),
                "--output",
                str(output_path),
                *_detector_args(model_path),
            ]
        )

    assert rc == 0
    pipeline.detect_to_jsonl.assert_called_once_with(
        [_row()],
        artifact=artifact,
        output_path=output_path,
    )


def test_v2_detect_cli_missing_secret_fails_before_encoder_load(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    model_path = tmp_path / "encoder"
    model_path.mkdir()
    input_path = tmp_path / "negative.jsonl"
    _write_jsonl(input_path, [_row()])
    monkeypatch.delenv("WFCLLM_SECRET_KEY", raising=False)

    with patch("scripts.wfcllm_v2_detect.load_v2_signature_scorer") as load_scorer:
        rc = detect_cli.main(
            [
                "calibrate",
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "calibration.json"),
                *_detector_args(model_path),
            ]
        )

    captured = capsys.readouterr()
    assert rc == 1
    assert "WFCLLM_SECRET_KEY" in captured.err
    assert "env-secret-key" not in captured.err
    load_scorer.assert_not_called()


def test_v2_detect_cli_evaluate_reuses_standard_detection_report(tmp_path: Path) -> None:
    positive_path = tmp_path / "positive.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    output_path = tmp_path / "report.json"
    _write_jsonl(positive_path, [_row()])
    _write_jsonl(negative_path, [_row()])
    report = {"primary": {"tpr_at_5fpr": 0.5}}

    with (
        patch("scripts.wfcllm_v2_detect.build_detection_report", return_value=report),
        patch("scripts.wfcllm_v2_detect.write_detection_report") as write,
    ):
        rc = detect_cli.main(
            [
                "evaluate",
                "--positive-details",
                str(positive_path),
                "--negative-details",
                str(negative_path),
                "--output",
                str(output_path),
            ]
        )

    assert rc == 0
    write.assert_called_once_with(output_path, report)
