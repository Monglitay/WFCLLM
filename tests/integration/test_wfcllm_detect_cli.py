from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.wfcllm_detect as detect_cli


def test_calibrate_cli_maps_detector_args_and_writes_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "negative.jsonl"
    input_record = _final_code_row("neg", "def target():\n    return 0\n")
    _write_jsonl(input_path, [input_record])
    output_path = tmp_path / "calibration.json"
    detector_paths = _make_detector_paths(tmp_path)
    pipeline = MagicMock()
    artifact = MagicMock()
    scorer = object()
    pipeline.calibrate.return_value = artifact

    with (
        patch(
            "scripts.wfcllm_detect.load_wfcllm_window_scorer",
            return_value=scorer,
        ) as load_scorer,
        patch(
            "scripts.wfcllm_detect.WFCLLMDetectionPipeline",
            return_value=pipeline,
        ) as pipeline_cls,
        patch("scripts.wfcllm_detect.write_calibration_artifact") as write_artifact,
    ):
        rc = detect_cli.main(
            [
                "calibrate",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *_detector_cli_args(detector_paths),
            ]
        )

    assert rc == 0
    _assert_detector_mapping(
        load_scorer=load_scorer,
        pipeline_cls=pipeline_cls,
        scorer=scorer,
        detector_paths=detector_paths,
    )
    pipeline.calibrate.assert_called_once_with([input_record])
    write_artifact.assert_called_once_with(output_path, artifact)


def test_detect_cli_maps_detector_args_and_writes_details(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_record = _final_code_row("pos", "def target():\n    return 1\n")
    _write_jsonl(input_path, [input_record])
    calibration_path = tmp_path / "calibration.json"
    output_path = tmp_path / "details.jsonl"
    detector_paths = _make_detector_paths(tmp_path)
    pipeline = MagicMock()
    artifact = MagicMock()
    scorer = object()

    with (
        patch(
            "scripts.wfcllm_detect.load_wfcllm_window_scorer",
            return_value=scorer,
        ) as load_scorer,
        patch(
            "scripts.wfcllm_detect.WFCLLMDetectionPipeline",
            return_value=pipeline,
        ) as pipeline_cls,
        patch(
            "scripts.wfcllm_detect.load_calibration_artifact",
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
                *_detector_cli_args(detector_paths),
            ]
        )

    assert rc == 0
    _assert_detector_mapping(
        load_scorer=load_scorer,
        pipeline_cls=pipeline_cls,
        scorer=scorer,
        detector_paths=detector_paths,
    )
    pipeline.detect_to_jsonl.assert_called_once_with(
        [input_record],
        artifact=artifact,
        output_path=output_path,
    )


def test_calibrate_cli_uses_env_secret_key_when_cli_secret_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "negative.jsonl"
    input_record = _final_code_row("neg", "def target():\n    return 0\n")
    _write_jsonl(input_path, [input_record])
    output_path = tmp_path / "calibration.json"
    detector_paths = _make_detector_paths(tmp_path)
    pipeline = MagicMock()
    artifact = MagicMock()
    scorer = object()
    pipeline.calibrate.return_value = artifact
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "env-secret-key")

    with (
        patch(
            "scripts.wfcllm_detect.load_wfcllm_window_scorer",
            return_value=scorer,
        ) as load_scorer,
        patch(
            "scripts.wfcllm_detect.WFCLLMDetectionPipeline",
            return_value=pipeline,
        ) as pipeline_cls,
        patch("scripts.wfcllm_detect.write_calibration_artifact") as write_artifact,
    ):
        rc = detect_cli.main(
            [
                "calibrate",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *_detector_cli_args(detector_paths, secret_key=None),
            ]
        )

    assert rc == 0
    _assert_detector_mapping(
        load_scorer=load_scorer,
        pipeline_cls=pipeline_cls,
        scorer=scorer,
        detector_paths=detector_paths,
        expected_secret_key="env-secret-key",
    )
    pipeline.calibrate.assert_called_once_with([input_record])
    write_artifact.assert_called_once_with(output_path, artifact)


def test_detect_cli_secret_key_arg_takes_precedence_over_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_record = _final_code_row("pos", "def target():\n    return 1\n")
    _write_jsonl(input_path, [input_record])
    calibration_path = tmp_path / "calibration.json"
    output_path = tmp_path / "details.jsonl"
    detector_paths = _make_detector_paths(tmp_path)
    pipeline = MagicMock()
    artifact = MagicMock()
    scorer = object()
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "env-secret-key")

    with (
        patch(
            "scripts.wfcllm_detect.load_wfcllm_window_scorer",
            return_value=scorer,
        ) as load_scorer,
        patch(
            "scripts.wfcllm_detect.WFCLLMDetectionPipeline",
            return_value=pipeline,
        ) as pipeline_cls,
        patch(
            "scripts.wfcllm_detect.load_calibration_artifact",
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
                *_detector_cli_args(detector_paths, secret_key="cli-secret-key"),
            ]
        )

    assert rc == 0
    _assert_detector_mapping(
        load_scorer=load_scorer,
        pipeline_cls=pipeline_cls,
        scorer=scorer,
        detector_paths=detector_paths,
        expected_secret_key="cli-secret-key",
    )


def test_calibrate_missing_secret_key_fails_without_leaking_value(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "negative.jsonl"
    _write_jsonl(input_path, [_final_code_row("neg", "def target():\n    return 0\n")])
    output_path = tmp_path / "calibration.json"
    detector_paths = _make_detector_paths(tmp_path)
    monkeypatch.delenv("WFCLLM_SECRET_KEY", raising=False)

    with patch("scripts.wfcllm_detect.load_wfcllm_window_scorer") as load_scorer:
        rc = detect_cli.main(
            [
                "calibrate",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *_detector_cli_args(detector_paths, secret_key=None),
            ]
        )

    captured = capsys.readouterr()
    assert rc == 1
    assert "[错误]" in captured.err
    assert "WFCLLM_SECRET_KEY" in captured.err
    assert "env-secret-key" not in captured.err
    assert "cli-secret-key" not in captured.err
    load_scorer.assert_not_called()


def test_calibrate_missing_input_fails_before_scorer_load(
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "calibration.json"
    detector_paths = _make_detector_paths(tmp_path)

    with patch("scripts.wfcllm_detect.load_wfcllm_window_scorer") as load_scorer:
        rc = detect_cli.main(
            [
                "calibrate",
                "--input",
                str(tmp_path / "missing.jsonl"),
                "--output",
                str(output_path),
                *_detector_cli_args(detector_paths),
            ]
        )

    assert rc == 1
    assert "[错误]" in capsys.readouterr().err
    load_scorer.assert_not_called()


def test_detect_missing_calibration_fails_before_scorer_load(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "rows.jsonl"
    _write_jsonl(
        input_path,
        [_final_code_row("pos", "def target():\n    return 1\n")],
    )
    output_path = tmp_path / "details.jsonl"
    detector_paths = _make_detector_paths(tmp_path)

    with patch("scripts.wfcllm_detect.load_wfcllm_window_scorer") as load_scorer:
        rc = detect_cli.main(
            [
                "detect",
                "--input",
                str(input_path),
                "--calibration",
                str(tmp_path / "missing-calibration.json"),
                "--output",
                str(output_path),
                *_detector_cli_args(detector_paths),
            ]
        )

    assert rc == 1
    assert "[错误]" in capsys.readouterr().err
    load_scorer.assert_not_called()


def test_detect_missing_input_fails_before_scorer_load(
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "details.jsonl"
    artifact = MagicMock()
    detector_paths = _make_detector_paths(tmp_path)

    with (
        patch(
            "scripts.wfcllm_detect.load_calibration_artifact",
            return_value=artifact,
        ),
        patch("scripts.wfcllm_detect.load_wfcllm_window_scorer") as load_scorer,
    ):
        rc = detect_cli.main(
            [
                "detect",
                "--input",
                str(tmp_path / "missing-rows.jsonl"),
                "--calibration",
                str(tmp_path / "calibration.json"),
                "--output",
                str(output_path),
                *_detector_cli_args(detector_paths),
            ]
        )

    assert rc == 1
    assert "[错误]" in capsys.readouterr().err
    load_scorer.assert_not_called()


def test_malformed_jsonl_fails_before_scorer_load(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "negative.jsonl"
    input_path.write_text('{"id": "broken"\n', encoding="utf-8")
    output_path = tmp_path / "calibration.json"
    detector_paths = _make_detector_paths(tmp_path)

    with patch("scripts.wfcllm_detect.load_wfcllm_window_scorer") as load_scorer:
        rc = detect_cli.main(
            [
                "calibrate",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *_detector_cli_args(detector_paths),
            ]
        )

    assert rc == 1
    assert "[错误]" in capsys.readouterr().err
    load_scorer.assert_not_called()


def test_evaluate_cli_writes_report_content_without_scorer_load(
    tmp_path: Path,
) -> None:
    positive_path = tmp_path / "positive.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    output_path = tmp_path / "report.json"
    detail = {
        "id": "x",
        "score": 0.5,
        "p_value": 0.5,
        "is_watermarked": True,
        "insufficient_evidence": False,
    }
    _write_jsonl(positive_path, [detail])
    _write_jsonl(negative_path, [{**detail, "is_watermarked": False}])

    with patch(
        "scripts.wfcllm_detect.load_wfcllm_window_scorer",
        side_effect=AssertionError("evaluate must not load scorer"),
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
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["primary"]["positive_samples"] == 1
    assert report["primary"]["negative_samples"] == 1
    assert report["primary"]["tpr_at_5fpr"] == 1.0
    assert report["primary"]["tpr_at_target_fpr"] == 1.0
    assert report["primary"]["fpr_target"] == 0.0
    assert report["primary"]["observed_fpr"] == 0.0
    assert report["data_coverage"]["p_value_present"] == 2


def test_evaluate_invalid_detail_row_returns_error(tmp_path: Path, capsys) -> None:
    positive_path = tmp_path / "positive.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    output_path = tmp_path / "report.json"
    _write_jsonl(
        positive_path,
        [{"id": "x", "is_watermarked": True, "insufficient_evidence": False}],
    )
    _write_jsonl(
        negative_path,
        [{"id": "y", "score": 0.1, "is_watermarked": False}],
    )

    with patch(
        "scripts.wfcllm_detect.load_wfcllm_window_scorer",
        side_effect=AssertionError("evaluate must not load scorer"),
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

    assert rc == 1
    assert "[错误]" in capsys.readouterr().err


def test_split_cli_writes_deterministic_task_based_splits(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    output_dir = tmp_path / "splits"
    repeat_output_dir = tmp_path / "splits-repeat"
    input_rows = [
        _final_code_row("HumanEval/0#0", "def target():\n    return 0\n"),
        _final_code_row("HumanEval/0#1", "def target():\n    return 1\n"),
        _final_code_row("HumanEval/1#0", "def target():\n    return 2\n"),
        _final_code_row("HumanEval/2#0", "def target():\n    return 3\n"),
    ]
    _write_jsonl(input_path, input_rows)

    with patch(
        "scripts.wfcllm_detect.load_wfcllm_window_scorer",
        side_effect=AssertionError("split must not load scorer"),
    ):
        rc = detect_cli.main(
            [
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
            ]
        )
        repeat_rc = detect_cli.main(
            [
                "split",
                "--input",
                str(input_path),
                "--output-dir",
                str(repeat_output_dir),
                "--dev-ratio",
                "0.34",
                "--calibration-ratio",
                "0.33",
                "--seed",
                "0",
            ]
        )

    assert rc == 0
    assert repeat_rc == 0
    split_rows = _read_split_rows(output_dir)
    assert _row_counter(row for rows in split_rows.values() for row in rows) == (
        _row_counter(input_rows)
    )
    task_locations: dict[str, str] = {}
    for split_name, rows in split_rows.items():
        for row in rows:
            task_id = row["id"].split("#", 1)[0]
            task_locations.setdefault(task_id, split_name)
            assert task_locations[task_id] == split_name

    for split_name in ("dev", "calibration", "test"):
        assert (output_dir / f"{split_name}.jsonl").read_text(
            encoding="utf-8"
        ) == (repeat_output_dir / f"{split_name}.jsonl").read_text(encoding="utf-8")


def test_split_invalid_ratio_returns_error(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "rows.jsonl"
    output_dir = tmp_path / "splits"
    _write_jsonl(
        input_path,
        [_final_code_row("HumanEval/0#0", "def target():\n    return 0\n")],
    )

    with patch(
        "scripts.wfcllm_detect.load_wfcllm_window_scorer",
        side_effect=AssertionError("split must not load scorer"),
    ):
        rc = detect_cli.main(
            [
                "split",
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--dev-ratio",
                "0.7",
                "--calibration-ratio",
                "0.4",
                "--seed",
                "0",
            ]
        )

    assert rc == 1
    assert "[错误]" in capsys.readouterr().err


def test_split_trace_row_returns_error_without_scorer_load(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "rows.jsonl"
    output_dir = tmp_path / "splits"
    _write_jsonl(
        input_path,
        [
            {
                **_final_code_row(
                    "HumanEval/0#0",
                    "def target():\n    return 0\n",
                ),
                "sampling_trace": [],
            }
        ],
    )

    with patch(
        "scripts.wfcllm_detect.load_wfcllm_window_scorer",
        side_effect=AssertionError("split must not load scorer"),
    ):
        rc = detect_cli.main(
            [
                "split",
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--dev-ratio",
                "0.0",
                "--calibration-ratio",
                "0.0",
                "--seed",
                "0",
            ]
        )

    assert rc == 1
    assert "sampling_trace" in capsys.readouterr().err


def _make_detector_paths(tmp_path: Path) -> dict[str, Path]:
    encoder_path = tmp_path / "codet5"
    encoder_path.mkdir()
    checkpoint_path = tmp_path / "encoder.pt"
    whitening_path = tmp_path / "whitening.json"
    return {
        "encoder_model_path": encoder_path,
        "encoder_checkpoint_path": checkpoint_path,
        "lsh_whitening_path": whitening_path,
    }


def _detector_cli_args(
    paths: dict[str, Path],
    *,
    secret_key: str | None = "1010",
) -> list[str]:
    args = [
        "--encoder-model-path",
        str(paths["encoder_model_path"]),
        "--encoder-checkpoint-path",
        str(paths["encoder_checkpoint_path"]),
        "--encoder-embed-dim",
        "256",
        "--device",
        "cpu",
        "--encoder-use-lora",
        "--encoder-use-bf16",
        "--lsh-whitening-path",
        str(paths["lsh_whitening_path"]),
        "--lsh-d",
        "6",
        "--gamma",
        "0.5",
        "--semantic-margin",
        "0.125",
        "--max-group-statements",
        "3",
        "--min-scoreable-contexts",
        "4",
        "--min-proxy-windows",
        "5",
        "--target-fpr",
        "0.1",
        "--use-ordinal-keying",
        "--evidence-mode",
        "hit_only",
        "--statistic",
        "calibrated_context_mean_proxy_penalized",
        "--proxy-penalty-alpha",
        "0.34",
        "--code-length-adjustment-beta",
        "0.3",
        "--code-length-reference-chars",
        "700",
        "--no-structure-aware",
    ]
    if secret_key is not None:
        args.extend(["--secret-key", secret_key])
    return args


def _assert_detector_mapping(
    *,
    load_scorer: MagicMock,
    pipeline_cls: MagicMock,
    scorer: object,
    detector_paths: dict[str, Path],
    expected_secret_key: str = "1010",
) -> None:
    pipeline_kwargs = pipeline_cls.call_args.kwargs
    config = pipeline_kwargs["config"]
    assert pipeline_kwargs == {"config": config, "scorer": scorer}
    assert load_scorer.call_args.kwargs == {
        "config": config,
        "encoder_model_path": str(detector_paths["encoder_model_path"]),
        "encoder_checkpoint_path": str(detector_paths["encoder_checkpoint_path"]),
        "embed_dim": 256,
        "device": "cpu",
        "use_lora": True,
        "use_bf16": True,
        "whitening_path": str(detector_paths["lsh_whitening_path"]),
    }
    assert config.secret_key == expected_secret_key
    assert config.lsh_d == 6
    assert config.gamma == 0.5
    assert config.semantic_margin == 0.125
    assert config.max_group_statements == 3
    assert config.min_scoreable_contexts == 4
    assert config.min_proxy_windows == 5
    assert config.target_fpr == 0.1
    assert config.use_ordinal_keying is True
    assert config.evidence_mode == "hit_only"
    assert config.statistic == "calibrated_context_mean_proxy_penalized"
    assert config.proxy_penalty_alpha == 0.34
    assert config.code_length_adjustment_beta == 0.3
    assert config.code_length_reference_chars == 700
    assert config.structure_aware is False


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _final_code_row(sample_id: str, final_code: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "dataset": "humaneval",
        "prompt": "def target():\n",
        "final_code": final_code,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_split_rows(output_dir: Path) -> dict[str, list[dict[str, object]]]:
    return {
        split_name: _read_jsonl(output_dir / f"{split_name}.jsonl")
        for split_name in ("dev", "calibration", "test")
    }


def _row_counter(records) -> Counter[str]:
    return Counter(json.dumps(record, sort_keys=True) for record in records)
