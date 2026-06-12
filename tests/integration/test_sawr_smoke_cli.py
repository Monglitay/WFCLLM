from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.run_sawr_smoke as sawr_cli
from wfcllm.sawr.rules import HashEmbeddingRule


def test_run_sawr_smoke_cli_builds_pipeline_config(
    tmp_path: Path,
    capsys,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_path = tmp_path / "sawr" / "humaneval_sawr_final_20260611_010101.jsonl"
    pipeline = MagicMock()
    pipeline.run.return_value = str(output_path)

    with (
        patch("scripts.run_sawr_smoke.SawrGenerator") as generator_cls,
        patch("scripts.run_sawr_smoke.SawrPipeline", return_value=pipeline) as pipeline_cls,
    ):
        rc = sawr_cli.main(
            [
                "--dataset",
                "humaneval",
                "--dataset-path",
                "data/datasets",
                "--model-path",
                str(model_path),
                "--output-dir",
                str(tmp_path / "sawr"),
                "--sample-limit",
                "10",
                "--sample-offset",
                "2",
                "--max-new-tokens",
                "64",
                "--temperature",
                "0.0",
                "--top-p",
                "1.0",
                "--top-k",
                "0",
                "--torch-dtype",
                "bf16",
                "--device",
                "cpu",
                "--seed",
                "9",
                "--load-in-4bit",
                "--eos-token-id",
                "2",
                "--max-group-statements",
                "2",
                "--retry-budget",
                "1",
                "--target-accept-rate",
                "0.25",
                "--prompt-mode",
                "completion",
            ]
        )

    assert rc == 0
    generator_cls.assert_called_once()
    pipeline_cls.assert_called_once()
    pipeline.run.assert_called_once()

    generator_config = generator_cls.call_args.kwargs["config"]
    rule = generator_cls.call_args.kwargs["rule"]
    config = pipeline_cls.call_args.kwargs["config"]

    assert isinstance(rule, HashEmbeddingRule)
    assert config.dataset == "humaneval"
    assert config.dataset_path == "data/datasets"
    assert config.output_dir == str(tmp_path / "sawr")
    assert config.sample_limit == 10
    assert config.sample_offset == 2
    assert config.max_group_statements == 2
    assert config.retry_budget == 1
    assert config.rule.target_accept_rate == 0.25
    assert config.generation is generator_config
    assert config.generation.model_path == str(model_path)
    assert config.generation.max_new_tokens == 64
    assert config.generation.temperature == 0.0
    assert config.generation.top_p == 1.0
    assert config.generation.top_k == 0
    assert config.generation.torch_dtype == "bf16"
    assert config.generation.device == "cpu"
    assert config.generation.seed == 9
    assert config.generation.load_in_4bit is True
    assert config.generation.eos_token_id == 2
    assert config.generation.prompt_mode == "completion"
    assert "\nprint" in config.generation.stop_sequences

    captured = capsys.readouterr()
    assert str(output_path) in captured.err


def test_run_sawr_smoke_cli_returns_one_for_invalid_config(
    tmp_path: Path,
    capsys,
) -> None:
    rc = sawr_cli.main(
        [
            "--dataset",
            "humaneval",
            "--dataset-path",
            "data/datasets",
            "--model-path",
            str(tmp_path / "missing-model"),
        ]
    )

    assert rc == 1
    assert "model_path does not exist" in capsys.readouterr().err


def test_run_sawr_smoke_cli_passes_resume_latest(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    pipeline = MagicMock()
    pipeline.run.return_value = str(
        tmp_path / "sawr" / "humaneval_sawr_final_20260611_010101.jsonl"
    )

    with (
        patch("scripts.run_sawr_smoke.SawrGenerator"),
        patch("scripts.run_sawr_smoke.SawrPipeline", return_value=pipeline) as pipeline_cls,
    ):
        rc = sawr_cli.main(
            [
                "--dataset",
                "humaneval",
                "--dataset-path",
                "data/datasets",
                "--model-path",
                str(model_path),
                "--resume",
                "latest",
            ]
        )

    assert rc == 0
    assert pipeline_cls.call_args.kwargs["config"].resume == "latest"
