from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.wfcllm_generate as generate_cli
from wfcllm.semantic.rules import HashEmbeddingRule


def test_wfcllm_generate_cli_builds_pipeline_config(
    tmp_path: Path,
    capsys,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_path = tmp_path / "run" / "inputs" / "final_code.jsonl"
    pipeline = MagicMock()
    pipeline.run.return_value = str(output_path)

    with (
        patch("scripts.wfcllm_generate.WFCLLMGenerator") as generator_cls,
        patch(
            "scripts.wfcllm_generate.WFCLLMGenerationPipeline",
            return_value=pipeline,
        ) as pipeline_cls,
    ):
        rc = generate_cli.main(
            [
                "--dataset",
                "humaneval",
                "--dataset-path",
                "data/datasets",
                "--model-path",
                str(model_path),
                "--output-dir",
                str(tmp_path / "run"),
                "--evidence-retry-attempts",
                "3",
                "--evidence-retry-seed-stride",
                "101",
                "--rule-name",
                "hash",
            ]
        )

    assert rc == 0
    generator_cls.assert_called_once()
    pipeline_cls.assert_called_once()
    pipeline.run.assert_called_once()

    rule = generator_cls.call_args.kwargs["rule"]
    config = pipeline_cls.call_args.kwargs["config"]
    assert isinstance(rule, HashEmbeddingRule)
    assert config.dataset == "humaneval"
    assert config.dataset_path == "data/datasets"
    assert config.output_dir == str(tmp_path / "run")
    assert config.evidence_retry_attempts == 3
    assert config.evidence_retry_seed_stride == 101
    assert config.rule.rule_name == "hash"
    assert config.generation is generator_cls.call_args.kwargs["config"]
    assert config.generation.model_path == str(model_path)
    assert f"[完成] WFCLLM final-code rows saved to {output_path}" in (
        capsys.readouterr().err
    )
