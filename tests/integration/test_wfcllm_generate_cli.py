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


def test_wfcllm_generate_cli_defaults_match_official_preset(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    parser = generate_cli._build_parser()
    args = parser.parse_args(["--model-path", str(model_path)])

    assert args.max_new_tokens == 256
    assert args.temperature == 0.25
    assert args.top_p == 0.95
    assert args.top_k == 0
    assert args.retry_repetition_penalty == 4.0
    assert args.torch_dtype == "bf16"
    assert args.device == "cuda"
    assert args.seed == 7
    assert args.load_in_4bit is True
    assert args.prompt_mode == "completion"
    assert args.max_group_statements == 2
    assert args.retry_budget == 2
    assert args.statement_retry_budget == 4
    assert args.window_retry_budget == 3
    assert args.compound_retry_budget == 2
    assert args.global_rollback_budget == 180
    assert args.max_total_sampled_tokens == 32768
    assert args.evidence_retry_attempts == 3
    assert args.evidence_retry_seed_stride == 101
    assert args.rule_name == "semantic_lsh"
    assert args.lsh_d == 4
    assert args.lsh_gamma == 0.25
    assert args.semantic_margin == 0.0
