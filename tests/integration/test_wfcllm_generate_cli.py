from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.wfcllm_generate as generate_cli
from wfcllm.semantic.rules import HashEmbeddingRule


def test_phase_runner_dispatches_gated_pipeline_once(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from wfcllm.cli.runners import run_generate
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset
    from wfcllm.orchestration.state import RunStateManager

    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    pipeline = MagicMock()
    pipeline.run.return_value = str(tmp_path / "run" / "inputs" / "final_code.jsonl")
    args = argparse.Namespace(
        _config_cache=config,
        _gated_generation_pipeline=pipeline,
        secret_key_file=tmp_path / "deployment.key",
        secret_key_env=None,
    )
    args.secret_key_file.write_bytes(b"deployment")
    monkeypatch.setattr(
        "wfcllm.cli.runners.resolve_validated_gate_bundle",
        lambda _args: (tmp_path / "bundle", "a" * 64),
    )
    state = RunStateManager(tmp_path / "state.json")

    assert run_generate(args, state) == 0
    pipeline.run.assert_called_once_with()
    assert state.get("generate", "output_path").endswith("final_code.jsonl")


def test_phase_runner_keeps_existing_preset_off_gated_pipeline(tmp_path: Path) -> None:
    import argparse

    from wfcllm.cli.runners import run_generate
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME, load_method_preset
    from wfcllm.orchestration.state import RunStateManager

    pipeline = MagicMock()
    args = argparse.Namespace(
        _config_cache=load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict(),
        _gated_generation_pipeline=pipeline,
    )
    assert run_generate(args, RunStateManager(tmp_path / "state.json")) == 0
    pipeline.run.assert_not_called()


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
