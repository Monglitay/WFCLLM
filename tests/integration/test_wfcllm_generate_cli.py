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


def test_wfcllm_generate_cli_uses_env_secret_key_for_semantic_rule(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_path = tmp_path / "run" / "inputs" / "final_code.jsonl"
    semantic_rule = object()
    pipeline = MagicMock()
    pipeline.run.return_value = str(output_path)
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "env-secret-key")

    with (
        patch(
            "scripts.wfcllm_generate.load_semantic_lsh_rule",
            return_value=semantic_rule,
        ) as load_rule,
        patch("scripts.wfcllm_generate.WFCLLMGenerator") as generator_cls,
        patch(
            "scripts.wfcllm_generate.WFCLLMGenerationPipeline",
            return_value=pipeline,
        ),
    ):
        rc = generate_cli.main(["--model-path", str(model_path)])

    assert rc == 0
    assert load_rule.call_args.kwargs["secret_key"] == "env-secret-key"
    assert generator_cls.call_args.kwargs["rule"] is semantic_rule


def test_wfcllm_generate_cli_secret_key_arg_takes_precedence_over_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_path = tmp_path / "run" / "inputs" / "final_code.jsonl"
    semantic_rule = object()
    pipeline = MagicMock()
    pipeline.run.return_value = str(output_path)
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "env-secret-key")

    with (
        patch(
            "scripts.wfcllm_generate.load_semantic_lsh_rule",
            return_value=semantic_rule,
        ) as load_rule,
        patch("scripts.wfcllm_generate.WFCLLMGenerator"),
        patch(
            "scripts.wfcllm_generate.WFCLLMGenerationPipeline",
            return_value=pipeline,
        ),
    ):
        rc = generate_cli.main(
            ["--model-path", str(model_path), "--secret-key", "cli-secret-key"]
        )

    assert rc == 0
    assert load_rule.call_args.kwargs["secret_key"] == "cli-secret-key"


def test_wfcllm_generate_cli_builds_v2_retry20_selector(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_path = tmp_path / "run" / "final.jsonl"
    ledger_path = tmp_path / "run" / "retry_ledger.jsonl"
    semantic_rule = object()
    v2_scorer = object()
    pipeline = MagicMock()
    pipeline.run.return_value = str(output_path)
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "env-secret-key")

    with (
        patch(
            "scripts.wfcllm_generate.load_semantic_lsh_rule",
            return_value=semantic_rule,
        ),
        patch(
            "scripts.wfcllm_generate.load_v2_signature_scorer",
            return_value=v2_scorer,
        ) as load_v2_scorer,
        patch("scripts.wfcllm_generate.V2RetryAttemptSelector") as selector_cls,
        patch("scripts.wfcllm_generate.WFCLLMGenerator"),
        patch(
            "scripts.wfcllm_generate.WFCLLMGenerationPipeline",
            return_value=pipeline,
        ) as pipeline_cls,
    ):
        rc = generate_cli.main(
            [
                "--model-path",
                str(model_path),
                "--method-version",
                "v2",
                "--evidence-retry-attempts",
                "20",
                "--retry-attempt-ledger-output",
                str(ledger_path),
                "--sample-id",
                "HumanEval/2",
                "--sample-id",
                "HumanEval/0",
                "--v2-signature-bits",
                "16",
                "--v2-aggregation",
                "standardized_bit_sum",
            ]
        )

    assert rc == 0
    load_v2_scorer.assert_called_once()
    assert load_v2_scorer.call_args.kwargs["secret_key"] == "env-secret-key"
    assert load_v2_scorer.call_args.kwargs["signature_bits"] == 16
    assert (
        load_v2_scorer.call_args.kwargs["aggregation"] == "standardized_bit_sum"
    )
    selector_cls.assert_called_once_with(scorer=v2_scorer)
    config = pipeline_cls.call_args.kwargs["config"]
    assert config.method_version == "v2"
    assert config.evidence_retry_attempts == 20
    assert config.sample_ids == ("HumanEval/2", "HumanEval/0")
    assert config.retry_attempt_ledger_output == str(ledger_path)
    assert pipeline_cls.call_args.kwargs["retry_selector"] is selector_cls.return_value


def test_wfcllm_generate_cli_rejects_v2_retry_other_than_twenty(
    tmp_path: Path,
    capsys,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    rc = generate_cli.main(
        [
            "--model-path",
            str(model_path),
            "--method-version",
            "v2",
            "--evidence-retry-attempts",
            "19",
            "--retry-attempt-ledger-output",
            str(tmp_path / "ledger.jsonl"),
        ]
    )

    assert rc == 1
    assert "exactly 20" in capsys.readouterr().err


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
