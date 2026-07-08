from __future__ import annotations

import json
import subprocess
import sys

import pytest

from wfcllm.sawr.config import (
    SawrGenerationConfig,
    SawrPipelineConfig,
    SawrRuleConfig,
)


def test_import_sawr_does_not_import_huggingface_datasets():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import wfcllm.sawr; "
                "raise SystemExit(1 if 'datasets' in sys.modules else 0)"
            ),
        ],
        check=False,
    )

    assert result.returncode == 0


def test_generation_config_requires_existing_model_path(tmp_path):
    missing_path = tmp_path / "missing-model"

    with pytest.raises(ValueError, match="model_path does not exist"):
        SawrGenerationConfig(model_path=str(missing_path))


def test_generation_config_accepts_local_model_path(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()

    config = SawrGenerationConfig(
        model_path=str(model_path),
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        retry_repetition_penalty=1.25,
        torch_dtype="bf16",
        device="cpu",
        seed=123,
        load_in_4bit=False,
    )

    assert config.model_path == str(model_path)
    assert config.max_new_tokens == 64
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.top_k == 0
    assert config.retry_repetition_penalty == 1.25
    assert config.torch_dtype == "bf16"
    assert config.device == "cpu"
    assert config.seed == 123
    assert config.load_in_4bit is False
    assert config.prompt_mode == "completion"
    assert "\nprint" in config.stop_sequences


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_new_tokens", 0, "max_new_tokens must be positive"),
        ("temperature", -0.1, "temperature must be non-negative"),
        ("top_p", 1.5, r"top_p must be in \(0, 1\]"),
        ("top_k", -1, "top_k must be non-negative"),
        (
            "retry_repetition_penalty",
            0.99,
            "retry_repetition_penalty must be >= 1.0",
        ),
        (
            "retry_repetition_penalty",
            float("nan"),
            "retry_repetition_penalty must be a finite number",
        ),
        (
            "retry_repetition_penalty",
            True,
            "retry_repetition_penalty must be a finite number",
        ),
        ("torch_dtype", "float64", "torch_dtype must be one of"),
        ("prompt_mode", "assistant", "prompt_mode must be one of"),
        ("stop_sequences", "\nprint", "stop_sequences must be a sequence"),
        ("stop_sequences", ("",), "stop_sequences entries must be non-empty strings"),
    ],
)
def test_generation_config_rejects_invalid_sampling_values(
    tmp_path,
    field,
    value,
    message,
):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    kwargs = {"model_path": str(model_path), field: value}

    with pytest.raises(ValueError, match=message):
        SawrGenerationConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_accept_rate": 1.1}, r"target_accept_rate must be in \[0, 1\]"),
        ({"rule_name": "unknown"}, "rule_name must be one of"),
    ],
)
def test_rule_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SawrRuleConfig(**kwargs)


def test_rule_config_accepts_semantic_lsh_parameters():
    config = SawrRuleConfig(
        rule_name="semantic_lsh",
        parameters={
            "secret_key": "1010",
            "encoder_model_path": "data/models/codet5-base",
            "encoder_embed_dim": 128,
            "lsh_d": 4,
            "lsh_gamma": 0.75,
            "margin": 0.0,
        },
    )

    assert config.rule_name == "semantic_lsh"
    assert config.parameters["lsh_d"] == 4


def test_rule_config_rejects_non_json_serializable_parameters():
    with pytest.raises(ValueError, match="parameters must be JSON-serializable"):
        SawrRuleConfig(parameters={"x": object()})


def test_rule_config_rejects_non_dict_parameters():
    with pytest.raises(ValueError, match="parameters must be a dict"):
        SawrRuleConfig(parameters=[1, 2])


def test_rule_config_copies_parameters_before_caller_mutation():
    parameters = {"x": {"enabled": True}}
    config = SawrRuleConfig(parameters=parameters)

    parameters["x"]["enabled"] = False
    parameters["new"] = "value"

    assert config.to_dict()["parameters"] == {"x": {"enabled": True}}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dataset": "apps"}, "dataset must be one of"),
        ({"sample_limit": -1}, "sample_limit must be non-negative"),
        ({"sample_offset": -1}, "sample_offset must be non-negative"),
        ({"max_group_statements": 0}, "max_group_statements must be positive"),
        ({"retry_budget": -1}, "retry_budget must be non-negative"),
        ({"statement_retry_budget": -1}, "statement_retry_budget must be non-negative"),
        ({"window_retry_budget": -1}, "window_retry_budget must be non-negative"),
        ({"compound_retry_budget": -1}, "compound_retry_budget must be non-negative"),
        ({"global_rollback_budget": -1}, "global_rollback_budget must be non-negative"),
        ({"max_total_sampled_tokens": 0}, "max_total_sampled_tokens must be positive"),
        ({"evidence_retry_attempts": 0}, "evidence_retry_attempts must be positive"),
        (
            {"evidence_retry_seed_stride": 0},
            "evidence_retry_seed_stride must be positive",
        ),
        ({"resume": "checkpoint"}, "resume must be None or 'latest'"),
    ],
)
def test_pipeline_config_rejects_invalid_values(tmp_path, kwargs, message):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path))
    pipeline_kwargs = {
        "dataset": "humaneval",
        "dataset_path": str(tmp_path / "datasets"),
        "output_dir": str(tmp_path / "outputs"),
        "generation": generation,
    }
    pipeline_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SawrPipelineConfig(**pipeline_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"generation": {"model_path": "missing"}}, "generation must be SawrGenerationConfig"),
        ({"rule": {"target_accept_rate": 0.5}}, "rule must be SawrRuleConfig"),
    ],
)
def test_pipeline_config_rejects_raw_nested_config_dicts(tmp_path, kwargs, message):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    pipeline_kwargs = {
        "dataset": "humaneval",
        "dataset_path": str(tmp_path / "datasets"),
        "output_dir": str(tmp_path / "outputs"),
        "generation": SawrGenerationConfig(model_path=str(model_path)),
    }
    pipeline_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SawrPipelineConfig(**pipeline_kwargs)


@pytest.mark.parametrize("dataset", ["humaneval", "mbpp"])
def test_pipeline_config_accepts_supported_dataset(tmp_path, dataset):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path))

    config = SawrPipelineConfig(
        dataset=dataset,
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        resume="latest",
    )

    assert config.dataset == dataset
    assert config.resume == "latest"


def test_pipeline_config_derives_absolute_sampled_token_budget(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path), max_new_tokens=40)

    config = SawrPipelineConfig(
        dataset="humaneval",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        retry_budget=3,
    )

    assert config.global_rollback_budget == 3
    assert config.max_total_sampled_tokens == 200


def test_pipeline_config_derives_rollback_budget_from_split_retry_budgets(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path), max_new_tokens=40)

    config = SawrPipelineConfig(
        dataset="humaneval",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        retry_budget=1,
        statement_retry_budget=15,
        window_retry_budget=10,
        compound_retry_budget=5,
    )

    assert config.global_rollback_budget == 30
    assert config.max_total_sampled_tokens == 1280


def test_pipeline_config_accepts_explicit_bounded_generation_controls(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path), max_new_tokens=40)

    config = SawrPipelineConfig(
        dataset="humaneval",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        retry_budget=3,
        global_rollback_budget=7,
        max_total_sampled_tokens=123,
    )

    assert config.global_rollback_budget == 7
    assert config.max_total_sampled_tokens == 123


def test_pipeline_config_accepts_evidence_only_retry_controls(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path), max_new_tokens=40)

    config = SawrPipelineConfig(
        dataset="humaneval",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        evidence_retry_attempts=3,
        evidence_retry_seed_stride=17,
    )

    assert config.evidence_retry_attempts == 3
    assert config.evidence_retry_seed_stride == 17


def test_pipeline_config_to_dict_contains_smoke_config_only(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path))
    rule = SawrRuleConfig(target_accept_rate=0.25)

    config = SawrPipelineConfig(
        dataset="humaneval",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        rule=rule,
    )

    config_dict = config.to_dict()

    assert config_dict["dataset"] == "humaneval"
    assert config_dict["generation"]["model_path"] == str(model_path)
    assert config_dict["rule"]["target_accept_rate"] == 0.25
    assert config_dict["evidence_retry_attempts"] == 1
    assert config_dict["evidence_retry_seed_stride"] == 1009
    assert "lsh_d" not in config_dict
    assert "lsh_gamma" not in config_dict
    assert "fpr_threshold" not in config_dict
    assert "token_channel" not in config_dict
    assert "calibration" not in config_dict


def test_pipeline_config_to_dict_is_json_serializable(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path))
    rule = SawrRuleConfig(parameters={"mode": "smoke", "weights": [1, 2, 3]})

    config = SawrPipelineConfig(
        dataset="mbpp",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        rule=rule,
        resume="latest",
    )

    json.dumps(config.to_dict())
