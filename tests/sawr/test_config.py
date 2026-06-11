from __future__ import annotations

import pytest

from wfcllm.sawr.config import (
    SawrGenerationConfig,
    SawrPipelineConfig,
    SawrRuleConfig,
)


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
    assert config.torch_dtype == "bf16"
    assert config.device == "cpu"
    assert config.seed == 123
    assert config.load_in_4bit is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_new_tokens", 0, "max_new_tokens must be positive"),
        ("temperature", -0.1, "temperature must be non-negative"),
        ("top_p", 1.5, r"top_p must be in \(0, 1\]"),
        ("top_k", -1, "top_k must be non-negative"),
        ("torch_dtype", "float64", "torch_dtype must be one of"),
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
        ({"rule_name": "semantic_lsh"}, "rule_name must be 'hash'"),
    ],
)
def test_rule_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SawrRuleConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dataset": "apps"}, "dataset must be one of"),
        ({"max_group_statements": 0}, "max_group_statements must be positive"),
        ({"retry_budget": -1}, "retry_budget must be non-negative"),
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
    assert "lsh_d" not in config_dict
    assert "lsh_gamma" not in config_dict
    assert "fpr_threshold" not in config_dict
    assert "token_channel" not in config_dict
    assert "calibration" not in config_dict
