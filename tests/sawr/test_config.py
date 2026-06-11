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


def test_rule_config_rejects_non_json_serializable_parameters():
    with pytest.raises(ValueError, match="parameters must be JSON-serializable"):
        SawrRuleConfig(parameters={"x": object()})


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
