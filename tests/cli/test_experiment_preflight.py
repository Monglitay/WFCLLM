from __future__ import annotations

import pytest

from wfcllm.cli.experiment_preflight import (
    validate_experiment_config,
    validate_runtime_capabilities,
)


def _config(*, language: str = "cpp", dataset: str = "humanevalpack") -> dict:
    return {
        "method": {
            "name": "gated_semantic_window_v1",
            "gate": {"require_validated": True},
            "rewrite": {"strategy": "model_semantic_window"},
        },
        "generation": {"language": language, "dataset": dataset},
        "semantic_lsh": {"rule_name": "semantic_lsh"},
        "experiment": {"profile": "full"},
    }


@pytest.mark.parametrize(
    ("language", "dataset"),
    [
        ("python", "humaneval"),
        ("python", "mbpp"),
        ("cpp", "humanevalpack"),
        ("java", "humanevalpack"),
    ],
)
def test_preflight_accepts_supported_pairs(language: str, dataset: str) -> None:
    config = _config(language=language, dataset=dataset)

    validate_experiment_config(config, language, dataset, "full")


def test_preflight_allows_explicit_unvalidated_gate_candidate_diagnostic() -> None:
    config = _config(language="python", dataset="humaneval")
    config["method"]["rewrite"]["strategy"] = "python_ast_equivalent"
    config["method"]["gate"]["require_validated"] = False
    config["experiment"]["allow_unvalidated_gate_candidate"] = True

    validate_experiment_config(config, "python", "humaneval", "full")


def test_preflight_rejects_unsupported_pair() -> None:
    config = _config(language="cpp", dataset="mbpp")

    with pytest.raises(ValueError, match="unsupported language/dataset pair"):
        validate_experiment_config(config, "cpp", "mbpp", "full")


def test_preflight_rejects_identity_mismatch() -> None:
    config = _config(language="java")

    with pytest.raises(ValueError, match="generation.language"):
        validate_experiment_config(config, "cpp", "humanevalpack", "full")


def test_preflight_rejects_carrier_rule() -> None:
    config = _config()
    config["semantic_lsh"]["rule_name"] = "keyed_text_region"

    with pytest.raises(ValueError, match="carrier"):
        validate_experiment_config(config, "cpp", "humanevalpack", "full")


def test_python_runtime_capability_is_available() -> None:
    config = _config(language="python", dataset="humaneval")
    config["method"]["rewrite"]["strategy"] = "python_ast_equivalent"

    validate_runtime_capabilities(config)


@pytest.mark.parametrize("language", ["cpp", "java"])
def test_multilanguage_runtime_requires_model_semantic_rewriter(language: str) -> None:
    config = _config(language=language)

    validate_runtime_capabilities(config)

    config["method"]["rewrite"]["strategy"] = "python_ast_equivalent"
    with pytest.raises(ValueError, match="model_semantic_window"):
        validate_runtime_capabilities(config)
