"""Load and resolve the single current Gate configuration family."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    """Load one required current-schema JSON configuration."""

    if not isinstance(config_path, Path):
        raise ValueError("config_path must be a pathlib.Path")
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file is missing: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"configuration file is invalid JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    return payload


def resolve_method_config(config: Mapping[str, object]) -> dict:
    """Expand a named public method config against its canonical preset."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    method = config.get("method")
    name = method.get("name") if isinstance(method, Mapping) else None
    if not isinstance(name, str):
        return deepcopy(dict(config))

    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset

    if name != GATED_SEMANTIC_WINDOW_V1_NAME:
        raise ValueError(f"unsupported method.name: {name!r}")
    merged = _deep_merge(load_method_preset(name).to_dict(), dict(config))
    experiment = config.get("experiment")
    if isinstance(experiment, Mapping):
        return _resolve_experiment_matrix_config(config, merged)
    raise ValueError("gated config must declare one full experiment profile")


def _resolve_experiment_matrix_config(
    raw_config: Mapping[str, object],
    merged: dict,
) -> dict:
    """Resolve one preflight-validated Full Reproduction Profile overlay."""
    from wfcllm.cli.experiment_preflight import (
        validate_experiment_config,
        validate_public_full_config_overlay,
    )

    generation = raw_config.get("generation")
    experiment = raw_config.get("experiment")
    if not isinstance(generation, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("experiment matrix config is incomplete")
    language = generation.get("language")
    dataset = generation.get("dataset")
    profile = experiment.get("profile")
    if not all(isinstance(value, str) for value in (language, dataset, profile)):
        raise ValueError("experiment matrix identity must contain strings")
    validate_public_full_config_overlay(
        raw_config, language, dataset, profile
    )

    semantic_lsh = merged["semantic_lsh"]
    method = merged["method"]
    method["semantic"]["lsh"]["d"] = semantic_lsh["lsh_d"]
    method["semantic"]["lsh"]["gamma"] = semantic_lsh["lsh_gamma"]
    method["semantic"]["lsh"]["margin"] = semantic_lsh.get(
        "semantic_margin", 0.0
    )
    method["semantic"].setdefault(
        "preservation",
        {"rule": "codet5-cosine-to-original/v1", "threshold": 0.9},
    )
    source_family = {
        "python": "oss_python",
        "cpp": "oss_cpp",
        "java": "oss_java",
        "js": "oss_js",
    }[language]
    merged["gate_data"]["sources"] = [
        "main_generation",
        source_family,
        "parser_boundary",
    ]
    merged["gate_data"]["scale"] = "full"
    merged["runtime"]["default_phases"] = [
        "encoder",
        "gate-data",
        "gate-train",
        "generate",
        "calibrate",
        "detect",
        "report",
        "audit",
    ]
    validate_experiment_config(merged, language, dataset, profile)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), dict(value))
        else:
            result[key] = deepcopy(value)
    return result
