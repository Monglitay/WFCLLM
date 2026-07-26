"""CLI override + JSON config merge helpers.

Lifted verbatim from run.py:63-296 (Phase 1 refactor). Pure functions; no behavior change.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path


def parse_optional_bool(raw_value: str) -> bool:
    """Parse a CLI boolean value using explicit true/false strings."""

    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: true, false, 1, 0, yes, no, on, off")


def load_config(config_path: Path) -> dict:
    """读取 JSON 配置文件，返回按阶段分组的 dict。文件不存在时返回空 dict。"""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        import sys
        print(f"[错误] 配置文件解析失败：{e}", file=sys.stderr)
        sys.exit(1)


def _strip_removed_gate_validation_keys(config: dict) -> dict:
    """Drop config keys retired with the gate-validate stage (ADR 0007).

    Historical gated configs may still carry these keys; they are ignored
    instead of rejected so third-party configs keep resolving.
    """

    config.pop("gate_validate", None)
    method = config.get("method")
    if isinstance(method, Mapping):
        method = dict(method)
        gate = method.get("gate")
        if isinstance(gate, Mapping):
            gate = dict(gate)
            gate.pop("require_validated", None)
            method["gate"] = gate
        config["method"] = method
    experiment = config.get("experiment")
    if isinstance(experiment, Mapping):
        experiment = dict(experiment)
        experiment.pop("allow_unvalidated_gate_candidate", None)
        config["experiment"] = experiment
    return config


def resolve_method_config(config: Mapping[str, object]) -> dict:
    """Expand a named public method config against its canonical preset."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    method = config.get("method")
    name = method.get("name") if isinstance(method, Mapping) else None
    if not isinstance(name, str):
        return deepcopy(dict(config))

    from wfcllm.method.presets import (
        EVIDENCE_RETRY_SEED7X3_NAME,
        GATED_SEMANTIC_WINDOW_V1_NAME,
        load_method_preset,
    )

    if name not in {EVIDENCE_RETRY_SEED7X3_NAME, GATED_SEMANTIC_WINDOW_V1_NAME}:
        return deepcopy(dict(config))
    if name == GATED_SEMANTIC_WINDOW_V1_NAME:
        config = _strip_removed_gate_validation_keys(deepcopy(dict(config)))
    merged = _deep_merge(load_method_preset(name).to_dict(), dict(config))
    experiment = config.get("experiment")
    if name == GATED_SEMANTIC_WINDOW_V1_NAME and isinstance(
        experiment, Mapping
    ):
        return _resolve_experiment_matrix_config(config, merged)
    from wfcllm.method.config import WFCLLMMethodPreset

    return WFCLLMMethodPreset(**merged).to_dict()


def _resolve_experiment_matrix_config(
    raw_config: Mapping[str, object],
    merged: dict,
) -> dict:
    """Resolve one preflight-validated no-carrier matrix overlay."""
    from wfcllm.cli.experiment_preflight import validate_experiment_config

    generation = raw_config.get("generation")
    experiment = raw_config.get("experiment")
    if not isinstance(generation, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("experiment matrix config is incomplete")
    language = generation.get("language")
    dataset = generation.get("dataset")
    profile = experiment.get("profile")
    if not all(isinstance(value, str) for value in (language, dataset, profile)):
        raise ValueError("experiment matrix identity must contain strings")
    validate_experiment_config(raw_config, language, dataset, profile)

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
    merged["gate_data"]["scale"] = "pilot" if profile == "fast" else "full"
    # Mirror the non-matrix semantics of wfcllm.method.config: a hash-bound
    # external gate bundle starts at the four main phases, everything else
    # runs the full local seven-phase chain.
    if method.get("gate", {}).get("bundle_path") is not None:
        merged["runtime"]["default_phases"] = [
            "generate",
            "calibrate",
            "detect",
            "report",
        ]
    else:
        merged["runtime"]["default_phases"] = [
            "encoder",
            "gate-data",
            "gate-train",
            "generate",
            "calibrate",
            "detect",
            "report",
        ]
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), dict(value))
        else:
            result[key] = deepcopy(value)
    return result


def resolve_extract_lsh_params(first_record: dict, ext_cfg: dict) -> tuple[int, float]:
    params = first_record.get("watermark_params") or {}
    lsh_d_raw = params.get("lsh_d", ext_cfg.get("lsh_d", 3))
    lsh_gamma_raw = params.get("lsh_gamma", ext_cfg.get("lsh_gamma", 0.5))
    try:
        return int(lsh_d_raw), float(lsh_gamma_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LSH 参数解析失败：lsh_d={lsh_d_raw!r}, lsh_gamma={lsh_gamma_raw!r}"
        ) from exc


def resolve_adaptive_gamma_config(args: argparse.Namespace, wm_cfg: dict):
    raise RuntimeError(
        "adaptive-gamma config resolution belongs to the archived legacy "
        "WFCLLM pipeline; see archive/legacy_wfcllm_2026_07/."
    )


def resolve_extract_adaptive_gamma_config(args: argparse.Namespace, cfg: dict):
    raise RuntimeError(
        "legacy extract adaptive-gamma config resolution has been archived; "
        "see archive/legacy_wfcllm_2026_07/."
    )


def resolve_token_channel_config(
    section: dict | None,
    args: argparse.Namespace | None = None,
):
    raise RuntimeError(
        "token-channel config resolution belongs to the archived legacy "
        "WFCLLM pipeline; see archive/legacy_wfcllm_2026_07/."
    )


def _apply_token_channel_cli_overrides(
    configured: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    merged: dict[str, object] = dict(configured)
    raw_joint_section = merged.get("joint")
    if raw_joint_section is None:
        joint_section: dict[str, object] = {}
    elif isinstance(raw_joint_section, dict):
        joint_section = dict(raw_joint_section)
    else:
        raise ValueError("joint must be a JSON object")

    scalar_overrides = {
        "enabled": getattr(args, "token_channel_enabled", None),
        "channel_mode": getattr(args, "token_channel_mode", None),
        "model_path": getattr(args, "token_channel_model_path", None),
        "context_width": getattr(args, "token_channel_context_width", None),
        "switch_threshold": getattr(args, "token_channel_switch_threshold", None),
        "delta": getattr(args, "token_channel_delta", None),
        "ignore_repeated_ngrams": getattr(args, "token_channel_ignore_repeated_ngrams", None),
        "ignore_repeated_prefixes": getattr(args, "token_channel_ignore_repeated_prefixes", None),
        "debug_mode": getattr(args, "token_channel_debug_mode", None),
        "lexical_min_block_tokens": getattr(args, "token_channel_lexical_min_block_tokens", None),
        "lexical_retry_decay_start": getattr(args, "token_channel_lexical_retry_decay_start", None),
        "lexical_retry_disable_after": getattr(args, "token_channel_lexical_retry_disable_after", None),
        "lexical_gate_probe_tokens": getattr(args, "token_channel_lexical_gate_probe_tokens", None),
        "lexical_gate_min_fraction": getattr(args, "token_channel_lexical_gate_min_fraction", None),
        "joint_semantic_weight": getattr(args, "token_channel_joint_semantic_weight", None),
        "joint_lexical_weight": getattr(args, "token_channel_joint_lexical_weight", None),
        "lexical_full_weight_min_positions": getattr(
            args,
            "token_channel_lexical_full_weight_min_positions",
            None,
        ),
        "joint_threshold": getattr(args, "token_channel_joint_threshold", None),
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            merged[key] = value

    joint_overrides = {
        "semantic_weight": getattr(args, "token_channel_joint_semantic_weight", None),
        "lexical_weight": getattr(args, "token_channel_joint_lexical_weight", None),
        "lexical_full_weight_min_positions": getattr(
            args,
            "token_channel_lexical_full_weight_min_positions",
            None,
        ),
        "threshold": getattr(args, "token_channel_joint_threshold", None),
    }
    for key, value in joint_overrides.items():
        if value is not None:
            joint_section[key] = value
    if joint_section:
        merged["joint"] = joint_section

    return merged


def build_extract_calibration_contract_builder(
    adaptive_detection_config,
    adaptive_gamma_config,
    lsh_d: int,
):
    raise RuntimeError(
        "legacy extract calibration contract building has been archived; "
        "see archive/legacy_wfcllm_2026_07/."
    )


def resolve_adaptive_detection_config(args: argparse.Namespace, ext_cfg: dict):
    raise RuntimeError(
        "legacy adaptive detection config resolution has been archived; "
        "see archive/legacy_wfcllm_2026_07/."
    )
