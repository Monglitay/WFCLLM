"""CLI override + JSON config merge helpers.

Lifted verbatim from run.py:63-296 (Phase 1 refactor). Pure functions; no behavior change.
"""
from __future__ import annotations

import argparse
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
    from wfcllm.watermark.config import AdaptiveGammaConfig

    configured = wm_cfg.get("adaptive_gamma") or {}
    defaults = AdaptiveGammaConfig()
    anchors = defaults.anchors.copy()
    raw_anchors = configured.get("anchors")
    if isinstance(raw_anchors, dict):
        anchors.update(raw_anchors)

    enabled = bool(configured.get("enabled", defaults.enabled))
    if (
        getattr(args, "gamma_strategy", None) is not None
        or getattr(args, "entropy_profile", None) is not None
        or getattr(args, "profile_id", None) is not None
    ):
        enabled = True

    return AdaptiveGammaConfig(
        enabled=enabled,
        strategy=(
            getattr(args, "gamma_strategy", None)
            or configured.get("strategy", defaults.strategy)
        ),
        profile_path=(
            getattr(args, "entropy_profile", None)
            if getattr(args, "entropy_profile", None) is not None
            else configured.get("profile_path", defaults.profile_path)
        ),
        profile_id=(
            getattr(args, "profile_id", None)
            if getattr(args, "profile_id", None) is not None
            else configured.get("profile_id", defaults.profile_id)
        ),
        gamma_min=float(configured.get("gamma_min", defaults.gamma_min)),
        gamma_max=float(configured.get("gamma_max", defaults.gamma_max)),
        anchors=anchors,
    )


def resolve_extract_adaptive_gamma_config(args: argparse.Namespace, cfg: dict):
    extract_cfg = cfg.get("extract", {})
    configured = extract_cfg.get("adaptive_gamma")
    if isinstance(configured, dict):
        return resolve_adaptive_gamma_config(
            args,
            {"adaptive_gamma": configured},
        )
    return resolve_adaptive_gamma_config(args, cfg.get("watermark", {}))


def resolve_token_channel_config(
    section: dict | None,
    args: argparse.Namespace | None = None,
):
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig

    if section is None:
        configured = {}
    elif isinstance(section, dict):
        configured = section
    else:
        raise ValueError("token_channel must be a JSON object")

    if args is not None:
        configured = _apply_token_channel_cli_overrides(configured, args)

    return TokenChannelConfig.from_mapping(configured)


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
    if not getattr(adaptive_detection_config, "prefer_adaptive", False):
        return None
    if not getattr(adaptive_gamma_config, "enabled", False):
        return None

    from wfcllm.extract.alignment import rebuild_block_contracts

    def builder(code: str) -> dict[str, dict]:
        return {
            contract["block_id"]: contract
            for contract in rebuild_block_contracts(
                code,
                adaptive_gamma_config=adaptive_gamma_config,
                default_lsh_d=lsh_d,
            )
        }

    return builder


def resolve_adaptive_detection_config(args: argparse.Namespace, ext_cfg: dict):
    from wfcllm.extract.config import AdaptiveDetectionConfig

    configured = ext_cfg.get("adaptive_detection") or {}
    defaults = AdaptiveDetectionConfig()

    require_block_contract_check = bool(
        configured.get(
            "require_block_contract_check",
            defaults.require_block_contract_check,
        )
    )
    fail_on_structure_mismatch = bool(
        configured.get(
            "fail_on_structure_mismatch",
            defaults.fail_on_structure_mismatch,
        )
    )
    if getattr(args, "strict_contract", False):
        require_block_contract_check = True
        fail_on_structure_mismatch = True

    return AdaptiveDetectionConfig(
        mode=(
            getattr(args, "adaptive_detection_mode", None)
            or configured.get("mode", defaults.mode)
        ),
        require_block_contract_check=require_block_contract_check,
        fail_on_structure_mismatch=fail_on_structure_mismatch,
        warn_on_numeric_mismatch=bool(
            configured.get(
                "warn_on_numeric_mismatch",
                defaults.warn_on_numeric_mismatch,
            )
        ),
        exclude_invalid_samples=bool(
            configured.get(
                "exclude_invalid_samples",
                defaults.exclude_invalid_samples,
            )
        ),
    )
