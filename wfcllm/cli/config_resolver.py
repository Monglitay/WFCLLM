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
