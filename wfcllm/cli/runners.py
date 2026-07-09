"""Per-phase runner functions invoked by PhaseOrchestrator.

Lifted from run.py (Phase 1 refactor). Functions are CLI-bound:
they consume argparse.Namespace, build phase configs, invoke pipelines,
and update RunStateManager.

Future refactor phases (Phase 5: pretrain, Phase 8: generator split) will
further decompose these into their phase packages. For Phase 1 they stay
together to avoid scope creep.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from wfcllm.cli.config_resolver import load_config
from wfcllm.orchestration.state import RunStateManager

# ---------------------------------------------------------------------------
# Compare-only mode flag names
# ---------------------------------------------------------------------------

COMPARE_ONLY_REQUIRED_FLAGS = (
    "compare_summary_left",
    "compare_details_left",
    "compare_summary_right",
    "compare_details_right",
    "compare_output",
)
COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS = (
    "compare_watermarked_left",
    "compare_watermarked_right",
)

# ---------------------------------------------------------------------------
# Config helper utilities (used by multiple runners)
# ---------------------------------------------------------------------------


def get_config(args: argparse.Namespace) -> dict:
    cfg = getattr(args, "_config_cache", None)
    if cfg is None:
        cfg = load_config(args.config)
        setattr(args, "_config_cache", cfg)
    return cfg


def configured_extract_input(args: argparse.Namespace) -> str | None:
    return (get_config(args).get("extract") or {}).get("input_file")


def is_compare_only_mode(args: argparse.Namespace) -> bool:
    required_present = all(getattr(args, flag, None) for flag in COMPARE_ONLY_REQUIRED_FLAGS)
    if not required_present:
        return False

    optional_watermarked_flags = tuple(
        getattr(args, flag, None) for flag in COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    )
    return not any(optional_watermarked_flags) or all(optional_watermarked_flags)


def has_explicit_extract_input(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "input_file", None) or configured_extract_input(args))


def validate_compare_only_mode(args: argparse.Namespace) -> str | None:
    """Return an error string if compare-only flags are used incorrectly, else None."""
    compare_flags = COMPARE_ONLY_REQUIRED_FLAGS + COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    if not any(getattr(args, flag, None) for flag in compare_flags):
        return None
    if args.phase not in {"extract", "legacy-extract"}:
        return "[错误] compare-only 模式仅支持 --phase legacy-extract"
    if not is_compare_only_mode(args):
        return (
            "[错误] compare-only 模式要求提供左右 summary/details 和 compare-output；"
            "watermarked 必须两侧同时提供或同时省略"
        )
    return None


def _phase_state_key(args: argparse.Namespace, default: str) -> str:
    override = getattr(args, "_state_phase_override", None)
    return override if isinstance(override, str) else default


def _run_with_state_phase(
    args: argparse.Namespace,
    state: RunStateManager,
    phase: str,
    runner,
) -> int:
    sentinel = object()
    previous = getattr(args, "_state_phase_override", sentinel)
    setattr(args, "_state_phase_override", phase)
    try:
        return runner(args, state)
    finally:
        if previous is sentinel:
            delattr(args, "_state_phase_override")
        else:
            setattr(args, "_state_phase_override", previous)


def _watermark_output_from_state(args: argparse.Namespace, state: RunStateManager) -> str | None:
    phase = _phase_state_key(args, "extract")
    if phase == "legacy-extract":
        return state.get("legacy-watermark", "output_file") or state.get(
            "watermark",
            "output_file",
        )
    return state.get("watermark", "output_file") or state.get(
        "legacy-watermark",
        "output_file",
    )


def _legacy_phase_archived(phase: str, guidance: str) -> int:
    print(
        f"[错误] {phase} implementation has been archived; {guidance}",
        file=sys.stderr,
    )
    return 1


# ---------------------------------------------------------------------------
# Per-phase runner functions
# ---------------------------------------------------------------------------


def run_generate(args: argparse.Namespace, state: RunStateManager) -> int:
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME

    state.mark_done("generate", method=EVIDENCE_RETRY_SEED7X3_NAME)
    print("=== WFCLLM generate ===")
    return 0


def run_calibrate(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("calibrate")
    print("=== WFCLLM calibrate ===")
    return 0


def run_detect(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("detect")
    print("=== WFCLLM detect ===")
    return 0


def run_report(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("report")
    print("=== WFCLLM report ===")
    return 0


def run_audit(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("audit")
    print("=== WFCLLM audit ===")
    return 0


def run_posthoc_pass_report(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done(
        "posthoc-pass-report",
        posthoc_only=True,
        not_used_for_generation=True,
        not_used_for_retry=True,
        not_used_for_selection=True,
        not_used_for_calibration=True,
        not_used_for_detection=True,
    )
    print("=== WFCLLM posthoc pass report ===")
    return 0


def run_diagnostic_selector(args: argparse.Namespace, state: RunStateManager) -> int:
    if not getattr(args, "diagnostic_only", False):
        print("[错误] diagnostic-selector requires --diagnostic-only", file=sys.stderr)
        return 1
    state.mark_done(
        "diagnostic-selector",
        diagnostic_only=True,
        not_official_method=True,
    )
    print("=== WFCLLM diagnostic selector ===")
    return 0


def run_legacy_ablation(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-ablation",
        "historical code is under archive/legacy_wfcllm_2026_07/code/ablation "
        "and scripts/legacy/run_ablation.py is retained as guidance only.",
    )


def run_legacy_watermark(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-watermark",
        "historical code is under archive/legacy_wfcllm_2026_07/code/watermark.",
    )


def run_legacy_extract(args: argparse.Namespace, state: RunStateManager) -> int:
    if is_compare_only_mode(args):
        return _run_with_state_phase(args, state, "legacy-extract", run_extract)
    return _legacy_phase_archived(
        "legacy-extract",
        "historical code is under archive/legacy_wfcllm_2026_07/code/extract.",
    )


def run_legacy_token_channel_train(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-token-channel-train",
        "historical token-channel code is under "
        "archive/legacy_wfcllm_2026_07/code/watermark/token_channel.",
    )


def run_legacy_build_entropy_profile(
    args: argparse.Namespace,
    state: RunStateManager,
) -> int:
    return _legacy_phase_archived(
        "legacy-build-entropy-profile",
        "historical adaptive-gamma code is under "
        "archive/legacy_wfcllm_2026_07/code/watermark/adaptive_gamma.",
    )


def run_legacy_pretrain(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-pretrain",
        "historical code is under archive/legacy_wfcllm_2026_07/code/pretrain.",
    )


def run_phase(phase: str, args: argparse.Namespace, state: RunStateManager) -> int:
    """Dispatch to registered phase runners."""
    runners = {
        "generate": run_generate,
        "calibrate": run_calibrate,
        "detect": run_detect,
        "report": run_report,
        "audit": run_audit,
        "posthoc-pass-report": run_posthoc_pass_report,
        "diagnostic-selector": run_diagnostic_selector,
        "encoder": run_encoder,
        "legacy-watermark": run_legacy_watermark,
        "legacy-extract": run_legacy_extract,
        "legacy-token-channel-train": run_legacy_token_channel_train,
        "legacy-build-entropy-profile": run_legacy_build_entropy_profile,
        "legacy-pretrain": run_legacy_pretrain,
        "legacy-ablation": run_legacy_ablation,
    }
    return runners[phase](args, state)


def run_encoder(args: argparse.Namespace, state: RunStateManager) -> int:
    """阶段一：训练语义编码器。"""
    import glob

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.train import main as encoder_main

    print("=== 阶段一：语义编码器预训练 ===")

    if args.eval_only:
        from wfcllm.encoder.train import evaluate_only

        default_best = str(Path(EncoderConfig().output_model_dir) / "best_model.pt")
        checkpoint = (
            args.checkpoint
            or (default_best if Path(default_best).exists() else None)
            or state.get("encoder", "checkpoint")
        )
        if not checkpoint:
            print("[错误] 未找到 checkpoint，请用 --checkpoint 指定路径", file=sys.stderr)
            return 1
        if not Path(checkpoint).exists():
            print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
            return 1
        print(f"[评测] 使用模型: {checkpoint}")

        config = EncoderConfig()
        if args.model_name:
            config.model_name = args.model_name
        if args.embed_dim:
            config.embed_dim = args.embed_dim
        if args.no_lora:
            config.use_lora = False
        if args.no_bf16:
            config.use_bf16 = False

        try:
            evaluate_only(checkpoint, config)
        except Exception as e:
            print(f"[错误] 评测失败：{e}", file=sys.stderr)
            return 1
        return 0

    config = EncoderConfig()
    if args.model_name:
        config.model_name = args.model_name
    if args.embed_dim:
        config.embed_dim = args.embed_dim
    if args.lr:
        config.lr = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.margin:
        config.margin = args.margin
    if args.no_lora:
        config.use_lora = False
    if args.no_bf16:
        config.use_bf16 = False

    if args.model_name is None:
        local_codet5 = Path(config.local_model_dir) / "codet5-base"
        if local_codet5.exists() and (local_codet5 / "config.json").exists():
            config.model_name = str(local_codet5)
            print(f"[自动] 使用本地模型: {config.model_name}")
        else:
            print(f"[回退] 使用 HF Hub 模型: {config.model_name}")

    try:
        encoder_main(config)
    except Exception as e:
        print(f"[错误] 编码器训练失败：{e}", file=sys.stderr)
        return 1

    best_model_path = str(Path(config.output_model_dir) / "best_model.pt")
    ckpt_pattern = str(Path(config.checkpoint_dir) / "encoder_epoch*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    checkpoint_path = checkpoints[-1] if checkpoints else config.checkpoint_dir

    state.mark_done("encoder", checkpoint=checkpoint_path, best_model_path=best_model_path)
    print(f"[完成] 编码器训练完毕，最优模型: {best_model_path}")
    return 0


def run_watermark(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "watermark",
        "use --phase legacy-watermark --legacy for archive guidance or the new generate phase.",
    )


def run_offline_analysis(args: argparse.Namespace) -> int:
    from wfcllm.evaluation.detection_report import (
        build_offline_regression_report,
        load_detail_artifact,
        load_summary_artifact,
        load_watermarked_artifact,
        write_offline_regression_report,
    )

    left_watermarked = (
        load_watermarked_artifact(args.compare_watermarked_left)
        if args.compare_watermarked_left
        else None
    )
    right_watermarked = (
        load_watermarked_artifact(args.compare_watermarked_right)
        if args.compare_watermarked_right
        else None
    )

    report = build_offline_regression_report(
        left_summary=load_summary_artifact(args.compare_summary_left),
        left_details=load_detail_artifact(args.compare_details_left),
        left_watermarked=left_watermarked,
        right_summary=load_summary_artifact(args.compare_summary_right),
        right_details=load_detail_artifact(args.compare_details_right),
        right_watermarked=right_watermarked,
    )
    output_path = write_offline_regression_report(args.compare_output, report)
    print(f"[完成] 离线回归报告已保存至 {output_path}")
    return 0


def run_extract(args: argparse.Namespace, state: RunStateManager) -> int:
    if is_compare_only_mode(args):
        return run_offline_analysis(args)
    return _legacy_phase_archived(
        "extract",
        "use --phase legacy-extract --legacy for archive guidance or the new detect phase.",
    )


def run_generate_negative(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "generate-negative",
        "historical negative-corpus generation is archived under "
        "archive/legacy_wfcllm_2026_07/code/extract/calibration.",
    )


def resolve_token_channel_train_config(args: argparse.Namespace) -> dict[str, object]:
    """Legacy token-channel config resolution is archived."""
    raise RuntimeError(
        "token-channel-train config resolution has been archived; see "
        "archive/legacy_wfcllm_2026_07/code/watermark/token_channel."
    )


def validate_token_channel_train_config(train_cfg: dict[str, object]) -> str | None:
    """Validate required user-facing inputs before workflow construction."""

    if not train_cfg.get("dataset"):
        return "[错误] token-channel-train 需要提供 dataset（可通过配置文件或 --dataset 指定）"
    if not train_cfg.get("lm_model_path"):
        return "[错误] token-channel-train 需要提供 lm_model_path（可通过配置文件或 --lm-model-path 指定）"
    return None


def run_token_channel_train(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "token-channel-train",
        "historical token-channel training is archived under "
        "archive/legacy_wfcllm_2026_07/code/watermark/token_channel.",
    )


def run_build_entropy_profile(
    args: argparse.Namespace,
    state: RunStateManager,
) -> int:
    return _legacy_phase_archived(
        "build-entropy-profile",
        "historical adaptive-gamma profile building is archived under "
        "archive/legacy_wfcllm_2026_07/code/watermark/adaptive_gamma.",
    )
