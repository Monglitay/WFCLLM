"""WFCLLM run-time entry point.

Replaces run.py:main() and its helpers. Wires up RunStateManager + PhaseRegistry
+ PrereqRegistry + PhaseOrchestrator.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.config_resolver import load_config
from wfcllm.cli.runners import (
    is_compare_only_mode,
    validate_compare_only_mode,
    run_audit,
    run_build_entropy_profile,
    run_calibrate,
    run_detect,
    run_diagnostic_selector,
    run_encoder,
    run_extract,
    run_generate,
    run_generate_negative,
    run_legacy_ablation,
    run_legacy_build_entropy_profile,
    run_legacy_extract,
    run_legacy_pretrain,
    run_legacy_token_channel_train,
    run_legacy_watermark,
    run_posthoc_pass_report,
    run_report,
    run_token_channel_train,
    run_watermark,
)
from wfcllm.pretrain.runner import run_pretrain
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.pipeline import PhaseOrchestrator
from wfcllm.orchestration.prereq import PrereqRegistry
from wfcllm.orchestration.state import (
    ALL_PHASES,
    DEFAULT_STATE_FILE,
    LEGACY_PHASES,
    RunStateManager,
)


def _cmd_status(state: RunStateManager) -> None:
    print("=== WFCLLM 阶段状态 ===")
    for phase in ALL_PHASES:
        info = state.status()[phase]
        done_str = "✓ 完成" if info["done"] else "○ 未完成"
        extras = {k: v for k, v in info.items() if k not in ("done", "completed_at")}
        extra_str = "  " + str(extras) if extras else ""
        print(f"  {phase:10s} {done_str}{extra_str}")


def _cmd_reset(state: RunStateManager) -> None:
    state.reset()
    print("已重置所有阶段状态。")


def _populate_phase_registry(reg: PhaseRegistry) -> None:
    reg.register("generate", run_generate)
    reg.register("calibrate", run_calibrate)
    reg.register("detect", run_detect)
    reg.register("report", run_report)
    reg.register("audit", run_audit)
    reg.register("posthoc-pass-report", run_posthoc_pass_report)
    reg.register("diagnostic-selector", run_diagnostic_selector)
    reg.register("encoder", run_encoder)
    reg.register("legacy-watermark", run_legacy_watermark)
    reg.register("legacy-extract", run_legacy_extract)
    reg.register("legacy-token-channel-train", run_legacy_token_channel_train)
    reg.register("legacy-build-entropy-profile", run_legacy_build_entropy_profile)
    reg.register("legacy-pretrain", run_legacy_pretrain)
    reg.register("legacy-ablation", run_legacy_ablation)


def validate_legacy_phase_request(args: argparse.Namespace) -> str | None:
    phase = getattr(args, "phase", None)
    if isinstance(phase, str) and phase in LEGACY_PHASES and not getattr(args, "legacy", False):
        return "[错误] legacy phase requires --legacy"
    return None


def main(argv: list[str] | None = None) -> int:
    log_level = logging.DEBUG if os.environ.get("WFCLLM_DEBUG") else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s %(levelname)s %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)
    state = RunStateManager(path=DEFAULT_STATE_FILE)

    legacy_error = validate_legacy_phase_request(args)
    if legacy_error is not None:
        print(legacy_error, file=sys.stderr)
        return 1

    if args.status:
        _cmd_status(state)
        return 0

    if args.reset:
        _cmd_reset(state)
        return 0

    compare_only_error = validate_compare_only_mode(args)
    if compare_only_error is not None:
        print(compare_only_error, file=sys.stderr)
        return 1

    # Preserve original run.py behavior: compare-only mode bypasses the "phase done → skip"
    # optimization since the user is supplying new compare-source files for this run.
    if is_compare_only_mode(args):
        args.force = True

    # Cache the loaded config on args for downstream helpers (matches old behavior).
    setattr(args, "_config_cache", load_config(args.config))

    phase_registry = PhaseRegistry()
    _populate_phase_registry(phase_registry)

    orchestrator = PhaseOrchestrator(
        state=state,
        phase_registry=phase_registry,
        prereq_registry=PrereqRegistry(),
    )
    return orchestrator.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
