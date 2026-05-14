"""WFCLLM run-time entry point.

Replaces run.py:main() and its helpers. Wires up RunStateManager + PhaseRegistry
+ PrereqRegistry + PhaseOrchestrator.
"""
from __future__ import annotations

import logging
import os
import sys

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.config_resolver import load_config
from wfcllm.cli.runners import (
    is_compare_only_mode,
    validate_compare_only_mode,
    run_encoder,
    run_extract,
    run_generate_negative,
    run_token_channel_train,
    run_watermark,
)
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.pipeline import PhaseOrchestrator
from wfcllm.orchestration.prereq import PrereqRegistry
from wfcllm.orchestration.state import (
    ALL_PHASES,
    DEFAULT_STATE_FILE,
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
    reg.register("encoder", run_encoder)
    reg.register("watermark", run_watermark)
    reg.register("extract", run_extract)
    reg.register("generate-negative", run_generate_negative)
    reg.register("token-channel-train", run_token_channel_train)


def main(argv: list[str] | None = None) -> int:
    log_level = logging.DEBUG if os.environ.get("WFCLLM_DEBUG") else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s %(levelname)s %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)
    state = RunStateManager(path=DEFAULT_STATE_FILE)

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
