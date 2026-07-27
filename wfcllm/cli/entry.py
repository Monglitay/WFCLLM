from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
import sys

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.config_resolver import load_config, resolve_method_config
from wfcllm.cli.runners import (
    run_audit,
    run_calibrate,
    run_detect,
    run_encoder,
    run_generate,
    run_gate_data,
    run_gate_train,
    run_report,
)
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.pipeline import PhaseOrchestrator
from wfcllm.orchestration.state import (
    DEFAULT_STATE_FILE,
    PHASES,
    RunStateManager,
)


def _cmd_status(state: RunStateManager) -> None:
    print("=== WFCLLM Gate-only phase status ===")
    status = state.status()
    for phase in PHASES:
        row = status[phase]
        label = "complete" if row["done"] else "pending"
        print(f"{phase}: {label}")


def _populate_phase_registry(registry: PhaseRegistry) -> None:
    registry.register("encoder", run_encoder)
    registry.register("gate-data", run_gate_data)
    registry.register("gate-train", run_gate_train)
    registry.register("generate", run_generate)
    registry.register("calibrate", run_calibrate)
    registry.register("detect", run_detect)
    registry.register("report", run_report)
    registry.register("audit", run_audit)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("WFCLLM_DEBUG") else logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    state_path = Path(args.state_file) if args.state_file else DEFAULT_STATE_FILE
    try:
        if args.status:
            state = RunStateManager(path=state_path)
            _cmd_status(state)
            return 0

        config = resolve_method_config(load_config(args.config))
        _validate_fresh_run_invocation(args, state_path)
        state = RunStateManager(path=state_path)
        if args.gate_data_scale is not None:
            config = deepcopy(config)
            gate_data = config.get("gate_data")
            if not isinstance(gate_data, dict):
                raise ValueError("--gate-data-scale requires gate_data config")
            gate_data["scale"] = args.gate_data_scale
        setattr(args, "_config_cache", config)

        registry = PhaseRegistry()
        _populate_phase_registry(registry)
        return PhaseOrchestrator(
            state=state,
            phase_registry=registry,
            resolved_config=config,
        ).run(args)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1


def _validate_fresh_run_invocation(args: object, state_path: Path) -> None:
    """Bind direct CLI phases to one fresh run root and persisted state."""

    phase = getattr(args, "phase", None)
    run_dir_value = getattr(args, "run_dir", None)
    if not isinstance(run_dir_value, str) or not run_dir_value:
        raise ValueError("execution requires a non-empty --run-dir")
    run_dir = Path(run_dir_value)
    starts_with_encoder = phase in {None, "encoder"}
    if starts_with_encoder:
        if state_path.exists() or state_path.is_symlink():
            raise ValueError(
                "Fresh Reproduction Run requires a new state file for encoder"
            )
        if run_dir.exists() or run_dir.is_symlink():
            raise ValueError(
                "Fresh Reproduction Run requires a new run directory for encoder"
            )
        return
    if not state_path.is_file():
        raise ValueError(
            f"{phase} requires the current run's existing state file"
        )
    if not run_dir.is_dir():
        raise ValueError(
            f"{phase} requires the current run's existing run directory"
        )


if __name__ == "__main__":
    raise SystemExit(main())
