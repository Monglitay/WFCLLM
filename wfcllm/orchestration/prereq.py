from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

_PREVIOUS_PHASE = {
    "gate-data": "encoder",
    "gate-train": "gate-data",
    "generate": "gate-train",
    "calibrate": "generate",
    "detect": "calibrate",
    "report": "detect",
    "audit": "report",
}


def ensure_gate_phase_prerequisites(
    phase: str,
    config: Mapping[str, object],
    args: Any,
    state: Any,
) -> None:
    method = config.get("method") if isinstance(config, Mapping) else None
    if not isinstance(method, Mapping) or method.get("name") != "gated_semantic_window_v1":
        raise ValueError("only gated_semantic_window_v1 is supported")
    prerequisite = _PREVIOUS_PHASE.get(phase)
    if prerequisite is not None and not state.is_done(prerequisite):
        raise ValueError(
            f"{phase} requires the current run's completed {prerequisite} phase"
        )
    if prerequisite is None:
        return
    state_run_dir = state.get("encoder", "run_dir")
    requested_run_dir = getattr(args, "run_dir", None)
    if state_run_dir is None and requested_run_dir is None:
        return
    if not isinstance(state_run_dir, str) or not state_run_dir:
        raise ValueError("current run state is missing its encoder run directory")
    if not isinstance(requested_run_dir, str) or not requested_run_dir:
        raise ValueError(f"{phase} requires --run-dir")
    if Path(os.path.abspath(requested_run_dir)) != Path(state_run_dir):
        raise ValueError(f"{phase} must use the encoder phase's current run directory")
