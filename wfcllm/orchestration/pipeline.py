from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.prereq import ensure_gate_phase_prerequisites
from wfcllm.orchestration.state import PHASES, RunStateManager


class PhaseOrchestrator:
    """Run the fixed Gate-only phase chain exactly once per fresh state file."""

    def __init__(
        self,
        state: RunStateManager,
        phase_registry: PhaseRegistry,
        resolved_config: Mapping[str, object] | None = None,
    ) -> None:
        self._state = state
        self._phases = phase_registry
        self._resolved_config = dict(resolved_config or {})

    def resolve_phase_sequence(
        self, args: argparse.Namespace | None = None
    ) -> list[str]:
        config: Mapping[str, object] = self._resolved_config
        if args is not None and isinstance(
            getattr(args, "_config_cache", None), Mapping
        ):
            config = args._config_cache
        runtime = config.get("runtime")
        raw = runtime.get("default_phases") if isinstance(runtime, Mapping) else None
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("runtime.default_phases must define the full phase chain")
        phases = list(raw)
        if phases != PHASES:
            raise ValueError("runtime.default_phases must equal the Gate-only phase chain")
        return phases

    def run(self, args: argparse.Namespace) -> int:
        phases = [args.phase] if args.phase else self.resolve_phase_sequence(args)
        for phase in phases:
            if phase == "encoder":
                run_dir_value = getattr(args, "run_dir", None)
                if self._state.existed_at_initialization:
                    raise ValueError(
                        "Fresh Reproduction Run requires a new state file for encoder"
                    )
                if (
                    isinstance(run_dir_value, (str, bytes))
                    and run_dir_value
                    and Path(run_dir_value).exists()
                ):
                    raise ValueError(
                        "Fresh Reproduction Run requires a new run directory for encoder"
                    )
            if self._state.is_done(phase):
                raise ValueError(
                    f"Fresh Reproduction Run refuses completed phase {phase!r}; "
                    "use a new run directory and state file"
                )
            config = getattr(args, "_config_cache", None) or self._resolved_config
            ensure_gate_phase_prerequisites(phase, config, args, self._state)
            return_code = self._phases.get(phase)(args, self._state)
            if return_code != 0:
                print(
                    f"[失败] {phase} 阶段退出码 {return_code}",
                    file=sys.stderr,
                )
                return return_code
        return 0

    def run_phase(
        self, phase: str, args: argparse.Namespace | None = None
    ) -> int:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        if args is None:
            args = argparse.Namespace(phase=phase)
            setattr(args, "_config_cache", dict(self._resolved_config))
        return self.run(args)
