"""PhaseOrchestrator: iterates phases, handles skip/force, invokes prereqs."""
from __future__ import annotations

import argparse
import logging
import sys

from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.prereq import PrereqRegistry
from wfcllm.orchestration.state import PHASES, RunStateManager

logger = logging.getLogger(__name__)


class PhaseOrchestrator:
    """Drives phase execution. Matches run.py:main() loop behavior 1:1."""

    def __init__(
        self,
        state: RunStateManager,
        phase_registry: PhaseRegistry,
        prereq_registry: PrereqRegistry,
    ) -> None:
        self._state = state
        self._phases = phase_registry
        self._prereqs = prereq_registry

    def run(self, args: argparse.Namespace) -> int:
        phases_to_run = [args.phase] if args.phase else PHASES

        for phase in phases_to_run:
            if self._should_skip(args, phase):
                print(f"[跳过] {phase}（已完成，使用 --force 强制重跑）")
                continue

            config = getattr(args, "_config_cache", None) or {}
            self._prereqs.ensure_satisfied(phase, config, runner=self)

            runner = self._phases.get(phase)
            rc = runner(args, self._state)
            if rc != 0:
                print(f"[失败] {phase} 阶段退出码 {rc}", file=sys.stderr)
                return rc

        return 0

    def dispatch_phase(self, phase: str, args: argparse.Namespace) -> int:
        """Invoke a registered runner directly, bypassing skip/force/prereq logic.

        Used by Prereq fixers that need to chain into another phase (e.g.
        watermark's entropy-profile prereq triggers the `build-entropy-profile`
        phase). Prereqs are intentionally skipped here to avoid infinite loops
        when a phase's own prereq tries to satisfy itself.
        """
        runner = self._phases.get(phase)
        return runner(args, self._state)

    def _should_skip(self, args: argparse.Namespace, phase: str) -> bool:
        """Replicates run.py:should_skip_completed_phase logic."""
        if not self._state.is_done(phase):
            return False
        if getattr(args, "force", False) or getattr(args, "eval_only", False):
            return False
        # compare-only mode handled in cli/entry.py before orchestrator runs
        if phase == "detect" and self._has_explicit_detect_input(args):
            return False
        return True

    @staticmethod
    def _has_explicit_detect_input(args: argparse.Namespace) -> bool:
        cli_input = getattr(args, "input", None)
        config_cache = getattr(args, "_config_cache", None) or {}
        config_input = (config_cache.get("detector") or {}).get("input")
        return bool(cli_input or config_input)
