"""PhaseOrchestrator: iterates phases, handles skip/force, invokes prereqs."""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence

from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.prereq import PrereqRegistry, ensure_gate_phase_prerequisites
from wfcllm.orchestration.state import ALL_PHASES, GATE_PHASES, PHASES, RunStateManager

logger = logging.getLogger(__name__)


class PhaseOrchestrator:
    """Drives phase execution. Matches run.py:main() loop behavior 1:1."""

    def __init__(
        self,
        state: RunStateManager,
        phase_registry: PhaseRegistry,
        prereq_registry: PrereqRegistry,
        resolved_config: Mapping[str, object] | None = None,
    ) -> None:
        self._state = state
        self._phases = phase_registry
        self._prereqs = prereq_registry
        self._resolved_config = dict(resolved_config or {})

    def resolve_phase_sequence(self, args: argparse.Namespace | None = None) -> list[str]:
        """Resolve method-specific defaults without changing the legacy baseline."""

        config: Mapping[str, object] = self._resolved_config
        if args is not None:
            cached = getattr(args, "_config_cache", None)
            if isinstance(cached, Mapping):
                config = cached
        runtime = config.get("runtime") if isinstance(config, Mapping) else None
        raw_phases = runtime.get("default_phases") if isinstance(runtime, Mapping) else None
        if raw_phases is None:
            return list(PHASES)
        if (
            not isinstance(raw_phases, Sequence)
            or isinstance(raw_phases, (str, bytes))
            or not raw_phases
        ):
            raise ValueError("runtime.default_phases must be a non-empty phase list")
        phases = list(raw_phases)
        if any(not isinstance(phase, str) or phase not in ALL_PHASES for phase in phases):
            raise ValueError("runtime.default_phases contains an unknown default phase")
        if len(phases) != len(set(phases)):
            raise ValueError("runtime.default_phases must not contain duplicates")
        return phases

    def run(self, args: argparse.Namespace) -> int:
        phases_to_run = [args.phase] if args.phase else self.resolve_phase_sequence(args)

        for phase in phases_to_run:
            if self._should_skip(args, phase):
                print(f"[跳过] {phase}（已完成，使用 --force 强制重跑）")
                continue

            config = getattr(args, "_config_cache", None) or {}
            ensure_gate_phase_prerequisites(phase, config, args, self._state)
            self._prereqs.ensure_satisfied(phase, config, runner=self)

            runner = self._phases.get(phase)
            rc = runner(args, self._state)
            if rc != 0:
                print(f"[失败] {phase} 阶段退出码 {rc}", file=sys.stderr)
                return rc

        return 0

    def run_phase(
        self,
        phase: str,
        args: argparse.Namespace | None = None,
    ) -> int:
        """Run one phase with the same prerequisite checks as the main loop."""

        if phase not in ALL_PHASES:
            raise ValueError(f"unknown phase: {phase}")
        if args is None:
            args = argparse.Namespace(
                phase=phase,
                force=False,
                eval_only=False,
                input=None,
                run_id=None,
                run_dir=None,
            )
            setattr(args, "_config_cache", dict(self._resolved_config))
        config = getattr(args, "_config_cache", None) or self._resolved_config
        ensure_gate_phase_prerequisites(phase, config, args, self._state)
        self._prereqs.ensure_satisfied(phase, dict(config), runner=self)
        return self._phases.get(phase)(args, self._state)

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
        if phase in GATE_PHASES:
            config = getattr(args, "_config_cache", None) or {}
            from wfcllm.cli.runners import _is_gated

            if not _is_gated(config):
                return False
            hashes = getattr(args, "_phase_input_hashes", None)
            current = hashes.get(phase) if isinstance(hashes, Mapping) else None
            if current is None:
                try:
                    from wfcllm.cli.runners import compute_phase_input_hash

                    current = compute_phase_input_hash(args, phase)
                except ValueError:
                    # Do not trust an old done row when current inputs cannot be
                    # authenticated; the runner will report the precise error.
                    return False
            previous = self._state.get(phase, "input_hash")
            if not isinstance(current, str) or current != previous:
                return False
            manifest_path = self._state.get(phase, "manifest_path")
            output_hash = self._state.get(phase, "output_manifest_hash")
            artifact_path = self._state.get(phase, "output_artifact_path")
            artifact_hash = self._state.get(phase, "output_artifact_hash")
            if (
                not isinstance(manifest_path, str)
                or not isinstance(output_hash, str)
                or not isinstance(artifact_path, str)
                or not isinstance(artifact_hash, str)
            ):
                return False
            try:
                from pathlib import Path
                from wfcllm.cli.runners import (
                    _canonical_local_path,
                    _safe_file_hash,
                    _stable_tree_hash,
                    expected_gate_state_paths,
                )

                expected_artifact, expected_manifest = expected_gate_state_paths(args, phase)
                if _canonical_local_path(Path(manifest_path)) != expected_manifest:
                    return False
                if _canonical_local_path(Path(artifact_path)) != expected_artifact:
                    return False

                if _safe_file_hash(Path(manifest_path)) != output_hash:
                    return False
                if _stable_tree_hash(Path(artifact_path)) != artifact_hash:
                    return False
            except ValueError:
                return False
        # compare-only mode handled in cli/entry.py before orchestrator runs
        if phase == "detect" and self._has_explicit_detect_input(args):
            return False
        if phase == "legacy-extract" and self._has_explicit_legacy_extract_input(args):
            return False
        return True

    @staticmethod
    def _has_explicit_detect_input(args: argparse.Namespace) -> bool:
        cli_input = getattr(args, "input", None)
        config_cache = getattr(args, "_config_cache", None) or {}
        config_input = (config_cache.get("detector") or {}).get("input")
        return bool(cli_input or config_input)

    @staticmethod
    def _has_explicit_legacy_extract_input(args: argparse.Namespace) -> bool:
        cli_input = getattr(args, "input_file", None)
        config_cache = getattr(args, "_config_cache", None) or {}
        config_input = (config_cache.get("extract") or {}).get("input_file")
        return bool(cli_input or config_input)
