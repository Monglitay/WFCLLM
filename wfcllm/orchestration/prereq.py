"""Prereq framework: main-flow phases auto-trigger missing dependencies.

Concrete prereqs (entropy profile, threshold calibration) are registered by their
producing components in later refactor phases. This module provides only the
framework; Phase 1 ships with zero prereqs registered.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class Prereq:
    """A single prerequisite: a check function and a fix function.

    - name: human-readable identifier (used in logs)
    - check(config) -> bool: returns True if prereq is satisfied
    - fix(config, runner) -> None: produces the missing artifact
        `runner` is the PhaseOrchestrator (or compatible) so the fix can invoke
        another phase. Pass None when no nested phase invocation is needed.
    """
    name: str
    check: Callable[[dict], bool]
    fix: Callable[[dict, Any], None]


class PrereqRegistry:
    """Module-singleton registry of phase → list[Prereq].

    `PrereqRegistry()` always returns the same underlying store. This matches
    the pattern where components register at import time (e.g.,
    `wfcllm/watermark/adaptive_gamma/__init__.py` registers the entropy-profile
    prereq when imported).
    """

    _by_phase: dict[str, list[Prereq]] = {}

    def register(self, phase: str, prereq: Prereq) -> None:
        self._by_phase.setdefault(phase, []).append(prereq)

    def ensure_satisfied(self, phase: str, config: dict, runner: Any) -> None:
        for prereq in self._by_phase.get(phase, []):
            if prereq.check(config):
                logger.debug("prereq satisfied: phase=%s name=%s", phase, prereq.name)
                continue
            logger.info(
                "[orchestrator] phase=%s missing prereq '%s'; auto-triggering",
                phase, prereq.name,
            )
            prereq.fix(config, runner)

    def clear(self) -> None:
        """Reset all registered prereqs (test fixture helper)."""
        self._by_phase.clear()


def ensure_gate_phase_prerequisites(
    phase: str,
    config: Mapping[str, object],
    args: Any,
    state: Any,
) -> None:
    """Fail-closed artifact prerequisites for the gated formal mainline."""

    method = config.get("method") if isinstance(config, Mapping) else None
    if not isinstance(method, Mapping) or method.get("name") != "gated_semantic_window_v1":
        return

    from wfcllm.cli.runners import (
        _gate_run_dir,
        _load_json_object,
        _load_formal_json,
        _reject_symlink_path,
        _require_experimental_manifest,
        _uses_fast_experimental_gate_training,
        resolve_gate_bundle,
    )

    fast_experimental = _uses_fast_experimental_gate_training(config)
    if phase == "gate-train":
        run_dir = _gate_run_dir(args, config)
        data_manifest = run_dir / "gate-data" / "manifest.json"
        _reject_symlink_path(data_manifest)
        if not data_manifest.is_file():
            raise ValueError(f"{phase} requires the gate-data manifest")
        if fast_experimental:
            _require_experimental_manifest(
                _load_json_object(data_manifest),
                "gate-data",
            )
        else:
            _load_formal_json(data_manifest, "gate-data")
    if phase in {"generate", "calibrate", "detect"}:
        try:
            _path, bundle_hash = resolve_gate_bundle(args)
        except ValueError as exc:
            raise ValueError(f"gate bundle prerequisite failed: {exc}") from exc
        if phase == "calibrate" and state.get("generate", "gate_bundle_sha256") != bundle_hash:
            raise ValueError("calibrate requires the same gate bundle as generate")
        if phase == "detect":
            if state.get("generate", "gate_bundle_sha256") != bundle_hash:
                raise ValueError("detect requires the same gate bundle as generate")
            if state.get("calibrate", "gate_bundle_sha256") != bundle_hash:
                raise ValueError("detect requires the same gate bundle as calibrate")
