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
