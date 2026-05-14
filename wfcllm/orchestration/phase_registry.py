"""Phase name → runner callable registry."""
from __future__ import annotations

import argparse
from collections.abc import Callable

from wfcllm.orchestration.state import RunStateManager

PhaseRunner = Callable[[argparse.Namespace, RunStateManager], int]


class PhaseRegistry:
    """Decouples PhaseOrchestrator from concrete runner imports.

    Runners self-register at import time; orchestrator looks them up by phase name.
    """

    def __init__(self) -> None:
        self._runners: dict[str, PhaseRunner] = {}

    def register(self, phase: str, runner: PhaseRunner) -> None:
        self._runners[phase] = runner

    def get(self, phase: str) -> PhaseRunner:
        if phase not in self._runners:
            raise KeyError(
                f"unknown phase: '{phase}'. registered: {sorted(self._runners)}"
            )
        return self._runners[phase]

    def phases(self) -> list[str]:
        return list(self._runners)
