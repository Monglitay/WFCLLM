# Repo Refactor Phase 1: Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `run.py`'s CLI / config / orchestration logic into dedicated subpackages, leaving `run.py` as a 3-line shim. Add the registry + prereq framework that subsequent phases will build on.

**Architecture:** Pure code-organization refactor. No behavior changes. `RunState` → `wfcllm/orchestration/state.py:RunStateManager`. CLI argument parsing → `wfcllm/cli/arguments.py`. Config merge helpers → `wfcllm/cli/config_resolver.py`. Per-phase runner wrappers → `wfcllm/cli/runners.py`. Phase dispatch + skip/force logic → `wfcllm/orchestration/pipeline.py:PhaseOrchestrator`. Two new utilities: generic `Registry[T]` container (`wfcllm/common/registry.py`) and prereq framework (`wfcllm/orchestration/prereq.py`). Existing tests must keep passing; one new unit test per new module.

**Tech Stack:** Python 3.11, pytest, dataclasses, argparse (already in use). No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §3, §4.5, §10 step 1.

**Pre-flight:**
1. On a clean `main` working tree.
2. Confirm baseline tests are green:
   ```
   HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py tests/test_run_config.py -v
   ```
   Expected: all pass (this is the "before" state we'll preserve).
3. Note current line counts (`wc -l run.py` = 1482). Target after this plan: `run.py` ≤ 10 lines.

---

## Task 1: Generic `Registry[T]` container

**Files:**
- Create: `wfcllm/common/registry.py`
- Test: `tests/common/test_registry.py`

A typed registry container that `lang/` and `datasets/` will use (later phases). Phase 1 only sets it up.

- [ ] **Step 1: Write the failing test**

Create `tests/common/test_registry.py`:

```python
"""Tests for the generic Registry[T] container."""
import pytest

from wfcllm.common.registry import Registry


class FakeAdapter:
    name = "fake"


class AnotherAdapter:
    name = "another"


def test_register_and_get_by_name():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("fake", FakeAdapter)
    assert isinstance(reg.get("fake"), FakeAdapter)


def test_get_unknown_name_raises_with_listing():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("fake", FakeAdapter)
    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    assert "missing" in str(excinfo.value)
    assert "fake" in str(excinfo.value)
    assert "test" in str(excinfo.value)


def test_duplicate_registration_raises():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("fake", FakeAdapter)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("fake", AnotherAdapter)


def test_names_returns_sorted_list():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("zebra", FakeAdapter)
    reg.register("alpha", FakeAdapter)
    assert reg.names() == ["alpha", "zebra"]
```

- [ ] **Step 2: Run test to verify it fails**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/common/test_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'wfcllm.common.registry'`

- [ ] **Step 3: Implement Registry**

Create `wfcllm/common/registry.py`:

```python
"""Generic typed registry used by LanguageAdapter / DatasetAdapter and other plug-in points."""
from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → factory class registry.

    Usage:
        registry: Registry[LanguageAdapter] = Registry("language")
        registry.register("python", PythonAdapter)
        adapter = registry.get("python")    # instantiates PythonAdapter()
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str, cls: type[T]) -> None:
        if name in self._entries:
            raise ValueError(
                f"{self._label} registry: '{name}' already registered "
                f"(existing={self._entries[name].__name__}, new={cls.__name__})"
            )
        self._entries[name] = cls

    def get(self, name: str) -> T:
        if name not in self._entries:
            raise KeyError(
                f"unknown {self._label}: '{name}'. "
                f"registered names: {self.names()}"
            )
        return self._entries[name]()

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
```

- [ ] **Step 4: Run test to verify it passes**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/common/test_registry.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```
git add wfcllm/common/registry.py tests/common/test_registry.py
git commit -m "feat: add generic Registry[T] container for adapter plug-in points"
```

---

## Task 2: `wfcllm/orchestration/` package skeleton

**Files:**
- Create: `wfcllm/orchestration/__init__.py`

Empty package marker so subsequent tasks can import.

- [ ] **Step 1: Create the directory and empty `__init__.py`**

```
mkdir -p wfcllm/orchestration
```

Create `wfcllm/orchestration/__init__.py` with:

```python
"""Cross-phase orchestration: run-state, prereqs, pipeline dispatch."""
```

- [ ] **Step 2: Verify package importable**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "import wfcllm.orchestration; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```
git add wfcllm/orchestration/__init__.py
git commit -m "chore: add wfcllm/orchestration/ package skeleton"
```

---

## Task 3: `RunStateManager` in `wfcllm/orchestration/state.py`

**Files:**
- Create: `wfcllm/orchestration/state.py`
- Test: `tests/integration/test_orchestration.py`
- Modify: `run.py:54-58, 297-340` (will remove `RunState` class — done in Task 11)

The existing `RunState` class (run.py:297) becomes `RunStateManager` in its own module. **Behavior unchanged**: same `run_state.json` schema, same methods (`is_done`, `get`, `mark_done`, `reset`, `status`), same `ALL_PHASES` source-of-truth (lifted unchanged).

- [ ] **Step 1: Create the integration test directory**

```
mkdir -p tests/integration
touch tests/integration/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/integration/test_orchestration.py`:

```python
"""Tests for orchestration components: RunStateManager."""
import json
from pathlib import Path

import pytest

from wfcllm.orchestration.state import RunStateManager, ALL_PHASES


def test_initial_state_all_phases_pending(tmp_path):
    state = RunStateManager(path=tmp_path / "run_state.json")
    for phase in ALL_PHASES:
        assert state.is_done(phase) is False


def test_mark_done_persists_and_records_metadata(tmp_path):
    state_path = tmp_path / "run_state.json"
    state = RunStateManager(path=state_path)
    state.mark_done("encoder", checkpoint="x.pt", best_model_path="best.pt")

    # Persisted to disk
    assert state_path.exists()
    raw = json.loads(state_path.read_text())
    assert raw["encoder"]["done"] is True
    assert raw["encoder"]["checkpoint"] == "x.pt"
    assert "completed_at" in raw["encoder"]

    # Re-load reads it back
    state2 = RunStateManager(path=state_path)
    assert state2.is_done("encoder") is True
    assert state2.get("encoder", "checkpoint") == "x.pt"


def test_reset_clears_all_phases(tmp_path):
    state = RunStateManager(path=tmp_path / "run_state.json")
    state.mark_done("encoder", checkpoint="x.pt")
    state.reset()
    assert state.is_done("encoder") is False


def test_status_returns_full_dict_with_all_phases(tmp_path):
    state = RunStateManager(path=tmp_path / "run_state.json")
    state.mark_done("watermark", output_file="out.jsonl")
    status = state.status()
    assert set(status.keys()) == set(ALL_PHASES)
    assert status["watermark"]["done"] is True
    assert status["encoder"]["done"] is False


def test_default_path_is_data_run_state_json():
    state = RunStateManager()
    assert state._path == Path("data/run_state.json")
```

- [ ] **Step 3: Run test to verify it fails**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: `ModuleNotFoundError: No module named 'wfcllm.orchestration.state'`

- [ ] **Step 4: Create `wfcllm/orchestration/state.py`**

```python
"""Run-state persistence: read/write data/run_state.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Phase name source-of-truth (lifted unchanged from run.py to preserve schema).
PHASES = ["encoder", "watermark", "extract"]
OPTIONAL_PHASES = ["generate-negative", "token-channel-train"]
ALL_PHASES = PHASES + OPTIONAL_PHASES

DEFAULT_STATE_FILE = Path("data/run_state.json")


class RunStateManager:
    """Persistent phase completion tracker.

    Schema is preserved byte-for-byte from the original `RunState` class in run.py
    so existing data/run_state.json files load cleanly.
    """

    def __init__(self, path: Path = DEFAULT_STATE_FILE) -> None:
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {phase: {"done": False} for phase in ALL_PHASES}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def is_done(self, phase: str) -> bool:
        return self._data.get(phase, {}).get("done", False)

    def get(self, phase: str, key: str) -> str | None:
        return self._data.get(phase, {}).get(key)

    def mark_done(self, phase: str, **kwargs) -> None:
        self._data[phase] = {
            "done": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._save()

    def reset(self) -> None:
        self._data = {phase: {"done": False} for phase in ALL_PHASES}
        self._save()

    def status(self) -> dict:
        return {
            phase: self._data.get(phase, {"done": False})
            for phase in ALL_PHASES
        }
```

- [ ] **Step 5: Run test to verify it passes**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```
git add wfcllm/orchestration/state.py tests/integration/test_orchestration.py tests/integration/__init__.py
git commit -m "feat: add RunStateManager (extracts run.py:RunState into orchestration/)"
```

---

## Task 4: `PhaseRegistry` in `wfcllm/orchestration/phase_registry.py`

**Files:**
- Create: `wfcllm/orchestration/phase_registry.py`
- Modify: `tests/integration/test_orchestration.py` (append tests)

Maps phase name → runner callable. Decouples `PhaseOrchestrator` (Task 6) from concrete runner imports.

- [ ] **Step 1: Append failing tests to `tests/integration/test_orchestration.py`**

Append these tests at the end of `tests/integration/test_orchestration.py`:

```python
# --- PhaseRegistry tests ---

from wfcllm.orchestration.phase_registry import PhaseRegistry


def test_phase_registry_register_and_lookup():
    reg = PhaseRegistry()
    def fake_runner(args, state): return 0
    reg.register("encoder", fake_runner)
    assert reg.get("encoder") is fake_runner


def test_phase_registry_unknown_phase_raises():
    reg = PhaseRegistry()
    with pytest.raises(KeyError, match="unknown phase"):
        reg.get("nonexistent")


def test_phase_registry_phases_lists_registered():
    reg = PhaseRegistry()
    reg.register("encoder", lambda a, s: 0)
    reg.register("watermark", lambda a, s: 0)
    assert sorted(reg.phases()) == ["encoder", "watermark"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: 3 new tests fail with `ModuleNotFoundError: No module named 'wfcllm.orchestration.phase_registry'`

- [ ] **Step 3: Implement `PhaseRegistry`**

Create `wfcllm/orchestration/phase_registry.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: 8 tests pass (5 from Task 3 + 3 new).

- [ ] **Step 5: Commit**

```
git add wfcllm/orchestration/phase_registry.py tests/integration/test_orchestration.py
git commit -m "feat: add PhaseRegistry for phase name to runner mapping"
```

---

## Task 5: Prereq framework in `wfcllm/orchestration/prereq.py`

**Files:**
- Create: `wfcllm/orchestration/prereq.py`
- Modify: `tests/integration/test_orchestration.py` (append tests)

Spec §6.3: main-flow phases auto-trigger missing prereqs. This task builds the framework; concrete prereqs (entropy profile for watermark, threshold for extract) are registered in later phases when their producing components exist.

- [ ] **Step 1: Append failing tests**

Append to `tests/integration/test_orchestration.py`:

```python
# --- Prereq tests ---

from wfcllm.orchestration.prereq import Prereq, PrereqRegistry


def test_prereq_check_satisfied_does_nothing():
    fix_called = []
    prereq = Prereq(
        name="dummy",
        check=lambda cfg: True,
        fix=lambda cfg, runner: fix_called.append(True),
    )
    PrereqRegistry().clear()
    reg = PrereqRegistry()
    reg.register("watermark", prereq)
    reg.ensure_satisfied("watermark", config={}, runner=None)
    assert fix_called == []


def test_prereq_unsatisfied_triggers_fix():
    fix_called = []
    prereq = Prereq(
        name="dummy",
        check=lambda cfg: False,
        fix=lambda cfg, runner: fix_called.append("ran"),
    )
    PrereqRegistry().clear()
    reg = PrereqRegistry()
    reg.register("watermark", prereq)
    reg.ensure_satisfied("watermark", config={}, runner=None)
    assert fix_called == ["ran"]


def test_prereq_no_registered_for_phase_is_noop():
    PrereqRegistry().clear()
    reg = PrereqRegistry()
    # no prereqs registered for "extract"
    reg.ensure_satisfied("extract", config={}, runner=None)
    # should not raise


def test_prereq_registry_is_module_singleton():
    """PrereqRegistry() returns the same underlying store each call (module-level)."""
    reg_a = PrereqRegistry()
    reg_a.clear()
    reg_a.register("watermark", Prereq("p", lambda c: True, lambda c, r: None))
    reg_b = PrereqRegistry()
    assert "watermark" in reg_b._by_phase
    reg_b.clear()  # cleanup for other tests
```

- [ ] **Step 2: Run tests to verify they fail**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: 4 new tests fail with `ModuleNotFoundError: No module named 'wfcllm.orchestration.prereq'`

- [ ] **Step 3: Implement prereq framework**

Create `wfcllm/orchestration/prereq.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: 12 tests pass (8 prior + 4 new).

- [ ] **Step 5: Commit**

```
git add wfcllm/orchestration/prereq.py tests/integration/test_orchestration.py
git commit -m "feat: add prereq framework (concrete prereqs registered in later phases)"
```

---

## Task 6: `PhaseOrchestrator` in `wfcllm/orchestration/pipeline.py`

**Files:**
- Create: `wfcllm/orchestration/pipeline.py`
- Modify: `tests/integration/test_orchestration.py` (append tests)

Wraps run.py's main() phase-loop logic: iterate phases, skip-if-done, force-rerun, fail-fast on non-zero return. Calls `PrereqRegistry.ensure_satisfied()` before each phase. Behavior matches existing `main()` loop.

- [ ] **Step 1: Append failing tests**

Append to `tests/integration/test_orchestration.py`:

```python
# --- PhaseOrchestrator tests ---

import argparse
from wfcllm.orchestration.pipeline import PhaseOrchestrator


def _make_args(**kwargs):
    """Build minimal argparse.Namespace for orchestrator tests."""
    defaults = dict(force=False, eval_only=False, phase=None, input_file=None)
    defaults.update(kwargs)
    ns = argparse.Namespace(**defaults)
    # config cache hook used by has_explicit_extract_input
    setattr(ns, "_config_cache", {"extract": {}})
    return ns


def test_orchestrator_runs_all_main_phases_when_phase_unspecified(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    ran = []

    reg = PhaseRegistry()
    for p in ("encoder", "watermark", "extract"):
        reg.register(p, lambda a, s, name=p: (ran.append(name), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args())
    assert rc == 0
    assert ran == ["encoder", "watermark", "extract"]


def test_orchestrator_skips_completed_phase(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    state.mark_done("encoder")
    ran = []
    reg = PhaseRegistry()
    reg.register("encoder", lambda a, s: (ran.append("encoder"), 0)[1])
    reg.register("watermark", lambda a, s: (ran.append("watermark"), 0)[1])
    reg.register("extract", lambda a, s: (ran.append("extract"), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args())
    assert rc == 0
    assert ran == ["watermark", "extract"]


def test_orchestrator_force_reruns_completed_phase(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    state.mark_done("encoder")
    ran = []
    reg = PhaseRegistry()
    reg.register("encoder", lambda a, s: (ran.append("encoder"), 0)[1])
    reg.register("watermark", lambda a, s: (ran.append("watermark"), 0)[1])
    reg.register("extract", lambda a, s: (ran.append("extract"), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args(force=True))
    assert rc == 0
    assert ran == ["encoder", "watermark", "extract"]


def test_orchestrator_fails_fast_on_nonzero(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    ran = []
    reg = PhaseRegistry()
    reg.register("encoder", lambda a, s: (ran.append("encoder"), 0)[1])
    reg.register("watermark", lambda a, s: (ran.append("watermark"), 7)[1])  # fail
    reg.register("extract", lambda a, s: (ran.append("extract"), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args())
    assert rc == 7
    assert ran == ["encoder", "watermark"]  # extract never runs


def test_orchestrator_runs_single_phase_when_specified(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    ran = []
    reg = PhaseRegistry()
    for p in ("encoder", "watermark", "extract", "generate-negative"):
        reg.register(p, lambda a, s, name=p: (ran.append(name), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args(phase="generate-negative"))
    assert rc == 0
    assert ran == ["generate-negative"]


def test_orchestrator_invokes_prereqs_before_phase(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    events = []

    reg = PhaseRegistry()
    reg.register("watermark", lambda a, s: (events.append("phase"), 0)[1])

    preq = PrereqRegistry()
    preq.register("watermark", Prereq(
        name="dummy",
        check=lambda cfg: False,                    # always missing
        fix=lambda cfg, runner: events.append("prereq"),
    ))

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=preq)
    rc = orch.run(_make_args(phase="watermark"))
    assert rc == 0
    assert events == ["prereq", "phase"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: 6 new tests fail with `ModuleNotFoundError: No module named 'wfcllm.orchestration.pipeline'`

- [ ] **Step 3: Implement `PhaseOrchestrator`**

Create `wfcllm/orchestration/pipeline.py`:

```python
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

    def _should_skip(self, args: argparse.Namespace, phase: str) -> bool:
        """Replicates run.py:should_skip_completed_phase logic."""
        if not self._state.is_done(phase):
            return False
        if getattr(args, "force", False) or getattr(args, "eval_only", False):
            return False
        # compare-only mode handled in cli/entry.py before orchestrator runs
        if phase == "extract" and self._has_explicit_extract_input(args):
            return False
        return True

    @staticmethod
    def _has_explicit_extract_input(args: argparse.Namespace) -> bool:
        cli_input = getattr(args, "input_file", None)
        config_cache = getattr(args, "_config_cache", None) or {}
        config_input = (config_cache.get("extract") or {}).get("input_file")
        return bool(cli_input or config_input)
```

- [ ] **Step 4: Run tests to verify they pass**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
```

Expected: 18 tests pass (12 prior + 6 new).

- [ ] **Step 5: Commit**

```
git add wfcllm/orchestration/pipeline.py tests/integration/test_orchestration.py
git commit -m "feat: add PhaseOrchestrator (extracts run.py main-loop dispatch)"
```

---

## Task 7: `wfcllm/cli/` package skeleton

**Files:**
- Create: `wfcllm/cli/__init__.py`

- [ ] **Step 1: Create the package**

```
mkdir -p wfcllm/cli
```

Create `wfcllm/cli/__init__.py`:

```python
"""CLI layer: argparse definitions, config resolution, entry point."""
```

- [ ] **Step 2: Verify importable**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "import wfcllm.cli; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```
git add wfcllm/cli/__init__.py
git commit -m "chore: add wfcllm/cli/ package skeleton"
```

---

## Task 8: Move CLI argument definitions → `wfcllm/cli/arguments.py`

**Files:**
- Create: `wfcllm/cli/arguments.py`
- Modify: `run.py:343-672` (extract `build_parser`; keep import shim for now)
- Test: `tests/integration/test_cli_resolution.py`

`build_parser()` defines ~80 CLI flags. We lift it verbatim into `cli/arguments.py`. To keep run.py functional during the multi-task migration, we import the function back into run.py (one-line shim). It's removed from run.py in Task 11.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_cli_resolution.py`:

```python
"""Tests for CLI argument parsing."""
import pytest
from wfcllm.cli.arguments import build_parser


def test_build_parser_returns_argparse():
    import argparse
    assert isinstance(build_parser(), argparse.ArgumentParser)


def test_parser_accepts_phase_choice():
    parser = build_parser()
    args = parser.parse_args(["--phase", "encoder"])
    assert args.phase == "encoder"


def test_parser_rejects_unknown_phase():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "nonexistent"])


def test_parser_status_flag():
    parser = build_parser()
    args = parser.parse_args(["--status"])
    assert args.status is True


def test_parser_reset_flag():
    parser = build_parser()
    args = parser.parse_args(["--reset"])
    assert args.reset is True


def test_parser_secret_key_override():
    parser = build_parser()
    args = parser.parse_args(["--secret-key", "abc"])
    assert args.secret_key == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v
```

Expected: `ModuleNotFoundError: No module named 'wfcllm.cli.arguments'`

- [ ] **Step 3: Lift `build_parser` into `wfcllm/cli/arguments.py`**

Read the current `build_parser` definition:

```
sed -n '343,672p' run.py
```

Copy the entire function (and its `DEFAULT_CONFIG_FILE` dependency if referenced) into `wfcllm/cli/arguments.py`. Header:

```python
"""Argparse definitions for the WFCLLM run.py entry point.

Lifted verbatim from run.py:343-672 (Phase 1 refactor). No behavior changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wfcllm.orchestration.state import ALL_PHASES

DEFAULT_CONFIG_FILE = Path("configs/base_config.json")


def build_parser() -> argparse.ArgumentParser:
    # ... entire function body copied verbatim from run.py
```

**Important**: paste the function exactly as it appears in run.py:343 (currently 330 lines). Do not modify any flag.

- [ ] **Step 4: Update `run.py` to re-export build_parser**

In `run.py`, replace lines 343-672 (the entire `build_parser` definition) with one line:

```python
from wfcllm.cli.arguments import build_parser, DEFAULT_CONFIG_FILE
```

Also remove the original `DEFAULT_CONFIG_FILE = Path("configs/base_config.json")` line near top of run.py (it now comes from `cli.arguments`).

- [ ] **Step 5: Run the parser test**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v
```

Expected: 6 tests pass.

- [ ] **Step 6: Run the existing run.py test suite**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py tests/test_run_config.py -v
```

Expected: All pass (no behavior change).

- [ ] **Step 7: Commit**

```
git add wfcllm/cli/arguments.py run.py tests/integration/test_cli_resolution.py
git commit -m "refactor: extract build_parser into wfcllm/cli/arguments.py"
```

---

## Task 9: Move config helpers → `wfcllm/cli/config_resolver.py`

**Files:**
- Create: `wfcllm/cli/config_resolver.py`
- Modify: `run.py:73-296` (extract `load_config`, `resolve_*`, `_apply_token_channel_cli_overrides`, `build_extract_calibration_contract_builder`)

The `load_config` + 7 `resolve_*` helpers (run.py:73-296) are pure functions for merging CLI overrides into config dicts. Lift them as-is.

- [ ] **Step 1: Append a failing test**

Append to `tests/integration/test_cli_resolution.py`:

```python
# --- config_resolver tests ---

import json
from wfcllm.cli.config_resolver import load_config, parse_optional_bool


def test_load_config_reads_json(tmp_path):
    cfg_path = tmp_path / "test.json"
    cfg_path.write_text(json.dumps({"encoder": {"lr": 1e-4}}))
    cfg = load_config(cfg_path)
    assert cfg == {"encoder": {"lr": 1e-4}}


def test_load_config_missing_file_returns_empty():
    cfg = load_config(Path("/nonexistent/path.json"))
    assert cfg == {}


def test_parse_optional_bool_true():
    assert parse_optional_bool("true") is True
    assert parse_optional_bool("TRUE") is True
    assert parse_optional_bool("1") is True


def test_parse_optional_bool_false():
    assert parse_optional_bool("false") is False
    assert parse_optional_bool("0") is False
```

Add `from pathlib import Path` at the top of the test file (if not already present).

- [ ] **Step 2: Run test to verify it fails**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v
```

Expected: 4 new tests fail with `ImportError: cannot import name 'load_config' from 'wfcllm.cli.config_resolver'`

- [ ] **Step 3: Move helpers into `wfcllm/cli/config_resolver.py`**

Read the helpers section:

```
sed -n '62,296p' run.py
```

Create `wfcllm/cli/config_resolver.py`:

```python
"""CLI override + JSON config merge helpers.

Lifted verbatim from run.py:62-296 (Phase 1 refactor). Pure functions; no behavior change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ... copy all of:
# - parse_optional_bool
# - load_config
# - resolve_extract_lsh_params
# - resolve_adaptive_gamma_config
# - resolve_extract_adaptive_gamma_config
# - resolve_token_channel_config
# - _apply_token_channel_cli_overrides
# - build_extract_calibration_contract_builder
# - resolve_adaptive_detection_config
```

Paste each function exactly as it is in run.py:62-296. Preserve all imports they need (`from wfcllm.watermark.config import ...` etc.).

- [ ] **Step 4: Update `run.py` to re-export from config_resolver**

In `run.py`, replace lines 62-296 with one block of imports:

```python
from wfcllm.cli.config_resolver import (
    parse_optional_bool,
    load_config,
    resolve_extract_lsh_params,
    resolve_adaptive_gamma_config,
    resolve_extract_adaptive_gamma_config,
    resolve_token_channel_config,
    _apply_token_channel_cli_overrides,
    build_extract_calibration_contract_builder,
    resolve_adaptive_detection_config,
)
```

- [ ] **Step 5: Run config_resolver tests**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v
```

Expected: 10 tests pass (6 + 4).

- [ ] **Step 6: Run the full run.py test suite**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py tests/test_run_config.py -v
```

Expected: All pass.

- [ ] **Step 7: Commit**

```
git add wfcllm/cli/config_resolver.py run.py tests/integration/test_cli_resolution.py
git commit -m "refactor: extract config-resolver helpers into wfcllm/cli/config_resolver.py"
```

---

## Task 10: Move per-phase runners → `wfcllm/cli/runners.py`

**Files:**
- Create: `wfcllm/cli/runners.py`
- Modify: `run.py:796-1481` (extract `run_encoder`, `run_watermark`, `run_offline_analysis`, `run_extract`, `run_generate_negative`, `resolve_token_channel_train_config`, `validate_token_channel_train_config`, `run_token_channel_train`, `run_phase`)

These functions are large but mostly straight imports + dataclass building + pipeline.run() + state.mark_done(). They're CLI-bound (read `args.Namespace`) so they live in `cli/`. Subsequent refactor phases will further decompose them into phase packages.

- [ ] **Step 1: Lift the runners block**

Read it first:

```
sed -n '784,1481p' run.py | head -100
```

Create `wfcllm/cli/runners.py`:

```python
"""Per-phase runner functions invoked by PhaseOrchestrator.

Lifted from run.py:784-1481 (Phase 1 refactor). Functions are CLI-bound:
they consume argparse.Namespace, build phase configs, invoke pipelines,
and update RunStateManager.

Future refactor phases (Phase 5: pretrain, Phase 8: generator split) will
further decompose these into their phase packages. For Phase 1 they stay
together to avoid scope creep.
"""
from __future__ import annotations

import argparse

from wfcllm.orchestration.state import RunStateManager

# Copy verbatim from run.py:
# - run_phase(phase, args, state) -> int                    (line 784)
# - run_encoder(args, state) -> int                          (line 796)
# - run_watermark(args, state) -> int                        (line 881)
# - run_offline_analysis(args) -> int                        (line 1015)
# - run_extract(args, state) -> int                          (line 1048)
# - run_generate_negative(args, state) -> int                (line 1286)
# - resolve_token_channel_train_config(args) -> dict         (line 1336)
# - validate_token_channel_train_config(train_cfg) -> str?   (line 1416)
# - run_token_channel_train(args, state) -> int              (line 1426)
```

Paste each function verbatim. Keep their imports intact (they import from `wfcllm.encoder`, `wfcllm.watermark.pipeline`, `wfcllm.extract.pipeline`, etc. — those modules are unchanged).

Update the `RunState` type annotation in each signature to `RunStateManager`:

```python
# Before (from run.py)
def run_encoder(args: argparse.Namespace, state: RunState) -> int:

# After (in cli/runners.py)
def run_encoder(args: argparse.Namespace, state: RunStateManager) -> int:
```

The class is identical; only the name changes.

- [ ] **Step 2: Update `run.py` to re-export runners**

Replace lines 784-1481 in run.py with:

```python
from wfcllm.cli.runners import (
    run_phase,
    run_encoder,
    run_watermark,
    run_offline_analysis,
    run_extract,
    run_generate_negative,
    resolve_token_channel_train_config,
    validate_token_channel_train_config,
    run_token_channel_train,
)
```

- [ ] **Step 3: Run the full run.py test suite**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py tests/test_run_config.py -v
```

Expected: All pass.

- [ ] **Step 4: Commit**

```
git add wfcllm/cli/runners.py run.py
git commit -m "refactor: extract per-phase runner functions into wfcllm/cli/runners.py"
```

---

## Task 11: Build new `main()` in `wfcllm/cli/entry.py`

**Files:**
- Create: `wfcllm/cli/entry.py`
- Modify: `run.py:297-340, 673-783` (will delete `RunState`, `cmd_status`, `cmd_reset`, `get_config`, `configured_extract_input`, `is_compare_only_mode`, `should_skip_completed_phase`, `has_explicit_extract_input`, `validate_compare_only_mode`, `main`)

This is where everything comes together. The new `main()` instantiates `RunStateManager`, builds a `PhaseRegistry` populated from `cli/runners.py`, builds a `PrereqRegistry` (empty for now), and calls `PhaseOrchestrator.run()`.

- [ ] **Step 1: Append a failing test**

Append to `tests/integration/test_cli_resolution.py`:

```python
# --- entry.main tests ---

from unittest.mock import patch, MagicMock
from wfcllm.cli.entry import main, _populate_phase_registry


def test_populate_phase_registry_registers_all_phases():
    from wfcllm.orchestration.phase_registry import PhaseRegistry
    from wfcllm.orchestration.state import ALL_PHASES
    reg = PhaseRegistry()
    _populate_phase_registry(reg)
    for phase in ALL_PHASES:
        # Don't call — runners need real configs. Just verify registration.
        assert phase in reg.phases()


def test_main_status_returns_zero(tmp_path, monkeypatch, capsys):
    # Redirect run_state.json to tmp
    state_path = tmp_path / "rs.json"
    monkeypatch.setattr(
        "wfcllm.cli.entry.DEFAULT_STATE_FILE",
        state_path,
    )
    rc = main(["--status"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "阶段状态" in captured.out


def test_main_reset_returns_zero(tmp_path, monkeypatch, capsys):
    state_path = tmp_path / "rs.json"
    monkeypatch.setattr(
        "wfcllm.cli.entry.DEFAULT_STATE_FILE",
        state_path,
    )
    rc = main(["--reset"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "已重置" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v
```

Expected: 3 new tests fail with `ModuleNotFoundError: No module named 'wfcllm.cli.entry'`

- [ ] **Step 3: Implement `wfcllm/cli/entry.py`**

```python
"""WFCLLM run-time entry point.

Replaces run.py:main() and its helpers. Wires up RunStateManager + PhaseRegistry
+ PrereqRegistry + PhaseOrchestrator.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.config_resolver import load_config
from wfcllm.cli.runners import (
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


# --- compare-only mode flags (lifted from run.py) ---

COMPARE_ONLY_REQUIRED_FLAGS = (
    "compare_summary_left",
    "compare_details_left",
    "compare_summary_right",
    "compare_details_right",
    "compare_output",
)
COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS = (
    "compare_watermarked_left",
    "compare_watermarked_right",
)


def _is_compare_only_mode(args: argparse.Namespace) -> bool:
    required_present = all(
        getattr(args, flag, None) for flag in COMPARE_ONLY_REQUIRED_FLAGS
    )
    if not required_present:
        return False
    optional = tuple(
        getattr(args, flag, None) for flag in COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    )
    return not any(optional) or all(optional)


def _validate_compare_only_mode(args: argparse.Namespace) -> str | None:
    compare_flags = COMPARE_ONLY_REQUIRED_FLAGS + COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    if not any(getattr(args, flag, None) for flag in compare_flags):
        return None
    if args.phase != "extract":
        return "[错误] compare-only 模式仅支持 --phase extract"
    if not _is_compare_only_mode(args):
        return (
            "[错误] compare-only 模式要求提供左右 summary/details 和 compare-output；"
            "watermarked 必须两侧同时提供或同时省略"
        )
    return None


# --- status / reset commands ---

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


# --- phase registry population ---

def _populate_phase_registry(reg: PhaseRegistry) -> None:
    reg.register("encoder", run_encoder)
    reg.register("watermark", run_watermark)
    reg.register("extract", run_extract)
    reg.register("generate-negative", run_generate_negative)
    reg.register("token-channel-train", run_token_channel_train)


# --- main entry ---

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

    compare_only_error = _validate_compare_only_mode(args)
    if compare_only_error is not None:
        print(compare_only_error, file=sys.stderr)
        return 1

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
```

- [ ] **Step 4: Run new entry tests**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v
```

Expected: 13 tests pass.

- [ ] **Step 5: Manually verify `--status` works end-to-end**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -m wfcllm.cli.entry --status
```

Expected: prints "=== WFCLLM 阶段状态 ===" and lists all 5 phases with their status from `data/run_state.json`.

- [ ] **Step 6: Commit**

```
git add wfcllm/cli/entry.py tests/integration/test_cli_resolution.py
git commit -m "feat: add wfcllm/cli/entry.py with PhaseOrchestrator-based main()"
```

---

## Task 12: Collapse `run.py` to 3-line shim

**Files:**
- Modify: `run.py` (delete everything except 3 lines)

All logic is now in `wfcllm/cli/` and `wfcllm/orchestration/`. `run.py` becomes the public entry script.

- [ ] **Step 1: Replace `run.py` entire contents**

Open `run.py` and replace everything with:

```python
"""WFCLLM unified entry script (thin shim around wfcllm.cli.entry)."""
from wfcllm.cli.entry import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify run.py works end-to-end**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status
```

Expected: identical output to Task 11 Step 5.

- [ ] **Step 3: Run the full test suite to confirm no regression**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

Expected: All tests pass (including `tests/test_run.py` 1734 lines, which imports from `run` module).

If `tests/test_run.py` imports anything that's been moved (e.g., `from run import RunState`), update those imports to the new locations:
- `from run import RunState` → `from wfcllm.orchestration.state import RunStateManager as RunState`
- `from run import load_config` → `from wfcllm.cli.config_resolver import load_config`
- etc.

Search and replace:

```
grep -rn "from run import" tests/
grep -rn "import run" tests/
```

For each match, update the import. Do **not** rewrite test logic — just patch the imports.

- [ ] **Step 4: Re-run the full suite**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```
git add run.py tests/
git commit -m "refactor: collapse run.py to 3-line shim (cli/entry handles everything)"
```

---

## Task 13: Verify run_state.json compatibility with existing checkpoint

**Files:**
- (Read-only verification)

The refactor preserves `run_state.json` schema. Confirm explicitly: load the existing `data/run_state.json` (which has real prior runs in it) through `RunStateManager` and verify all phases report correctly.

- [ ] **Step 1: Save a backup of existing state**

```
cp data/run_state.json data/run_state.json.phase1-backup
```

- [ ] **Step 2: Verify load + status displays existing state**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status
```

Expected: shows the actual completion status from the existing file (e.g., encoder ✓, watermark ✓, etc. — matching what was there pre-refactor).

- [ ] **Step 3: Verify state file unchanged after status check**

```
diff data/run_state.json data/run_state.json.phase1-backup
```

Expected: no output (files identical — `--status` is read-only).

- [ ] **Step 4: Commit the verification artifact only if needed**

No commit for this step — it's verification only. Delete the backup:

```
rm data/run_state.json.phase1-backup
```

---

## Task 14: Smoke-test a real phase invocation

**Files:**
- (No file changes; runtime verification only)

Force-rerun one cheap phase to confirm orchestrator + runner integration works end-to-end. Pick `generate-negative` since it's optional and skippable.

- [ ] **Step 1: Check current status of generate-negative**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status | grep generate
```

Expected: shows the phase status.

- [ ] **Step 2: Invoke --status one more time to confirm idempotency**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status
```

Expected: same output.

- [ ] **Step 3: Run the orchestration test suite explicitly**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py tests/integration/test_cli_resolution.py tests/common/test_registry.py -v
```

Expected: All 26 tests pass (4 registry + 5 state + 3 phase_registry + 4 prereq + 6 orchestrator + 10 cli + 3 entry).

- [ ] **Step 4: Commit (only if any file changed; otherwise skip)**

If nothing changed, no commit needed.

---

## Task 15: Update CLAUDE.md project notes (if needed)

**Files:**
- Read: `CLAUDE.md`
- Modify: `CLAUDE.md` (only if existing rules become stale)

- [ ] **Step 1: Read current CLAUDE.md**

```
cat CLAUDE.md
```

- [ ] **Step 2: Check for stale references**

Look for any mention of:
- `run.py` (still valid; it's still the entry point)
- `RunState` (now `RunStateManager`; update if mentioned)
- `wfcllm/encoder`, `wfcllm/watermark`, `wfcllm/extract`, `wfcllm/common` (unchanged)

If anything references internal module paths that changed, update those lines.

- [ ] **Step 3: Commit only if changes were made**

```
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md after Phase 1 module reorganization"
```

If no changes needed, skip this step.

---

## Task 16: Final verification + close out Phase 1

**Files:**
- (No file changes)

- [ ] **Step 1: Verify run.py is now ≤ 10 lines**

```
wc -l run.py
```

Expected: ≤ 10 lines (target was 3-line shim with module docstring).

- [ ] **Step 2: Verify the full test suite passes**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

Expected: All tests pass. Count of new tests added: ≥26 (registry + orchestration + cli).

- [ ] **Step 3: Confirm git log shows clean commits**

```
git log --oneline -20
```

Expected: ~14 well-scoped commits since Phase 1 start (one per task, no fixup commits).

- [ ] **Step 4: Final commit (only if any housekeeping fixes)**

If everything is clean, no extra commit. Otherwise commit any housekeeping.

---

## Phase 1 Summary (after all tasks)

**Files created (10):**
- `wfcllm/common/registry.py`
- `wfcllm/orchestration/__init__.py`
- `wfcllm/orchestration/state.py`
- `wfcllm/orchestration/phase_registry.py`
- `wfcllm/orchestration/prereq.py`
- `wfcllm/orchestration/pipeline.py`
- `wfcllm/cli/__init__.py`
- `wfcllm/cli/arguments.py`
- `wfcllm/cli/config_resolver.py`
- `wfcllm/cli/runners.py`
- `wfcllm/cli/entry.py`
- `tests/common/test_registry.py`
- `tests/integration/__init__.py`
- `tests/integration/test_orchestration.py`
- `tests/integration/test_cli_resolution.py`

**Files modified:**
- `run.py` (1482 lines → ~4 lines)
- `tests/test_run.py` (import updates only; not split — that's Phase 9)

**Files deleted:**
- None (all old code was lifted, not removed)

**Behavior preserved:**
- `run_state.json` schema unchanged
- CLI flags unchanged
- All existing tests still pass
- Existing trained checkpoints unchanged

**New behaviors:**
- Generic `Registry[T]` available for Phase 6 adapters
- `PrereqRegistry` framework available; no prereqs registered yet (registered by Phase 3 / 4)
- Tests organized under `tests/integration/` for cross-phase tests

**What this unlocks:**
- Phase 2 (`evaluation/` package) can register evaluation entry points cleanly
- Phase 3 (`adaptive_gamma/`) can register entropy-profile prereq for watermark phase
- Phase 4 (`extract/calibration/`) can register threshold prereq for extract phase
- Phase 5 (`pretrain/`) can add a new phase `pretrain` via `PhaseRegistry`
- Phase 6 (`lang/`, `datasets/`) can use `Registry[T]` for adapter plug-in points

---

## Notes for the implementer

- **Do not split `tests/test_run.py` yet** — that's Phase 9. For now, just patch import paths if needed.
- **Do not move `run_encoder`/`run_watermark`/etc. out of `cli/runners.py` yet** — they get further decomposed in Phase 5 (encoder → pretrain) and Phase 8 (watermark internal split). Phase 1 just puts them in a clean home.
- **`tests/integration/test_cli_resolution.py`** also covers entry.py; that's fine — they're related concerns.
- **If a copy-paste introduces a bug**, the symptom will be a test failure in `tests/test_run.py` or `tests/test_run_config.py` after Task 8/9/10. Diff against the pre-Phase 1 commit to find the typo.
- **Don't optimize while copying.** Even if a function looks ugly or has a docstring you'd improve, leave it byte-identical. Polish in a separate commit later.
