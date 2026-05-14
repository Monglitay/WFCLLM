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
