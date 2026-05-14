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


# --- dispatch_phase tests ---

def test_dispatch_phase_invokes_registered_runner(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    captured = {}

    def fake_runner(args, st):
        captured["args"] = args
        captured["state"] = st
        return 0

    reg = PhaseRegistry()
    reg.register("build-entropy-profile", fake_runner)
    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.dispatch_phase("build-entropy-profile", _make_args())
    assert rc == 0
    assert captured["state"] is state


def test_dispatch_phase_unknown_phase_raises(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    reg = PhaseRegistry()
    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    with pytest.raises(KeyError, match="unknown phase"):
        orch.dispatch_phase("nonexistent", _make_args())


def test_dispatch_phase_propagates_nonzero_return(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    reg = PhaseRegistry()
    reg.register("build-entropy-profile", lambda a, s: 5)
    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.dispatch_phase("build-entropy-profile", _make_args())
    assert rc == 5


def test_dispatch_phase_skips_prereq_check(tmp_path):
    """dispatch_phase is for in-flight chaining; it must not re-trigger prereqs (would loop)."""
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    reg = PhaseRegistry()
    reg.register("build-entropy-profile", lambda a, s: 0)

    prereq_invocations = []
    PrereqRegistry().register("build-entropy-profile", Prereq(
        name="should-not-fire",
        check=lambda cfg: (prereq_invocations.append("checked"), False)[1],
        fix=lambda cfg, runner: prereq_invocations.append("fixed"),
    ))

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    orch.dispatch_phase("build-entropy-profile", _make_args())
    assert prereq_invocations == []
    PrereqRegistry().clear()


def test_all_phases_includes_build_entropy_profile():
    from wfcllm.orchestration.state import ALL_PHASES, OPTIONAL_PHASES
    assert "build-entropy-profile" in OPTIONAL_PHASES
    assert "build-entropy-profile" in ALL_PHASES


def test_run_state_manager_tracks_build_entropy_profile(tmp_path):
    from wfcllm.orchestration.state import RunStateManager
    state = RunStateManager(path=tmp_path / "rs.json")
    assert state.is_done("build-entropy-profile") is False
    state.mark_done("build-entropy-profile", profile_path="data/calibration/p.json")
    assert state.is_done("build-entropy-profile") is True
    assert state.get("build-entropy-profile", "profile_path") == "data/calibration/p.json"
