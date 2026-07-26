"""Tests for orchestration components: RunStateManager."""
import argparse
import json
import subprocess
from pathlib import Path

import pytest

from wfcllm.orchestration.state import (
    ALL_PHASES,
    LEGACY_PHASES,
    OPTIONAL_PHASES,
    PHASES,
    RunStateManager,
)
from tests.integration.conftest import PROJECT_ROOT, RUN_PY, write_json, write_jsonl


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
    state.mark_done("generate", output_file="out.jsonl")
    status = state.status()
    assert set(status.keys()) == set(ALL_PHASES)
    assert status["generate"]["done"] is True
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


def _concurrent_mark_state(path: str, phase: str) -> None:
    RunStateManager(path=Path(path)).mark_done(phase, worker=phase)


def _make_args(**kwargs):
    """Build minimal argparse.Namespace for orchestrator tests."""
    defaults = dict(force=False, eval_only=False, phase=None, input=None)
    defaults.update(kwargs)
    ns = argparse.Namespace(**defaults)
    # config cache hook used by has_explicit_detect_input
    setattr(ns, "_config_cache", {"detector": {}})
    return ns


def test_orchestrator_runs_all_main_phases_when_phase_unspecified(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    ran = []

    reg = PhaseRegistry()
    for p in PHASES:
        reg.register(p, lambda a, s, name=p: (ran.append(name), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args())
    assert rc == 0
    assert ran == PHASES


def test_orchestrator_skips_completed_phase(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    state.mark_done("generate")
    ran = []
    reg = PhaseRegistry()
    for p in PHASES:
        reg.register(p, lambda a, s, name=p: (ran.append(name), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args())
    assert rc == 0
    assert ran == ["calibrate", "detect", "report", "audit"]


def test_orchestrator_force_reruns_completed_phase(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    state.mark_done("generate")
    ran = []
    reg = PhaseRegistry()
    for p in PHASES:
        reg.register(p, lambda a, s, name=p: (ran.append(name), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args(force=True))
    assert rc == 0
    assert ran == PHASES


def test_orchestrator_reruns_completed_detect_with_explicit_cli_input(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    state.mark_done("detect")
    ran = []
    reg = PhaseRegistry()
    reg.register("detect", lambda a, s: (ran.append("detect"), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args(phase="detect", input="data/runs/final_code.jsonl"))

    assert rc == 0
    assert ran == ["detect"]


def test_orchestrator_reruns_completed_detect_with_configured_input(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    state.mark_done("detect")
    ran = []
    reg = PhaseRegistry()
    reg.register("detect", lambda a, s: (ran.append("detect"), 0)[1])
    args = _make_args(phase="detect")
    setattr(args, "_config_cache", {"detector": {"input": "data/runs/final_code.jsonl"}})

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(args)

    assert rc == 0
    assert ran == ["detect"]


def test_orchestrator_fails_fast_on_nonzero(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    ran = []
    reg = PhaseRegistry()
    reg.register("generate", lambda a, s: (ran.append("generate"), 0)[1])
    reg.register("calibrate", lambda a, s: (ran.append("calibrate"), 7)[1])  # fail
    reg.register("detect", lambda a, s: (ran.append("detect"), 0)[1])
    reg.register("report", lambda a, s: (ran.append("report"), 0)[1])
    reg.register("audit", lambda a, s: (ran.append("audit"), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args())
    assert rc == 7
    assert ran == ["generate", "calibrate"]  # detect never runs


def test_orchestrator_runs_single_phase_when_specified(tmp_path):
    PrereqRegistry().clear()
    state = RunStateManager(path=tmp_path / "rs.json")
    ran = []
    reg = PhaseRegistry()
    for p in (*PHASES, "posthoc-pass-report"):
        reg.register(p, lambda a, s, name=p: (ran.append(name), 0)[1])

    orch = PhaseOrchestrator(state=state, phase_registry=reg, prereq_registry=PrereqRegistry())
    rc = orch.run(_make_args(phase="posthoc-pass-report"))
    assert rc == 0
    assert ran == ["posthoc-pass-report"]


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


def test_all_phases_includes_legacy_build_entropy_profile():
    from wfcllm.orchestration.state import ALL_PHASES, LEGACY_PHASES
    assert "legacy-build-entropy-profile" in LEGACY_PHASES
    assert "legacy-build-entropy-profile" in ALL_PHASES


def test_run_state_manager_tracks_build_entropy_profile(tmp_path):
    from wfcllm.orchestration.state import RunStateManager
    state = RunStateManager(path=tmp_path / "rs.json")
    assert state.is_done("legacy-build-entropy-profile") is False
    state.mark_done("legacy-build-entropy-profile", profile_path="data/calibration/p.json")
    assert state.is_done("legacy-build-entropy-profile") is True
    assert state.get("legacy-build-entropy-profile", "profile_path") == "data/calibration/p.json"


# ── Tests migrated from tests/test_run.py (TestRunState) ──────────────────────


def test_phases_order():
    assert PHASES == ["generate", "calibrate", "detect", "report", "audit"]
    assert OPTIONAL_PHASES == ["encoder", "posthoc-pass-report", "diagnostic-selector"]
    assert LEGACY_PHASES == [
        "legacy-watermark",
        "legacy-extract",
        "legacy-token-channel-train",
        "legacy-build-entropy-profile",
        "legacy-pretrain",
        "legacy-ablation",
    ]
    from wfcllm.orchestration.state import GATE_PHASES
    assert ALL_PHASES == PHASES + GATE_PHASES + OPTIONAL_PHASES + LEGACY_PHASES


def test_gate_phases_are_allowed_without_changing_legacy_main_phase_constant():
    from wfcllm.orchestration.state import GATE_PHASES

    assert PHASES == ["generate", "calibrate", "detect", "report", "audit"]
    assert GATE_PHASES == ["gate-data", "gate-train"]
    assert all(phase in ALL_PHASES for phase in GATE_PHASES)


def test_gate_validate_phase_is_removed_from_all_registries():
    from wfcllm.cli.arguments import build_parser
    from wfcllm.orchestration.state import ALL_PHASES, GATE_PHASES

    assert "gate-validate" not in GATE_PHASES
    assert "gate-validate" not in ALL_PHASES
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--phase", "gate-validate"])


def test_gated_method_uses_configured_fast_phase_sequence(tmp_path):
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset

    state = RunStateManager(tmp_path / "state.json")
    orchestrator = PhaseOrchestrator(
        state=state,
        phase_registry=PhaseRegistry(),
        prereq_registry=PrereqRegistry(),
        resolved_config=load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict(),
    )

    assert orchestrator.resolve_phase_sequence() == [
        "gate-data", "gate-train", "generate",
        "calibrate", "detect", "report",
    ]


def test_gated_generate_run_phase_requires_gate_bundle_before_dispatch(tmp_path):
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset

    orchestrator = PhaseOrchestrator(
        state=RunStateManager(tmp_path / "state.json"),
        phase_registry=PhaseRegistry(),
        prereq_registry=PrereqRegistry(),
        resolved_config=load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict(),
    )

    with pytest.raises(ValueError, match="gate bundle prerequisite failed"):
        orchestrator.run_phase("generate")


def test_existing_method_keeps_five_phase_sequence(tmp_path):
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME, load_method_preset

    orchestrator = PhaseOrchestrator(
        state=RunStateManager(tmp_path / "state.json"),
        phase_registry=PhaseRegistry(),
        prereq_registry=PrereqRegistry(),
        resolved_config=load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict(),
    )

    assert orchestrator.resolve_phase_sequence() == PHASES


def test_orchestrator_rejects_unknown_configured_default_phase(tmp_path):
    orchestrator = PhaseOrchestrator(
        state=RunStateManager(tmp_path / "state.json"),
        phase_registry=PhaseRegistry(),
        prereq_registry=PrereqRegistry(),
        resolved_config={"runtime": {"default_phases": ["made-up"]}},
    )

    with pytest.raises(ValueError, match="default phase"):
        orchestrator.resolve_phase_sequence()


def test_old_five_phase_state_file_remains_readable_after_gate_phase_addition(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({phase: {"done": phase == "generate"} for phase in PHASES}))

    state = RunStateManager(path)

    assert state.is_done("generate") is True
    assert state.is_done("gate-data") is False
    assert "gate-data" in state.status()


def test_completed_gate_phase_is_not_skipped_when_input_hash_changes(tmp_path):
    state = RunStateManager(tmp_path / "state.json")
    state.mark_done("gate-train", input_hash="a" * 64)
    args = _make_args(phase="gate-train")
    args._phase_input_hashes = {"gate-train": "b" * 64}
    ran = []
    registry = PhaseRegistry()
    registry.register("gate-train", lambda _args, _state: (ran.append(True), 0)[1])
    orchestrator = PhaseOrchestrator(state, registry, PrereqRegistry())

    assert orchestrator.run(args) == 0
    assert ran == [True]


@pytest.mark.parametrize("phase", ["gate-data", "gate-train"])
def test_completed_gate_phase_skips_only_when_complete_output_tree_is_unchanged(tmp_path, phase):
    import hashlib
    from wfcllm.cli.runners import _safe_tree_hash
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset

    run_dir = tmp_path / "run"
    output = run_dir / phase
    output.mkdir(parents=True)
    manifest_name = {
        "gate-data": "manifest.json",
        "gate-train": "candidate_bundle_manifest.json",
    }[phase]
    manifest = output / manifest_name
    manifest.write_text("stable")
    declared_artifact = output / "declared.bin"
    declared_artifact.write_bytes(b"original")
    output_hash = hashlib.sha256(b"stable").hexdigest()
    artifact_hash = _safe_tree_hash(output)
    input_hash = "a" * 64
    state = RunStateManager(tmp_path / "state.json")
    state.mark_done(
        phase,
        input_hash=input_hash,
        manifest_path=str(manifest),
        output_manifest_hash=output_hash,
        output_artifact_path=str(output),
        output_artifact_hash=artifact_hash,
    )
    args = _make_args(phase=phase)
    args.run_dir = str(run_dir)
    args.run_id = None
    args._config_cache = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    args._phase_input_hashes = {phase: input_hash}
    ran = []
    registry = PhaseRegistry()
    registry.register(phase, lambda _args, _state: (ran.append(True), 0)[1])
    orchestrator = PhaseOrchestrator(state, registry, PrereqRegistry())

    assert orchestrator.run(args) == 0
    assert ran == []

    declared_artifact.write_bytes(b"tampered")
    assert orchestrator._should_skip(args, phase) is False


def test_non_gated_config_cannot_use_old_gate_done_row_to_skip_runner(tmp_path):
    import hashlib
    from wfcllm.cli.runners import _safe_tree_hash, run_gate_data
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME, load_method_preset

    output = tmp_path / "old-run" / "gate-data"
    output.mkdir(parents=True)
    manifest = output / "manifest.json"
    manifest.write_text("stable")
    input_hash = "a" * 64
    state = RunStateManager(tmp_path / "state.json")
    state.mark_done(
        "gate-data",
        input_hash=input_hash,
        manifest_path=str(manifest),
        output_manifest_hash=hashlib.sha256(b"stable").hexdigest(),
        output_artifact_path=str(output),
        output_artifact_hash=_safe_tree_hash(output),
    )
    args = _make_args(phase="gate-data")
    args.run_dir = str(tmp_path / "old-run")
    args.run_id = None
    args._config_cache = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict()
    args._phase_input_hashes = {"gate-data": input_hash}
    registry = PhaseRegistry()
    registry.register("gate-data", run_gate_data)

    with pytest.raises(ValueError, match="gate phases require"):
        PhaseOrchestrator(state, registry, PrereqRegistry()).run(args)


def test_gate_state_paths_from_another_run_cannot_authorize_skip(tmp_path):
    from wfcllm.cli.runners import _safe_file_hash, _stable_tree_hash
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset

    run_a = tmp_path / "run-a"
    output_a = run_a / "gate-data"
    output_a.mkdir(parents=True)
    manifest_a = output_a / "manifest.json"
    manifest_a.write_text("stable")
    input_hash = "a" * 64
    state = RunStateManager(tmp_path / "state.json")
    state.mark_done(
        "gate-data",
        input_hash=input_hash,
        manifest_path=str(manifest_a),
        output_manifest_hash=_safe_file_hash(manifest_a),
        output_artifact_path=str(output_a),
        output_artifact_hash=_stable_tree_hash(output_a),
    )
    args = _make_args(phase="gate-data")
    args.run_dir = str(tmp_path / "run-b")
    args.run_id = None
    args._config_cache = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    args._phase_input_hashes = {"gate-data": input_hash}
    ran = []
    registry = PhaseRegistry()
    registry.register("gate-data", lambda _args, _state: (ran.append(True), 0)[1])

    assert PhaseOrchestrator(state, registry, PrereqRegistry()).run(args) == 0
    assert ran == [True]


def test_reset_clears_all(tmp_path):
    state_file = tmp_path / "run_state.json"
    state = RunStateManager(state_file)
    state.mark_done("encoder")
    state.reset()
    assert state.is_done("encoder") is False


def test_status_dict(tmp_path):
    state_file = tmp_path / "run_state.json"
    state = RunStateManager(state_file)
    state.mark_done("encoder", checkpoint="x.pt")
    status = state.status()
    assert status["encoder"]["done"] is True
    assert status["generate"]["done"] is False
    assert status["legacy-watermark"]["done"] is False


def test_two_state_instances_merge_instead_of_stale_overwrite(tmp_path):
    path = tmp_path / "state.json"
    first = RunStateManager(path)
    stale = RunStateManager(path)

    first.mark_done("generate", owner="first")
    stale.mark_done("calibrate", owner="second")

    reloaded = RunStateManager(path)
    assert reloaded.is_done("generate") is True
    assert reloaded.get("generate", "owner") == "first"
    assert reloaded.is_done("calibrate") is True
    assert reloaded.get("calibrate", "owner") == "second"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"unknown-phase": {"done": False}},
        {"generate": {"done": "yes"}},
        {"generate": []},
    ],
)
def test_state_rejects_malformed_or_unknown_root_schema(tmp_path, payload):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="run state"):
        RunStateManager(path)


@pytest.mark.parametrize("concurrency_round", range(8))
def test_multiprocess_state_updates_do_not_lose_distinct_phases(
    tmp_path, concurrency_round
):
    import multiprocessing

    path = tmp_path / "state.json"
    phases = ["generate", "calibrate", "detect", "report", "audit"]
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_concurrent_mark_state, args=(str(path), phase))
        for phase in phases
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
        assert worker.exitcode == 0

    state = RunStateManager(path)
    assert all(state.is_done(phase) for phase in phases)
    assert path.stat().st_mode & 0o777 == 0o600


# ── Tests migrated from tests/test_run.py (TestCLI status/reset/offline) ──────


def test_status_exits_zero():
    result = subprocess.run(
        ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--status"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "encoder" in result.stdout


def test_reset_exits_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--reset"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "重置" in result.stdout or "reset" in result.stdout.lower()


def test_unknown_phase_exits_nonzero():
    result = subprocess.run(
        ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--phase", "invalid"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_run_offline_analysis_writes_json_report(tmp_path):
    from wfcllm.cli.runners import run_offline_analysis

    left_summary = tmp_path / "left_summary.json"
    right_summary = tmp_path / "right_summary.json"
    left_details = tmp_path / "left_details.jsonl"
    right_details = tmp_path / "right_details.jsonl"
    left_watermarked = tmp_path / "left_watermarked.jsonl"
    right_watermarked = tmp_path / "right_watermarked.jsonl"
    report_output = tmp_path / "offline_analysis.json"

    write_json(
        left_summary,
        {
            "dataset": "HumanEval",
            "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
            "summary": {"watermark_rate": 1.0},
        },
    )
    write_json(
        right_summary,
        {
            "dataset": "HumanEval",
            "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            "summary": {"watermark_rate": 0.0},
        },
    )
    write_jsonl(
        left_details,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": True,
                "z_score": 2.4,
                "p_value": 0.02,
                "independent_blocks": 8,
                "hits": 6,
            }
        ],
    )
    write_jsonl(
        right_details,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": False,
                "z_score": 1.0,
                "p_value": 0.14,
                "independent_blocks": 8,
                "hits": 5,
            }
        ],
    )
    write_jsonl(
        left_watermarked,
        [
            {
                "id": "HumanEval/0",
                "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
                "total_blocks": 8,
                "embedded_blocks": 6,
                "failed_blocks": 0,
                "fallback_blocks": 0,
                "embed_rate": 0.75,
            }
        ],
    )
    write_jsonl(
        right_watermarked,
        [
            {
                "id": "HumanEval/0",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
                "total_blocks": 8,
                "embedded_blocks": 5,
                "failed_blocks": 1,
                "fallback_blocks": 0,
                "embed_rate": 0.625,
            }
        ],
    )

    args = argparse.Namespace(
        compare_summary_left=str(left_summary),
        compare_details_left=str(left_details),
        compare_watermarked_left=str(left_watermarked),
        compare_summary_right=str(right_summary),
        compare_details_right=str(right_details),
        compare_watermarked_right=str(right_watermarked),
        compare_output=str(report_output),
    )

    rc = run_offline_analysis(args)

    assert rc == 0
    assert report_output.exists()
    report = json.loads(report_output.read_text(encoding="utf-8"))
    assert set(report) == {
        "compatibility",
        "parameter_diff",
        "detail_delta",
        "embedding_delta",
        "anomalies",
        "regression_classification",
    }


# ── Tests migrated from tests/test_run_config.py (config loading) ─────────────


def test_load_config_returns_dict(tmp_path):
    from wfcllm.cli.config_resolver import load_config
    cfg = {"encoder": {"lr": 0.001}, "watermark": {}, "extract": {}}
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg))
    result = load_config(f)
    assert result["encoder"]["lr"] == 0.001


def test_load_config_missing_phase_ok(tmp_path):
    from wfcllm.cli.config_resolver import load_config
    cfg = {"encoder": {"lr": 0.001}}
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg))
    result = load_config(f)
    assert result.get("watermark", {}) == {}


def test_load_config_file_not_found(tmp_path):
    from wfcllm.cli.config_resolver import load_config
    result = load_config(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_config_invalid_json(tmp_path):
    from wfcllm.cli.config_resolver import load_config
    f = tmp_path / "bad.json"
    f.write_text("{ not valid json }")
    with pytest.raises(SystemExit):
        load_config(f)
