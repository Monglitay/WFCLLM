from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from wfcllm.cli import runners
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.pipeline import PhaseOrchestrator
from wfcllm.orchestration.state import ALL_PHASES, PHASES, RunStateManager


EXPECTED_PHASES = [
    "encoder",
    "gate-data",
    "gate-train",
    "generate",
    "calibrate",
    "detect",
    "report",
    "audit",
]


def _config() -> dict[str, object]:
    return {
        "method": {"name": "gated_semantic_window_v1"},
        "runtime": {"default_phases": list(EXPECTED_PHASES)},
    }


def _args(phase: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(phase=phase, _config_cache=_config())


def test_phase_source_of_truth_is_exact_gate_only_chain() -> None:
    assert PHASES == EXPECTED_PHASES
    assert ALL_PHASES is PHASES


def test_new_state_contains_only_current_phases(tmp_path: Path) -> None:
    state = RunStateManager(tmp_path / "state.json")

    assert list(state.status()) == EXPECTED_PHASES
    assert all(not row["done"] for row in state.status().values())
    assert not hasattr(state, "reset")


def test_state_persists_current_phase_metadata(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    RunStateManager(path).mark_done("encoder", checkpoint="encoder/best_model.pt")

    state = RunStateManager(path)
    assert state.is_done("encoder")
    assert state.get("encoder", "checkpoint") == "encoder/best_model.pt"
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"generate": {"done": False}},
        {
            **{phase: {"done": False} for phase in EXPECTED_PHASES},
            "retired-phase": {"done": False},
        },
        {
            **{phase: {"done": False} for phase in EXPECTED_PHASES},
            "generate": {"done": "yes"},
        },
    ],
)
def test_state_rejects_noncurrent_schema(
    tmp_path: Path, payload: object
) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run state"):
        RunStateManager(path)


def test_two_state_instances_merge_distinct_phase_updates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    first = RunStateManager(path)
    stale = RunStateManager(path)

    first.mark_done("encoder", owner="first")
    stale.mark_done("gate-data", owner="second")

    state = RunStateManager(path)
    assert state.get("encoder", "owner") == "first"
    assert state.get("gate-data", "owner") == "second"


def test_orchestrator_runs_current_chain_once_in_order(tmp_path: Path) -> None:
    state = RunStateManager(tmp_path / "state.json")
    registry = PhaseRegistry()
    observed: list[str] = []

    for phase in EXPECTED_PHASES:
        def runner(
            _args: argparse.Namespace,
            current_state: RunStateManager,
            *,
            phase_name: str = phase,
        ) -> int:
            observed.append(phase_name)
            current_state.mark_done(phase_name)
            return 0

        registry.register(phase, runner)

    result = PhaseOrchestrator(
        state,
        registry,
        resolved_config=_config(),
    ).run(_args())

    assert result == 0
    assert observed == EXPECTED_PHASES
    assert all(row["done"] for row in state.status().values())


def test_orchestrator_refuses_completed_phase_instead_of_resuming(
    tmp_path: Path,
) -> None:
    state = RunStateManager(tmp_path / "state.json")
    state.mark_done("encoder")
    registry = PhaseRegistry()
    registry.register("encoder", lambda _args, _state: 0)

    with pytest.raises(ValueError, match="Fresh Reproduction Run"):
        PhaseOrchestrator(
            state,
            registry,
            resolved_config=_config(),
        ).run(_args("encoder"))


def test_encoder_rejects_existing_pending_state_file(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({phase: {"done": False} for phase in EXPECTED_PHASES}),
        encoding="utf-8",
    )
    state = RunStateManager(state_path)
    registry = PhaseRegistry()
    registry.register("encoder", lambda _args, _state: 0)
    args = _args("encoder")
    args.run_dir = str(tmp_path / "new-run")

    with pytest.raises(ValueError, match="new state file"):
        PhaseOrchestrator(state, registry, resolved_config=_config()).run(args)


def test_encoder_rejects_existing_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    state = RunStateManager(tmp_path / "new-state.json")
    registry = PhaseRegistry()
    registry.register("encoder", lambda _args, _state: 0)
    args = _args("encoder")
    args.run_dir = str(run_dir)

    with pytest.raises(ValueError, match="new run directory"):
        PhaseOrchestrator(state, registry, resolved_config=_config()).run(args)


def test_phase_requires_immediate_current_run_predecessor(
    tmp_path: Path,
) -> None:
    state = RunStateManager(tmp_path / "state.json")
    registry = PhaseRegistry()
    registry.register("generate", lambda _args, _state: 0)

    with pytest.raises(ValueError, match="completed gate-train"):
        PhaseOrchestrator(
            state,
            registry,
            resolved_config=_config(),
        ).run(_args("generate"))


def test_orchestrator_rejects_any_alternate_phase_sequence(
    tmp_path: Path,
) -> None:
    config = _config()
    config["runtime"] = {"default_phases": EXPECTED_PHASES[:-1]}
    orchestrator = PhaseOrchestrator(
        RunStateManager(tmp_path / "state.json"),
        PhaseRegistry(),
        resolved_config=config,
    )

    with pytest.raises(ValueError, match="Gate-only phase chain"):
        orchestrator.resolve_phase_sequence()


def test_non_gate_method_is_rejected_before_dispatch(tmp_path: Path) -> None:
    config = _config()
    config["method"] = {"name": "removed_method"}
    args = argparse.Namespace(phase="encoder", _config_cache=config)

    with pytest.raises(ValueError, match="only gated_semantic_window_v1"):
        PhaseOrchestrator(
            RunStateManager(tmp_path / "state.json"),
            PhaseRegistry(),
            resolved_config=config,
        ).run(args)


def test_report_requires_current_detection_and_calibration_artifacts(
    tmp_path: Path,
) -> None:
    config = _config()
    config["generation"] = {"language": "python", "dataset": "humaneval"}
    args = argparse.Namespace(
        _config_cache=config,
        run_dir=str(tmp_path / "run"),
        generation_model_path="/models/generator",
        _gated_report_metrics=None,
        _posthoc_pass_report=None,
    )

    with pytest.raises(ValueError, match="positive detection details"):
        runners.run_report(args, RunStateManager(tmp_path / "state.json"))


def test_audit_fails_when_a_required_current_artifact_is_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    final_code = run_dir / "inputs" / "final_code.jsonl"
    final_code.parent.mkdir(parents=True)
    final_code.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def f():\n",
                "final_code": "def f():\n    return 1\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        _config_cache=_config(),
        run_dir=str(run_dir),
    )

    with pytest.raises(ValueError, match="required audit artifact is missing"):
        runners.run_audit(args, RunStateManager(tmp_path / "state.json"))
