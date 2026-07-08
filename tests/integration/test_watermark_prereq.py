"""Tests for the watermark phase's entropy-profile prereq (Phase 3)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from wfcllm.orchestration.prereq import PrereqRegistry


@pytest.fixture(autouse=True)
def _clear_prereqs():
    PrereqRegistry().clear()
    # Re-import the subpackage to re-register the prereq under a fresh registry.
    import importlib

    import wfcllm.watermark.adaptive_gamma as ag
    if hasattr(ag, "_PREREQ_REGISTERED"):
        ag._PREREQ_REGISTERED = False
    PrereqRegistry().clear()
    importlib.reload(ag)
    yield
    PrereqRegistry().clear()


def _entropy_prereq(phase: str = "legacy-watermark"):
    reg = PrereqRegistry()
    return next(p for p in reg._by_phase[phase] if p.name == "entropy-profile")


def test_cli_entry_import_registers_adaptive_gamma_prereqs_in_fresh_process():
    script = "\n".join(
        [
            "from wfcllm.cli.entry import main",
            "from wfcllm.orchestration.prereq import PrereqRegistry",
            "print('\\n'.join(sorted(PrereqRegistry()._by_phase)))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )

    phases = set(result.stdout.splitlines())
    assert "legacy-watermark" in phases
    assert "watermark" in phases


def test_adaptive_gamma_import_registers_legacy_watermark_prereq():
    reg = PrereqRegistry()
    by_phase = reg._by_phase
    assert "legacy-watermark" in by_phase
    names = [p.name for p in by_phase["legacy-watermark"]]
    assert "entropy-profile" in names


def test_adaptive_gamma_keeps_old_watermark_prereq_for_compatibility():
    reg = PrereqRegistry()
    by_phase = reg._by_phase
    assert "watermark" in by_phase
    names = [p.name for p in by_phase["watermark"]]
    assert "entropy-profile" in names


def test_check_returns_true_when_adaptive_gamma_disabled():
    prereq = _entropy_prereq()
    config = {"watermark": {"adaptive_gamma": {"enabled": False}}}
    assert prereq.check(config) is True


def test_check_returns_true_when_profile_file_exists(tmp_path):
    profile = tmp_path / "p.json"
    profile.write_text("{}", encoding="utf-8")
    prereq = _entropy_prereq()
    config = {
        "watermark": {
            "adaptive_gamma": {"enabled": True, "profile_path": str(profile)},
        }
    }
    assert prereq.check(config) is True


def test_check_returns_false_when_profile_missing(tmp_path):
    prereq = _entropy_prereq()
    config = {
        "watermark": {
            "adaptive_gamma": {"enabled": True, "profile_path": str(tmp_path / "missing.json")},
        }
    }
    assert prereq.check(config) is False


def test_legacy_fix_invokes_legacy_build_entropy_profile_via_runner():
    prereq = _entropy_prereq()

    captured = {}

    class FakeRunner:
        def dispatch_phase(self, name, args):
            captured["phase"] = name
            captured["args"] = args
            return 0

    config = {"watermark": {"adaptive_gamma": {"enabled": True}}}
    fake_args = argparse.Namespace()
    setattr(fake_args, "_config_cache", config)
    # Real fixers receive (config, runner); a real orchestrator does not pass args
    # to ensure_satisfied. Our prereq must fall back to a sensible default args object.
    prereq.fix(config, FakeRunner())
    assert captured["phase"] == "legacy-build-entropy-profile"
    assert captured["args"].build_profile_input_log is None
    assert captured["args"]._config_cache == {"adaptive_gamma": {"enabled": True}}


def test_legacy_fix_dispatches_build_runner_with_adaptive_gamma_config(tmp_path):
    from wfcllm.cli.runners import run_legacy_build_entropy_profile
    from wfcllm.orchestration.phase_registry import PhaseRegistry
    from wfcllm.orchestration.pipeline import PhaseOrchestrator
    from wfcllm.orchestration.state import RunStateManager

    prereq = _entropy_prereq()
    log_path = tmp_path / "wm.log"
    log_path.write_text("entropy=0.25\nentropy=0.75\n", encoding="utf-8")
    output_path = tmp_path / "profile.json"
    config = {
        "watermark": {
            "adaptive_gamma": {
                "enabled": True,
                "input_log": str(log_path),
                "output": str(output_path),
                "language": "python",
                "model_family": "codet5-base",
            },
        },
    }
    state = RunStateManager(tmp_path / "run_state.json")
    phase_registry = PhaseRegistry()
    phase_registry.register("legacy-build-entropy-profile", run_legacy_build_entropy_profile)
    orchestrator = PhaseOrchestrator(
        state=state,
        phase_registry=phase_registry,
        prereq_registry=PrereqRegistry(),
    )

    prereq.fix(config, orchestrator)

    assert output_path.exists()
    assert state.is_done("legacy-build-entropy-profile") is True
    assert state.get("legacy-build-entropy-profile", "profile_path") == str(output_path)


def test_legacy_fix_raises_when_runner_returns_nonzero():
    prereq = _entropy_prereq()

    class FailingRunner:
        def dispatch_phase(self, name, args):
            return 7

    config = {"watermark": {"adaptive_gamma": {"enabled": True}}}
    with pytest.raises(RuntimeError, match="legacy-build-entropy-profile"):
        prereq.fix(config, FailingRunner())


def test_old_watermark_fix_still_targets_old_build_entropy_profile():
    prereq = _entropy_prereq("watermark")
    captured = {}

    class FakeRunner:
        def dispatch_phase(self, name, args):
            captured["phase"] = name
            captured["args"] = args
            return 0

    config = {"watermark": {"adaptive_gamma": {"enabled": True}}}
    prereq.fix(config, FakeRunner())
    assert captured["phase"] == "build-entropy-profile"
