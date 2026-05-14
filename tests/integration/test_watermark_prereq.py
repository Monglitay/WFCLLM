"""Tests for the watermark phase's entropy-profile prereq (Phase 3)."""
from __future__ import annotations

import argparse
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
    importlib.reload(ag)
    yield
    PrereqRegistry().clear()


def test_adaptive_gamma_import_registers_watermark_prereq():
    reg = PrereqRegistry()
    by_phase = reg._by_phase
    assert "watermark" in by_phase
    names = [p.name for p in by_phase["watermark"]]
    assert "entropy-profile" in names


def test_check_returns_true_when_adaptive_gamma_disabled():
    reg = PrereqRegistry()
    prereq = next(p for p in reg._by_phase["watermark"] if p.name == "entropy-profile")
    config = {"watermark": {"adaptive_gamma": {"enabled": False}}}
    assert prereq.check(config) is True


def test_check_returns_true_when_profile_file_exists(tmp_path):
    profile = tmp_path / "p.json"
    profile.write_text("{}", encoding="utf-8")
    reg = PrereqRegistry()
    prereq = next(p for p in reg._by_phase["watermark"] if p.name == "entropy-profile")
    config = {
        "watermark": {
            "adaptive_gamma": {"enabled": True, "profile_path": str(profile)},
        }
    }
    assert prereq.check(config) is True


def test_check_returns_false_when_profile_missing(tmp_path):
    reg = PrereqRegistry()
    prereq = next(p for p in reg._by_phase["watermark"] if p.name == "entropy-profile")
    config = {
        "watermark": {
            "adaptive_gamma": {"enabled": True, "profile_path": str(tmp_path / "missing.json")},
        }
    }
    assert prereq.check(config) is False


def test_fix_invokes_build_entropy_profile_via_runner():
    reg = PrereqRegistry()
    prereq = next(p for p in reg._by_phase["watermark"] if p.name == "entropy-profile")

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
    assert captured["phase"] == "build-entropy-profile"


def test_fix_raises_when_runner_returns_nonzero():
    reg = PrereqRegistry()
    prereq = next(p for p in reg._by_phase["watermark"] if p.name == "entropy-profile")

    class FailingRunner:
        def dispatch_phase(self, name, args):
            return 7

    config = {"watermark": {"adaptive_gamma": {"enabled": True}}}
    with pytest.raises(RuntimeError, match="build-entropy-profile"):
        prereq.fix(config, FailingRunner())
