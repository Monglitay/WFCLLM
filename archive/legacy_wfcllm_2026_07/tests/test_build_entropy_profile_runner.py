"""Tests for run_build_entropy_profile runner (Phase 3)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from wfcllm.cli.runners import run_build_entropy_profile
from wfcllm.orchestration.state import RunStateManager


def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for the runner under test."""
    defaults = dict(
        config=Path("/nonexistent/cfg.json"),  # forces empty config
        build_profile_input_log=None,
        build_profile_output=None,
        build_profile_language=None,
        build_profile_model_family=None,
        build_profile_strategy=None,
        build_profile_id=None,
    )
    defaults.update(overrides)
    ns = argparse.Namespace(**defaults)
    setattr(ns, "_config_cache", {})
    return ns


def test_run_build_entropy_profile_writes_artifact_and_marks_state(tmp_path):
    log_path = tmp_path / "wm.log"
    log_path.write_text(
        "\n".join(f"entropy={v:.4f}" for v in (0.1, 0.2, 0.3, 0.4, 0.5)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "profile.json"
    state = RunStateManager(path=tmp_path / "rs.json")

    args = _make_args(
        build_profile_input_log=str(log_path),
        build_profile_output=str(output),
        build_profile_language="python",
        build_profile_model_family="codet5-base",
    )
    rc = run_build_entropy_profile(args, state)
    assert rc == 0
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["language"] == "python"
    assert state.is_done("build-entropy-profile") is True
    assert state.get("build-entropy-profile", "profile_path") == str(output)


def test_run_build_entropy_profile_falls_back_to_config(tmp_path):
    log_path = tmp_path / "wm.log"
    log_path.write_text("entropy=0.5\n", encoding="utf-8")
    output = tmp_path / "profile.json"
    state = RunStateManager(path=tmp_path / "rs.json")
    args = _make_args()
    args._config_cache = {
        "adaptive_gamma": {
            "input_log": str(log_path),
            "output": str(output),
            "language": "python",
            "model_family": "codet5-base",
            "strategy": "piecewise_quantile",
        }
    }
    rc = run_build_entropy_profile(args, state)
    assert rc == 0
    assert output.exists()


def test_run_build_entropy_profile_returns_error_when_required_fields_missing(tmp_path, capsys):
    state = RunStateManager(path=tmp_path / "rs.json")
    args = _make_args()  # nothing supplied
    rc = run_build_entropy_profile(args, state)
    assert rc == 1
    assert "[错误]" in capsys.readouterr().err
