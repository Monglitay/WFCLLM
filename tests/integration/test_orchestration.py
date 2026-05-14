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
