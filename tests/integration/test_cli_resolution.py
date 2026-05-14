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


# --- config_resolver tests ---

import json
from pathlib import Path
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


# --- entry.main tests ---

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
