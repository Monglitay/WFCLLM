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


def test_main_compare_only_mode_forces_phase_rerun(tmp_path, monkeypatch, capsys):
    """Compare-only mode should bypass the 'phase already done → skip' optimization."""
    state_path = tmp_path / "rs.json"
    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", state_path)

    # Pre-mark legacy-extract as done in the state file
    from wfcllm.orchestration.state import RunStateManager
    RunStateManager(path=state_path).mark_done("legacy-extract", details_file="prior.jsonl")

    # Replace the legacy extract runner with a sentinel so we can assert it WAS called
    called = []
    def sentinel_runner(args, state):
        called.append("legacy-extract")
        return 0

    monkeypatch.setattr("wfcllm.cli.entry.run_legacy_extract", sentinel_runner)

    # Construct compare-only CLI invocation
    argv = [
        "--phase", "legacy-extract",
        "--legacy",
        "--compare-summary-left", str(tmp_path / "sl.json"),
        "--compare-details-left", str(tmp_path / "dl.jsonl"),
        "--compare-summary-right", str(tmp_path / "sr.json"),
        "--compare-details-right", str(tmp_path / "dr.jsonl"),
        "--compare-output", str(tmp_path / "out.json"),
    ]
    from wfcllm.cli.entry import main
    rc = main(argv)

    # If the fix is in place, runner gets called despite extract being done.
    assert called == ["legacy-extract"], (
        "compare-only mode failed to bypass skip — extract should have been re-run"
    )
    assert rc == 0


# --- CLI subprocess / parser tests (from test_run.py TestCLI) ---

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNNERS_PY = PROJECT_ROOT / "wfcllm" / "cli" / "runners.py"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def test_cli_subprocess_invocations_do_not_use_bare_run_py_script_name():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "run"):
            continue
        if not node.args or not isinstance(node.args[0], ast.List):
            continue
        constants = [
            elt.value
            for elt in node.args[0].elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        assert "run.py" not in constants


def test_build_parser_parses_resume_argument():
    from wfcllm.cli.arguments import build_parser

    args = build_parser().parse_args(
        ["--phase", "legacy-extract", "--legacy", "--resume", "latest"]
    )
    assert args.resume == "latest"


def test_build_parser_accepts_token_channel_flags():
    from wfcllm.cli.arguments import build_parser

    args = build_parser().parse_args(
        [
            "--phase",
            "legacy-watermark",
            "--legacy",
            "--token-channel-enabled",
            "true",
            "--token-channel-mode",
            "dual-channel",
            "--token-channel-model-path",
            "data/models/token-channel-demo",
            "--token-channel-delta",
            "1.5",
            "--token-channel-joint-threshold",
            "5.0",
        ]
    )

    assert args.token_channel_enabled is True
    assert args.token_channel_mode == "dual-channel"
    assert args.token_channel_model_path == "data/models/token-channel-demo"
    assert args.token_channel_delta == pytest.approx(1.5)
    assert args.token_channel_joint_threshold == pytest.approx(5.0)


def test_build_parser_accepts_token_channel_train_phase_and_flags():
    from wfcllm.cli.arguments import build_parser

    args = build_parser().parse_args(
        [
            "--phase",
            "legacy-token-channel-train",
            "--legacy",
            "--token-channel-cache-path",
            "data/token_channel/custom_cache.json",
            "--token-channel-model-path",
            "data/models/token-channel-demo",
            "--token-channel-context-width",
            "256",
            "--token-channel-hidden-size",
            "96",
            "--token-channel-batch-size",
            "32",
            "--token-channel-epochs",
            "4",
            "--token-channel-lr",
            "0.01",
            "--token-channel-entropy-threshold",
            "1.5",
            "--token-channel-diversity-threshold",
            "3",
            "--token-channel-split-ratio",
            "0.8",
            "--token-channel-seed",
            "7",
        ]
    )

    assert args.phase == "legacy-token-channel-train"
    assert args.token_channel_cache_path == "data/token_channel/custom_cache.json"
    assert args.token_channel_model_path == "data/models/token-channel-demo"
    assert args.token_channel_context_width == 256
    assert args.token_channel_hidden_size == 96
    assert args.token_channel_batch_size == 32
    assert args.token_channel_epochs == 4
    assert args.token_channel_lr == pytest.approx(0.01)
    assert args.token_channel_entropy_threshold == pytest.approx(1.5)
    assert args.token_channel_diversity_threshold == 3
    assert args.token_channel_split_ratio == pytest.approx(0.8)
    assert args.token_channel_seed == 7


# --- Parser tests (from test_run_config.py) ---


def test_parser_default_config():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.config == Path("configs/base_config.json")
    assert args.fpr is None


def test_parser_custom_config():
    parser = build_parser()
    args = parser.parse_args(["--config", "configs/my.json"])
    assert args.config == Path("configs/my.json")


def test_parser_accepts_adaptive_watermark_and_extract_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--gamma-strategy",
            "piecewise_quantile",
            "--entropy-profile",
            "configs/demo_profile.json",
            "--profile-id",
            "python__demo__v1",
            "--adaptive-detection-mode",
            "prefer-adaptive",
            "--strict-contract",
        ]
    )

    assert args.gamma_strategy == "piecewise_quantile"
    assert args.entropy_profile == "configs/demo_profile.json"
    assert args.profile_id == "python__demo__v1"
    assert args.adaptive_detection_mode == "prefer-adaptive"
    assert args.strict_contract is True
