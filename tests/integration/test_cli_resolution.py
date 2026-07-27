from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.cli.arguments import DEFAULT_CONFIG_FILE, build_parser
from wfcllm.cli.config_resolver import load_config, resolve_method_config
from wfcllm.cli.entry import (
    _populate_phase_registry,
    _validate_fresh_run_invocation,
    main,
)
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.state import PHASES


ROOT = Path(__file__).resolve().parents[2]


def test_parser_exposes_only_current_phases_and_runtime_inputs() -> None:
    parser = build_parser()

    for phase in PHASES:
        assert parser.parse_args(["--phase", phase]).phase == phase
    args = parser.parse_args(
        [
            "--generation-model-path",
            "models/generator",
            "--rewrite-model-path",
            "models/rewriter",
            "--semantic-encoder-model-path",
            "models/encoder",
            "--gate-base-model-path",
            "models/gate",
        ]
    )
    assert args.config == DEFAULT_CONFIG_FILE
    assert args.rewrite_model_path == "models/rewriter"


@pytest.mark.parametrize(
    "removed_option",
    [
        "--run-id",
        "--sample-limit",
        "--input",
        "--calibration",
        "--positive-details",
        "--negative-details",
        "--detector-device",
    ],
)
def test_parser_rejects_layout_overrides_and_removed_runtime_options(
    removed_option: str,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([removed_option, "removed"])


def test_phase_registry_contains_exact_current_chain() -> None:
    registry = PhaseRegistry()
    _populate_phase_registry(registry)

    assert registry.phases() == PHASES


def test_required_config_loader_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file is missing"):
        load_config(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_config(invalid)

    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_config(non_object)


def test_default_config_resolves_to_full_gate_profile() -> None:
    config = resolve_method_config(load_config(ROOT / DEFAULT_CONFIG_FILE))

    assert config["method"]["name"] == "gated_semantic_window_v1"
    assert config["generation"]["language"] == "python"
    assert config["generation"]["dataset"] == "humaneval"
    assert config["experiment"]["profile"] == "full"
    assert config["runtime"]["default_phases"] == PHASES


def test_status_uses_temporary_state_and_lists_only_current_phases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["--status", "--state-file", str(tmp_path / "state.json")])

    assert result == 0
    output = capsys.readouterr().out
    assert [line.split(":", 1)[0] for line in output.splitlines()[1:]] == PHASES


def test_missing_config_returns_nonzero_without_creating_run_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    result = main(
        [
            "--config",
            str(tmp_path / "missing.json"),
            "--run-dir",
            str(run_dir),
            "--state-file",
            str(tmp_path / "state.json"),
        ]
    )

    assert result == 1
    assert not run_dir.exists()


def test_config_loader_reads_utf8_json_object(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"method": {"name": "gated_semantic_window_v1"}}),
        encoding="utf-8",
    )

    assert load_config(path)["method"]["name"] == "gated_semantic_window_v1"


def test_direct_encoder_requires_new_state_and_run_paths(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    run_dir = tmp_path / "run"
    args = build_parser().parse_args(
        [
            "--phase",
            "encoder",
            "--state-file",
            str(state_path),
            "--run-dir",
            str(run_dir),
        ]
    )
    _validate_fresh_run_invocation(args, state_path)

    state_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="new state file"):
        _validate_fresh_run_invocation(args, state_path)


def test_direct_later_phase_requires_existing_state_and_run_paths(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    run_dir = tmp_path / "run"
    args = build_parser().parse_args(
        [
            "--phase",
            "gate-data",
            "--state-file",
            str(state_path),
            "--run-dir",
            str(run_dir),
        ]
    )

    with pytest.raises(ValueError, match="existing state file"):
        _validate_fresh_run_invocation(args, state_path)
    state_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existing run directory"):
        _validate_fresh_run_invocation(args, state_path)
