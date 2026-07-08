from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.entry import _populate_phase_registry
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.state import PHASES, RunStateManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_phase_list_is_new_wfcllm_mainline() -> None:
    assert PHASES == ["generate", "calibrate", "detect", "report", "audit"]


def test_parser_accepts_new_official_phases() -> None:
    parser = build_parser()

    for phase in [
        "generate",
        "calibrate",
        "detect",
        "report",
        "audit",
        "posthoc-pass-report",
        "diagnostic-selector",
    ]:
        args = parser.parse_args(["--phase", phase])
        assert args.phase == phase


def test_parser_does_not_expose_allow_quality_proxy() -> None:
    parser = build_parser()

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--allow-quality-proxy" not in option_strings


def test_parser_accepts_new_official_cli_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--phase",
            "detect",
            "--run-id",
            "run-007",
            "--run-dir",
            "data/runs/run-007",
            "--input",
            "final_code.jsonl",
            "--negative-input",
            "negative_final_code.jsonl",
            "--calibration",
            "calibration.json",
            "--positive-details",
            "positive_details.jsonl",
            "--negative-details",
            "negative_details.jsonl",
            "--diagnostic-only",
        ]
    )

    assert args.run_id == "run-007"
    assert args.run_dir == "data/runs/run-007"
    assert args.input == "final_code.jsonl"
    assert args.negative_input == "negative_final_code.jsonl"
    assert args.calibration == "calibration.json"
    assert args.positive_details == "positive_details.jsonl"
    assert args.negative_details == "negative_details.jsonl"
    assert args.diagnostic_only is True


def test_phase_registry_registers_new_mainline_runners() -> None:
    registry = PhaseRegistry()
    _populate_phase_registry(registry)

    for phase in ["generate", "calibrate", "detect", "report", "audit"]:
        assert callable(registry.get(phase))


def test_minimal_mainline_runners_mark_state_and_print_banners(tmp_path, capsys) -> None:
    from wfcllm.cli.runners import (
        run_audit,
        run_calibrate,
        run_detect,
        run_generate,
        run_report,
    )
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME

    state = RunStateManager(tmp_path / "run_state.json")
    args = argparse.Namespace()

    assert run_generate(args, state) == 0
    assert state.is_done("generate") is True
    assert state.get("generate", "method") == EVIDENCE_RETRY_SEED7X3_NAME

    for phase, runner in [
        ("calibrate", run_calibrate),
        ("detect", run_detect),
        ("report", run_report),
        ("audit", run_audit),
    ]:
        assert runner(args, state) == 0
        assert state.is_done(phase) is True

    output = capsys.readouterr().out
    for phase in ["generate", "calibrate", "detect", "report", "audit"]:
        assert f"=== WFCLLM {phase} ===" in output


def test_wfcllm_evidence_retry_config_matches_canonical_preset() -> None:
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME, load_method_preset

    config = _read_json("configs/wfcllm/evidence_retry_seed7x3.json")
    assert config == load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict()


def test_wfcllm_detector_proxy_penalized_alpha04_config_uses_mainline_detector() -> None:
    config = _read_json("configs/wfcllm/detector_proxy_penalized_alpha04.json")

    assert config["detector"]["statistic"] == "calibrated_context_mean_proxy_penalized"
    assert config["detector"]["proxy_penalty_alpha"] == 0.4
    assert config["detector"]["evidence_mode"] == "hit_plus_margin"
    assert config["detector"]["target_fpr"] == 0.05
    assert config["calibration"]["target_fpr"] == 0.05
    assert config["artifacts"]["run_root"] == "data/runs"


def test_wfcllm_repro_humaneval_full164_config_has_repro_run_shape() -> None:
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME

    config = _read_json("configs/wfcllm/repro_humaneval_full164.json")

    assert config["method"]["name"] == EVIDENCE_RETRY_SEED7X3_NAME
    assert config["generation"]["dataset"] == "humaneval"
    assert config["generation"]["sample_limit"] == 164
    assert config["artifacts"]["run_root"] == "data/runs"
    assert config["runtime"]["default_phases"] == PHASES


def test_wfcllm_evidence_only_selector_diagnostic_config_is_marked_diagnostic() -> None:
    config = _read_json("configs/wfcllm/diagnostics/evidence_only_selector_diagnostic.json")

    assert config["name"] == "evidence_only_selector_diagnostic"
    assert config["diagnostic_only"] is True
    assert config["not_official_method"] is True


def _read_json(path: str) -> dict:
    with (PROJECT_ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)
