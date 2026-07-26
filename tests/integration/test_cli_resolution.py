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


def test_parser_accepts_gate_phases_and_non_public_secret_sources():
    parser = build_parser()
    args = parser.parse_args([
        "--phase", "gate-data",
        "--secret-key-file", "deployment.key",
        "--training-key-bank-file", "training.keys",
        "--holdout-key-bank-env", "WFCLLM_HOLDOUT_KEYS",
    ])

    assert args.phase == "gate-data"
    assert args.secret_key_file == "deployment.key"
    assert args.training_key_bank_file == "training.keys"
    assert args.holdout_key_bank_env == "WFCLLM_HOLDOUT_KEYS"


def test_runtime_config_resolver_expands_and_validates_gated_preset():
    from wfcllm.cli.config_resolver import resolve_method_config
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME

    resolved = resolve_method_config({"method": {"name": GATED_SEMANTIC_WINDOW_V1_NAME}})

    assert resolved["method"]["name"] == GATED_SEMANTIC_WINDOW_V1_NAME
    assert resolved["runtime"]["default_phases"][:3] == [
        "gate-data", "gate-train", "generate"
    ]


def _gate_args(tmp_path, monkeypatch):
    import argparse
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset

    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    config["gate_data"]["scale"] = "pilot"
    source = tmp_path / "source.json"
    source.write_text('{"schema_version":"wfcllm-gate-source-manifest/v1"}\n')
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    generation_model = tmp_path / "generation-model"
    semantic_model = tmp_path / "semantic-model"
    gate_model = tmp_path / "gate-model"
    for path in (generation_model, semantic_model, gate_model):
        path.mkdir()
    training = tmp_path / "training.keys"
    holdout = tmp_path / "holdout.keys"
    training.write_bytes(b"training-runtime-secret")
    holdout.write_bytes(b"holdout-runtime-secret")
    args = argparse.Namespace(
        _config_cache=config,
        run_dir=str(tmp_path / "run"),
        run_id=None,
        gate_source_manifest=str(source),
        gate_source_catalog=str(catalog),
        generation_model_path=str(generation_model),
        rewrite_model_path=None,
        semantic_encoder_model_path=str(semantic_model),
        semantic_encoder_checkpoint_path=None,
        semantic_whitening_path=None,
        gate_base_model_path=str(gate_model),
        model_device="cpu",
        gate_device="cpu",
        gate_cache_dir=str(tmp_path / "cache"),
        gate_batch_size=9,
        gate_resume_checkpoint=None,
        pilot_feasibility=None,
        training_key_bank_file=str(training),
        training_key_bank_env=None,
        holdout_key_bank_file=str(holdout),
        holdout_key_bank_env=None,
        secret_key_file=None,
        secret_key_env=None,
    )
    return args, source


def test_gate_data_runner_persists_config_input_and_output_hashes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from wfcllm.cli.runners import run_gate_data
    from wfcllm.orchestration.state import RunStateManager

    args, _source = _gate_args(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "wfcllm.cli.runners._formal_gate_dependencies",
        lambda _args, _phase: SimpleNamespace(diagnostic_test_backend=False),
    )

    def fake_pipeline(config, dependencies):
        output = config.output_root / "gate-data"
        output.mkdir(parents=True)
        manifest = {
            "schema_version": "wfcllm-gate-data-manifest/v1",
            "config_hash": config.config_hash,
            "diagnostic_test_backend": False,
                "experimental_only": False,
                "diagnostic_only": False,
                "not_official_method": False,
                "formal_eligible": True,
        }
        path = output / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace(manifest=manifest, manifest_path=path, output_dir=output)

    monkeypatch.setattr("wfcllm.gate.pipeline.run_gate_data", fake_pipeline)
    state = RunStateManager(tmp_path / "state.json")

    assert run_gate_data(args, state) == 0
    for field in ("config_hash", "input_hash", "output_manifest_hash", "output_artifact_hash"):
        assert len(state.get("gate-data", field)) == 64


def test_gate_data_runner_rejects_input_mutation_during_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from wfcllm.cli.runners import run_gate_data
    from wfcllm.orchestration.state import RunStateManager

    args, source = _gate_args(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "wfcllm.cli.runners._formal_gate_dependencies",
        lambda _args, _phase: SimpleNamespace(diagnostic_test_backend=False),
    )

    def mutating_pipeline(config, dependencies):
        source.write_text('{"changed":true}\n')
        output = config.output_root / "gate-data"
        output.mkdir(parents=True)
        manifest = {
            "schema_version": "wfcllm-gate-data-manifest/v1",
            "config_hash": config.config_hash,
            "diagnostic_test_backend": False,
                "experimental_only": False,
                "diagnostic_only": False,
                "not_official_method": False,
                "formal_eligible": True,
        }
        path = output / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace(manifest=manifest, manifest_path=path, output_dir=output)

    monkeypatch.setattr("wfcllm.gate.pipeline.run_gate_data", mutating_pipeline)
    state = RunStateManager(tmp_path / "state.json")

    with pytest.raises(ValueError, match="input changed"):
        run_gate_data(args, state)
    assert state.is_done("gate-data") is False


def test_main_orchestration_rejects_diagnostic_gate_dependencies(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from wfcllm.cli.runners import run_gate_data
    from wfcllm.orchestration.state import RunStateManager

    args, _source = _gate_args(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "wfcllm.gate.dependencies.build_local_gate_dependencies",
        lambda **_kwargs: SimpleNamespace(diagnostic_test_backend=True),
    )

    with pytest.raises(ValueError, match="diagnostic test backend"):
        run_gate_data(args, RunStateManager(tmp_path / "state.json"))


def test_external_validated_bundle_hash_is_bound_across_main_phases(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from wfcllm.cli.runners import _safe_tree_hash, run_calibrate, run_detect, run_generate
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset
    from wfcllm.orchestration.state import RunStateManager

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "formal.bin").write_bytes(b"formal")
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    config["method"]["gate"]["bundle_path"] = str(bundle)
    config["method"]["gate"]["bundle_sha256"] = _safe_tree_hash(bundle)
    deployment = tmp_path / "deployment.key"
    deployment.write_bytes(b"deployment")
    negative = tmp_path / "negative.jsonl"
    positive = tmp_path / "positive.jsonl"
    calibration = tmp_path / "calibration.json"
    for path in (negative, positive):
        path.write_text("", encoding="utf-8")
    calibration.write_text("{}", encoding="utf-8")
    pipeline = SimpleNamespace(calibrate_jsonl=MagicMock(), detect_jsonl=MagicMock())
    args = argparse.Namespace(
        _config_cache=config,
        run_dir=str(tmp_path / "run"), run_id=None,
        negative_input=str(negative), input=str(positive), calibration=str(calibration),
        positive_details=str(tmp_path / "details.jsonl"),
        secret_key_file=str(deployment), secret_key_env=None,
        _gated_detection_pipeline=pipeline,
    )
    monkeypatch.setattr("wfcllm.gate.bundle.GateBundle.load", lambda path: object())
    monkeypatch.setattr(
        "wfcllm.detection.gated_pipeline.load_gated_calibration_artifact",
        lambda path: object(),
    )
    state = RunStateManager(tmp_path / "state.json")

    assert run_generate(args, state) == 0
    assert run_calibrate(args, state) == 0
    assert run_detect(args, state) == 0
    hashes = {state.get(phase, "gate_bundle_sha256") for phase in ("generate", "calibrate", "detect")}
    assert hashes == {config["method"]["gate"]["bundle_sha256"]}


def test_external_bundle_hash_mismatch_is_rejected_before_model_load(tmp_path, monkeypatch):
    import argparse
    from wfcllm.cli.runners import run_generate
    from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset
    from wfcllm.orchestration.state import RunStateManager

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "formal.bin").write_bytes(b"formal")
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    config["method"]["gate"]["bundle_path"] = str(bundle)
    config["method"]["gate"]["bundle_sha256"] = "0" * 64
    deployment = tmp_path / "deployment.key"
    deployment.write_bytes(b"deployment")
    args = argparse.Namespace(_config_cache=config, run_dir=str(tmp_path / "run"), run_id=None, secret_key_file=str(deployment), secret_key_env=None)
    loaded = []
    monkeypatch.setattr("wfcllm.gate.bundle.GateBundle.load", lambda path: loaded.append(path))

    with pytest.raises(ValueError, match="hash mismatch"):
        run_generate(args, RunStateManager(tmp_path / "state.json"))
    assert loaded == []


def test_gate_validate_input_hash_changes_with_holdout_key(tmp_path):
    from wfcllm.cli.runners import compute_phase_input_hash

    args, _source = _gate_args(tmp_path, None)
    run = Path(args.run_dir)
    data = run / "gate-data"
    candidate = run / "gate-train" / "candidate_bundle"
    candidate.mkdir(parents=True)
    data.mkdir(parents=True)
    (data / "manifest.json").write_text("data")
    (candidate / "model.bin").write_bytes(b"candidate")
    (candidate.parent / "candidate_bundle_manifest.json").write_text("candidate")

    before = compute_phase_input_hash(args, "gate-validate")
    assert "holdout_key_bank" not in args._gate_runtime_secrets
    Path(args.holdout_key_bank_file).write_bytes(b"different-holdout-runtime-secret")
    after = compute_phase_input_hash(args, "gate-validate")

    assert before != after


def test_real_main_gate_data_reports_missing_production_runtime_path(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"schema_version": "wfcllm-gate-source-manifest/v1", "sources": []}))
    training = tmp_path / "training.json"
    holdout = tmp_path / "holdout.json"
    training.write_text(json.dumps([f"training-{index}" for index in range(32)]))
    holdout.write_text(json.dumps([f"holdout-{index}" for index in range(8)]))
    pilot = tmp_path / "pilot.json"
    pilot.write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "run.py"),
            "--phase", "gate-data",
            "--config", str(root / "configs/wfcllm/gated_semantic_window_v1.json"),
            "--run-dir", str(tmp_path / "run"),
            "--gate-source-manifest", str(source),
            "--training-key-bank-file", str(training),
            "--holdout-key-bank-file", str(holdout),
            "--pilot-feasibility", str(pilot),
        ],
        cwd=tmp_path,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--gate-source-catalog is required for gate-data" in result.stderr


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("gate-train", "requires the gate-data manifest"),
        ("gate-validate", "requires the gate-data manifest"),
    ],
)
def test_real_main_gate_phase_reports_specific_missing_local_resource(
    tmp_path, phase, expected
):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "run.py"),
            "--phase", phase,
            "--config", str(root / "configs/wfcllm/gated_semantic_window_v1.json"),
            "--run-dir", str(tmp_path / "empty-run"),
        ],
        cwd=tmp_path,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert expected in result.stderr


@pytest.mark.parametrize("phase", ["gate-data", "gate-train", "gate-validate"])
def test_entry_main_requires_production_runtime_for_each_gate_phase(
    tmp_path, monkeypatch, capsys, phase
):
    from wfcllm.cli.entry import main

    root = Path(__file__).resolve().parents[2]
    run = tmp_path / "run"
    data = run / "gate-data"
    train = run / "gate-train"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps({
        "schema_version": "wfcllm-gate-data-manifest/v1",
        "diagnostic_test_backend": False,
        "experimental_only": False,
        "diagnostic_only": False,
        "not_official_method": False,
        "formal_eligible": True,
    }))
    (data / "feasibility_summary.json").write_text("{}")
    (train / "candidate_bundle").mkdir(parents=True)
    (train / "candidate_bundle" / "model.bin").write_bytes(b"candidate")
    (train / "candidate_bundle_manifest.json").write_text(json.dumps({
        "schema_version": "wfcllm-gate-train-candidate/v1",
        "diagnostic_test_backend": False,
        "formal_eligible": True,
    }))
    source = tmp_path / "source.json"
    source.write_text("{}")
    training = tmp_path / "training.json"
    holdout = tmp_path / "holdout.json"
    training.write_text(json.dumps([f"training-{index}" for index in range(32)]))
    holdout.write_text(json.dumps([f"holdout-{index}" for index in range(8)]))
    pilot = tmp_path / "pilot.json"
    pilot.write_text("{}")

    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", tmp_path / "state.json")
    rc = main([
        "--phase", phase,
        "--config", str(root / "configs/wfcllm/gated_semantic_window_v1.json"),
        "--run-dir", str(run),
        "--gate-source-manifest", str(source),
        "--training-key-bank-file", str(training),
        "--holdout-key-bank-file", str(holdout),
        "--pilot-feasibility", str(pilot),
    ])

    assert rc == 1
    assert f"--gate-source-catalog is required for {phase}" in capsys.readouterr().err


def test_entry_main_non_gated_config_cannot_skip_matching_old_gate_state(
    tmp_path, monkeypatch, capsys
):
    import hashlib

    from wfcllm.cli.entry import main
    from wfcllm.cli.runners import _safe_tree_hash
    from wfcllm.orchestration.state import RunStateManager

    root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "old-run"
    output = run_dir / "gate-data"
    output.mkdir(parents=True)
    manifest = output / "manifest.json"
    manifest.write_text("stable")
    input_hash = "a" * 64
    state_path = tmp_path / "state.json"
    RunStateManager(state_path).mark_done(
        "gate-data",
        input_hash=input_hash,
        manifest_path=str(manifest),
        output_manifest_hash=hashlib.sha256(b"stable").hexdigest(),
        output_artifact_path=str(output),
        output_artifact_hash=_safe_tree_hash(output),
    )
    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", state_path)
    monkeypatch.setattr(
        "wfcllm.cli.runners.compute_phase_input_hash",
        lambda _args, _phase: input_hash,
    )

    rc = main(
        [
            "--phase",
            "gate-data",
            "--config",
            str(root / "configs/base_config.json"),
            "--run-dir",
            str(run_dir),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "gate phases require gated_semantic_window_v1" in captured.err
    assert "[跳过]" not in captured.out


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


def test_unvalidated_gate_candidate_accepts_full_validated_training_config_hash():
    from copy import deepcopy
    from pathlib import Path

    from wfcllm.cli.config_resolver import load_config, resolve_method_config
    from wfcllm.cli.runners import _unvalidated_candidate_config_hash_matches
    from wfcllm.gate.production import experiment_contract_hash

    validated_config = resolve_method_config(
        load_config(Path("configs/wfcllm/experiments/python_humaneval_full.json"))
    )
    trained_hash = experiment_contract_hash(validated_config)

    runtime_config = deepcopy(validated_config)
    runtime_config["method"]["gate"]["require_validated"] = False
    runtime_config.setdefault("experiment", {})["allow_unvalidated_gate_candidate"] = True

    assert _unvalidated_candidate_config_hash_matches(trained_hash, runtime_config)
