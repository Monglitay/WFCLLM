from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "wfcllm_gate.py"


def _thresholds() -> dict[str, int | float]:
    from wfcllm.gate.feasibility import FEASIBILITY_THRESHOLD_ITEMS

    return dict(FEASIBILITY_THRESHOLD_ITEMS)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, env=env, text=True, capture_output=True, check=False)


def _provenance(payload: dict, resolved: dict) -> dict:
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "resolved_config": resolved, "config_hash": digest}


def test_help_lists_all_low_level_subcommands() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert all(name in result.stdout for name in ("data", "train", "validate"))


def test_fake_data_runs_offline_and_marks_diagnostic(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    payload = _provenance({
        "output_root": str(tmp_path / "out"),
        "scale": "pilot",
        "parser_contract": "wfcllm-window/v1",
        "rewriter_config_hash": "1" * 64,
        "semantic_encoder_hash": "2" * 64,
        "lsh_config_hash": "3" * 64,
        "training_key_source": str(tmp_path / "training.json"),
        "holdout_key_source": str(tmp_path / "holdout.json"),
        "fake_group_count": 100,
        "feasibility_contract_version": "gate-data-feasibility/v1",
        "feasibility_thresholds": _thresholds(),
    }, {"method": "diagnostic-single"})
    config.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "training.json").write_text("[]", encoding="utf-8")
    (tmp_path / "holdout.json").write_text("[]", encoding="utf-8")
    result = _run("data", "--backend", "fake", "--config", str(config))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "out" / "gate-data" / "manifest.json").read_text())
    assert manifest["diagnostic_test_backend"] is True
    assert manifest["formal_eligible"] is False


def test_data_without_key_sources_fails(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output_root": str(tmp_path / "out"), "scale": "pilot"}), encoding="utf-8")
    result = _run("data", "--backend", "fake", "--config", str(config))
    assert result.returncode != 0
    assert "key source" in result.stderr.lower()


def test_cli_rejects_mismatched_canonical_config_hash(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "output_root": str(tmp_path / "out"), "scale": "pilot",
        "resolved_config": {"method": "one"}, "config_hash": "0" * 64,
        "training_key_source": str(tmp_path / "training.json"),
        "holdout_key_source": str(tmp_path / "holdout.json"),
    }), encoding="utf-8")
    (tmp_path / "training.json").write_text("[]", encoding="utf-8")
    (tmp_path / "holdout.json").write_text("[]", encoding="utf-8")
    result = _run("data", "--backend", "fake", "--config", str(config))
    assert result.returncode == 2
    assert "does not match" in result.stderr


def test_cli_rejects_config_through_ancestor_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "config.json").write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    result = _run("data", "--backend", "fake", "--config", str(alias / "config.json"))
    assert result.returncode == 2
    assert "symlink" in result.stderr.lower()


def test_fake_validate_cannot_publish_formal_manifest(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    data = tmp_path / "data"
    candidate.mkdir()
    data.mkdir()
    (data / "manifest.json").write_text("{}\n", encoding="utf-8")
    config = tmp_path / "validate.json"
    out = tmp_path / "out"
    config.write_text(json.dumps(_provenance({"output_root": str(out), "candidate_bundle": str(candidate), "data_dir": str(data)}, {"method": "diagnostic-validate"})), encoding="utf-8")
    result = _run("validate", "--backend", "fake", "--config", str(config))
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "gate-validate" / "failed_validation_summary.json").read_text())
    assert summary["diagnostic_test_backend"] is True
    assert not (out / "gate-validate" / "gate_bundle_manifest.json").exists()


def test_fake_backend_completes_data_train_validate_offline_without_formal_publication(tmp_path: Path) -> None:
    training_keys = tmp_path / "training.json"
    holdout_keys = tmp_path / "holdout.json"
    training_keys.write_text("[]", encoding="utf-8")
    holdout_keys.write_text("[]", encoding="utf-8")
    resolved = {"method": "diagnostic-e2e"}
    config_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    common = {
        "resolved_config": resolved,
        "config_hash": config_hash,
        "parser_contract": "wfcllm-window/v1",
        "rewriter_config_hash": "b" * 64,
        "semantic_encoder_hash": "c" * 64,
        "lsh_config_hash": "d" * 64,
        "training_key_source": str(training_keys),
        "holdout_key_source": str(holdout_keys),
        "feasibility_contract_version": "gate-data-feasibility/v1",
        "feasibility_thresholds": _thresholds(),
    }
    full_root = tmp_path / "full"
    full_config = tmp_path / "full.json"
    full_config.write_text(json.dumps({**common, "output_root": str(full_root), "scale": "full", "fake_group_count": 300}), encoding="utf-8")
    assert _run("data", "--backend", "fake", "--config", str(full_config)).returncode == 0

    train_config = tmp_path / "train.json"
    train_config.write_text(json.dumps({"output_root": str(full_root), "data_dir": str(full_root / "gate-data"), "resolved_config": resolved, "config_hash": config_hash}), encoding="utf-8")
    trained = _run("train", "--backend", "fake", "--config", str(train_config))
    assert trained.returncode == 0, trained.stderr
    train_manifest = json.loads((full_root / "gate-train" / "candidate_bundle_manifest.json").read_text())
    assert train_manifest["diagnostic_test_backend"] is True
    assert train_manifest["formal_eligible"] is False
    assert (full_root / "gate-train" / "development_summary.json").exists()

    validate_config = tmp_path / "validate-e2e.json"
    validate_config.write_text(json.dumps({"output_root": str(full_root), "candidate_bundle": str(full_root / "gate-train" / "candidate_bundle"), "data_dir": str(full_root / "gate-data"), "resolved_config": resolved, "config_hash": config_hash}), encoding="utf-8")
    validated = _run("validate", "--backend", "fake", "--config", str(validate_config))
    assert validated.returncode == 0, validated.stderr
    assert not (full_root / "gate-validate" / "gate_bundle_manifest.json").exists()
