"""Wiring tests for the mandatory per-dataset gated encoder phase."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from wfcllm.cli import runners
from wfcllm.cli.arguments import build_parser
from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset
from wfcllm.orchestration.prereq import ensure_gate_phase_prerequisites
from wfcllm.orchestration.state import RunStateManager


def _gated_args(tmp_path: Path) -> argparse.Namespace:
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("", encoding="utf-8")
    model = tmp_path / "semantic-base"
    model.mkdir(exist_ok=True)
    return argparse.Namespace(
        _config_cache=config,
        run_dir=str(tmp_path / "run"),
        run_id=None,
        state_file=str(tmp_path / "state.json"),
        gate_source_catalog=str(catalog),
        semantic_encoder_model_path=str(model),
        semantic_encoder_checkpoint_path=None,
        language=None,
        eval_only=False,
    )


def _stub_trainer(captured: dict):
    def _stub(settings):
        captured["settings"] = settings
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best = output_dir / "best_model.pt"
        best.write_bytes(b"stub-projection")
        return {
            "best_model_path": str(best),
            "built_group_counts": {"train": 5, "validation": 1, "test": 1},
            "language": settings.language,
        }

    return _stub


def test_run_encoder_gated_branch_trains_projection_and_marks_state(
    tmp_path: Path, monkeypatch
) -> None:
    args = _gated_args(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection",
        _stub_trainer(captured),
    )
    state = RunStateManager(tmp_path / "state.json")

    assert runners.run_encoder(args, state) == 0

    settings = captured["settings"]
    assert Path(settings.source_catalog) == tmp_path / "catalog.jsonl"
    assert Path(settings.model_path) == tmp_path / "semantic-base"
    assert Path(settings.output_dir) == tmp_path / "run" / "encoder"
    assert settings.language == "python"
    assert state.is_done("encoder") is True
    assert state.get("encoder", "best_model_path").endswith("best_model.pt")
    assert state.get("encoder", "checkpoint") == state.get(
        "encoder", "best_model_path"
    )
    assert state.get("encoder", "language") == "python"
    assert state.get("encoder", "built_group_counts") == {
        "train": 5,
        "validation": 1,
        "test": 1,
    }


def test_run_encoder_gated_branch_uses_generation_language(
    tmp_path: Path, monkeypatch
) -> None:
    args = _gated_args(tmp_path)
    args._config_cache["generation"]["language"] = "cpp"
    captured: dict = {}
    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection",
        _stub_trainer(captured),
    )
    state = RunStateManager(tmp_path / "state.json")

    assert runners.run_encoder(args, state) == 0
    assert captured["settings"].language == "cpp"
    assert state.get("encoder", "language") == "cpp"


def test_run_encoder_gated_branch_requires_catalog_and_model(
    tmp_path: Path, monkeypatch
) -> None:
    args = _gated_args(tmp_path)
    args.gate_source_catalog = None
    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection",
        _stub_trainer({}),
    )
    state = RunStateManager(tmp_path / "state.json")

    with pytest.raises(ValueError, match="gate-source-catalog"):
        runners.run_encoder(args, state)
    assert state.is_done("encoder") is False


def test_run_encoder_non_gated_branch_keeps_legacy_contrastive_path(
    tmp_path: Path, monkeypatch
) -> None:
    ran = {}
    monkeypatch.setattr(
        "wfcllm.encoder.train.main", lambda config: ran.setdefault("config", config)
    )

    def _fail(_settings):
        raise AssertionError("projection training must not run for non-gated configs")

    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection", _fail
    )
    args = argparse.Namespace(
        _config_cache={},
        eval_only=False,
        checkpoint=None,
        model_name=str(tmp_path / "model"),
        embed_dim=None,
        lr=None,
        batch_size=None,
        epochs=None,
        margin=None,
        no_lora=False,
        no_bf16=False,
    )
    state = RunStateManager(tmp_path / "state.json")

    assert runners.run_encoder(args, state) == 0
    assert ran["config"].model_name == str(tmp_path / "model")
    assert state.is_done("encoder") is True


def test_gate_data_prereq_requires_encoder_completion(tmp_path: Path) -> None:
    args = _gated_args(tmp_path)
    state = RunStateManager(tmp_path / "state.json")

    with pytest.raises(ValueError, match="--phase encoder"):
        ensure_gate_phase_prerequisites(
            "gate-data", args._config_cache, args, state
        )

    state.mark_done("encoder", best_model_path=str(tmp_path / "best_model.pt"))
    ensure_gate_phase_prerequisites("gate-data", args._config_cache, args, state)


def test_gate_data_prereq_accepts_explicit_checkpoint(tmp_path: Path) -> None:
    args = _gated_args(tmp_path)
    args.semantic_encoder_checkpoint_path = str(tmp_path / "encoder.pt")
    state = RunStateManager(tmp_path / "state.json")

    ensure_gate_phase_prerequisites("gate-data", args._config_cache, args, state)


def _runtime_options_config() -> dict:
    return {
        "method": {
            "rewrite": {},
            "semantic": {"lsh": {"d": 1, "gamma": 0.5}},
        },
        "semantic_lsh": {
            "lsh_d": 1,
            "lsh_gamma": 0.5,
            "rule_name": "keyed_text_region",
        },
        "gate_train": {},
    }


def _runtime_args(tmp_path: Path, state_file: Path) -> argparse.Namespace:
    source_catalog = tmp_path / "sources.jsonl"
    source_catalog.write_text("", encoding="utf-8")
    for name in ("generator", "semantic", "gate-base", "cache"):
        (tmp_path / name).mkdir(exist_ok=True)
    return build_parser().parse_args(
        [
            "--state-file", str(state_file),
            "--gate-source-catalog", str(source_catalog),
            "--generation-model-path", str(tmp_path / "generator"),
            "--semantic-encoder-model-path", str(tmp_path / "semantic"),
            "--gate-base-model-path", str(tmp_path / "gate-base"),
            "--gate-cache-dir", str(tmp_path / "cache"),
        ]
    )


def test_runtime_options_fall_back_to_encoder_state_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "encoder" / "best_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"stub")
    state_file = tmp_path / "state.json"
    RunStateManager(state_file).mark_done(
        "encoder", best_model_path=str(checkpoint)
    )
    args = _runtime_args(tmp_path, state_file)

    options = runners._local_hf_runtime_options(
        args, _runtime_options_config(), "gate-data"
    )

    assert options.semantic_encoder_checkpoint_path == checkpoint


def test_runtime_options_prefer_explicit_checkpoint_over_state(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "state.json"
    RunStateManager(state_file).mark_done(
        "encoder", best_model_path=str(tmp_path / "state-checkpoint.pt")
    )
    args = _runtime_args(tmp_path, state_file)
    explicit = tmp_path / "explicit.pt"
    args.semantic_encoder_checkpoint_path = str(explicit)

    options = runners._local_hf_runtime_options(
        args, _runtime_options_config(), "gate-data"
    )

    assert options.semantic_encoder_checkpoint_path == explicit


def test_runtime_options_without_state_or_explicit_checkpoint_stay_none(
    tmp_path: Path,
) -> None:
    args = _runtime_args(tmp_path, tmp_path / "missing-state.json")

    options = runners._local_hf_runtime_options(
        args, _runtime_options_config(), "gate-data"
    )

    assert options.semantic_encoder_checkpoint_path is None
