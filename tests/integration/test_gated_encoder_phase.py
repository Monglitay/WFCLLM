"""Wiring tests for the mandatory per-dataset Gate encoder phase."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from wfcllm.cli import runners
from wfcllm.method.presets import (
    GATED_SEMANTIC_WINDOW_V1_NAME,
    load_method_preset,
)
from wfcllm.orchestration.state import RunStateManager


def _gated_args(tmp_path: Path) -> argparse.Namespace:
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("", encoding="utf-8")
    model = tmp_path / "semantic-base"
    model.mkdir()
    return argparse.Namespace(
        _config_cache=config,
        run_dir=str(tmp_path / "run"),
        run_id=None,
        gate_source_catalog=str(catalog),
        semantic_encoder_model_path=str(model),
        state_file=str(tmp_path / "state.json"),
        language=None,
    )


def _stub_trainer(captured: dict[str, object]):
    def stub(settings):
        captured["settings"] = settings
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True)
        best = output_dir / "best_model.pt"
        best.write_bytes(b"stub-projection")
        return {
            "best_model_path": str(best),
            "built_group_counts": {
                "train": 5,
                "validation": 1,
                "test": 1,
            },
            "language": settings.language,
        }

    return stub


def test_encoder_trains_per_dataset_projection_and_binds_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _gated_args(tmp_path)
    captured: dict[str, object] = {}
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
    assert state.get("encoder", "checkpoint") == str(
        tmp_path / "run" / "encoder" / "best_model.pt"
    )
    assert state.get("encoder", "run_dir") == str(tmp_path / "run")
    assert state.get("encoder", "built_group_counts") == {
        "train": 5,
        "validation": 1,
        "test": 1,
    }
    assert state.get("encoder", "best_model_sha256") == hashlib.sha256(
        b"stub-projection"
    ).hexdigest()
    assert runners._resolve_semantic_encoder_checkpoint(
        args, args._config_cache
    ) == tmp_path / "run" / "encoder" / "best_model.pt"


def test_encoder_rejects_forged_external_checkpoint_state(
    tmp_path: Path,
) -> None:
    args = _gated_args(tmp_path)
    external = tmp_path / "external" / "best_model.pt"
    external.parent.mkdir()
    external.write_bytes(b"historical")
    state = RunStateManager(Path(args.state_file))
    from wfcllm.gate.production import experiment_contract_hash

    state.mark_done(
        "encoder",
        best_model_path=str(external),
        best_model_sha256=hashlib.sha256(b"historical").hexdigest(),
        source_catalog_sha256=hashlib.sha256(
            Path(args.gate_source_catalog).read_bytes()
        ).hexdigest(),
        config_sha256=experiment_contract_hash(args._config_cache),
    )

    with pytest.raises(ValueError, match="checkpoint path"):
        runners._resolve_semantic_encoder_checkpoint(args, args._config_cache)


def test_encoder_rejects_checkpoint_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _gated_args(tmp_path)
    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection",
        _stub_trainer({}),
    )
    state = RunStateManager(Path(args.state_file))
    runners.run_encoder(args, state)
    checkpoint = tmp_path / "run" / "encoder" / "best_model.pt"
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checkpoint hash"):
        runners._resolve_semantic_encoder_checkpoint(args, args._config_cache)


@pytest.mark.parametrize("language", ["python", "cpp", "java", "js"])
def test_encoder_uses_current_experiment_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
) -> None:
    args = _gated_args(tmp_path)
    args._config_cache["generation"]["language"] = language
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection",
        _stub_trainer(captured),
    )

    runners.run_encoder(args, RunStateManager(tmp_path / "state.json"))

    assert captured["settings"].language == language


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("gate_source_catalog", "gate-source-catalog"),
        ("semantic_encoder_model_path", "semantic-encoder-model-path"),
    ],
)
def test_encoder_requires_current_local_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    args = _gated_args(tmp_path)
    setattr(args, field, None)
    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection",
        _stub_trainer({}),
    )
    state = RunStateManager(tmp_path / "state.json")

    with pytest.raises(ValueError, match=message):
        runners.run_encoder(args, state)
    assert not state.is_done("encoder")


def test_encoder_rejects_removed_method_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _gated_args(tmp_path)
    args._config_cache["method"]["name"] = "removed_method"

    def fail(_settings):
        raise AssertionError("training must not run for a removed method")

    monkeypatch.setattr(
        "wfcllm.encoder.projection_training.train_semantic_projection",
        fail,
    )

    with pytest.raises(ValueError, match="gated_semantic_window_v1"):
        runners.run_encoder(
            args,
            RunStateManager(tmp_path / "state.json"),
        )
