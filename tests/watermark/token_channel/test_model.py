"""Tests for the token-channel model and artifact metadata."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.model import (
    TOKEN_CHANNEL_METADATA_REQUIRED_KEYS,
    TokenChannelArtifactMetadata,
    TokenChannelModel,
    export_token_channel_checkpoint,
    check_token_channel_compatibility,
    load_token_channel_artifact,
    load_token_channel_artifact_metadata,
    save_token_channel_artifact_metadata,
)
from wfcllm.watermark.token_channel.train import (
    TokenChannelEpochMetrics,
    TokenChannelTrainingEvidence,
    build_training_evidence,
    build_token_channel_batch,
    run_training_step,
    save_token_channel_training_artifacts,
)


def _metadata() -> dict[str, object]:
    return {
        "schema_version": "token-channel/v1",
        "tokenizer_name": "offline-tokenizer",
        "tokenizer_vocab_size": 8,
        "context_width": 4,
        "feature_version": "token-channel-features/v1",
        "training_config": {"dropout": 0.0},
    }


def test_metadata_contract_lists_required_keys() -> None:
    assert TOKEN_CHANNEL_METADATA_REQUIRED_KEYS == {
        "schema_version",
        "tokenizer_name",
        "tokenizer_vocab_size",
        "context_width",
        "feature_version",
        "training_config",
    }


def test_token_channel_model_returns_switch_and_preference_logits() -> None:
    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    features = TokenChannelFeatures(
        node_type="if_statement",
        parent_node_type="block",
        block_relative_offset=1,
        in_code_body=True,
        structure_mask=True,
    )

    output = model(torch.tensor([1, 2, 3, 4], dtype=torch.long), features)

    assert output.switch_logit.ndim == 0
    assert output.preference_logits.shape == (8,)


def test_token_channel_model_rejects_non_integer_prefix_tensor() -> None:
    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    features = TokenChannelFeatures(
        node_type="if_statement",
        parent_node_type="block",
        block_relative_offset=1,
        in_code_body=True,
        structure_mask=True,
    )

    with pytest.raises(ValueError, match="integer"):
        model(torch.tensor([1.0, 2.0]), features)


def test_compute_loss_returns_distill_ce_and_switch_terms() -> None:
    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    features = TokenChannelFeatures(
        node_type="if_statement",
        parent_node_type="block",
        block_relative_offset=1,
        in_code_body=True,
        structure_mask=True,
    )
    batch = {
        "prefix_tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "next_token": torch.tensor([2], dtype=torch.long),
        "teacher_logits": torch.tensor([[0.2, 0.8, 0.3, 1.0, 0.1, -0.1, -0.2, -0.3]]),
        "switch_target": torch.tensor([1.0], dtype=torch.float32),
    }

    output = model(batch["prefix_tokens"][0], features)
    losses = model.compute_loss(batch=batch, output=output)

    assert set(losses) >= {"total_loss", "distillation_loss", "ce_loss", "switch_loss"}
    assert losses["total_loss"].ndim == 0


def test_switch_loss_uses_precomputed_switch_target() -> None:
    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    batch = {
        "next_token": torch.tensor([2], dtype=torch.long),
        "teacher_logits": torch.zeros((1, 8), dtype=torch.float32),
        "switch_target": torch.tensor([1.0], dtype=torch.float32),
    }
    output = {
        "switch_logit": torch.tensor([-10.0], dtype=torch.float32),
        "preference_logits": torch.zeros((1, 8), dtype=torch.float32),
    }

    losses = model.compute_loss(batch=batch, output=output)

    assert losses["switch_loss"].item() > 1.0


def test_build_token_channel_batch_converts_rows_to_tensors() -> None:
    batch = build_token_channel_batch(
        [
            {
                "prefix_tokens": [1, 2],
                "next_token": 3,
                "teacher_logits": [0.1] * 8,
                "switch_target": 1,
            },
            {
                "prefix_tokens": [4],
                "next_token": 2,
                "teacher_logits": [0.2] * 8,
                "switch_target": 0,
            },
        ],
        context_width=4,
    )

    assert batch["prefix_tokens"].shape == (2, 4)
    assert batch["next_token"].tolist() == [3, 2]
    assert batch["switch_target"].tolist() == [1.0, 0.0]


def test_run_training_step_returns_loss_terms() -> None:
    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    features = TokenChannelFeatures(
        node_type="if_statement",
        parent_node_type="block",
        block_relative_offset=1,
        in_code_body=True,
        structure_mask=True,
    )
    batch = build_token_channel_batch(
        [
            {
                "prefix_tokens": [1, 2, 3],
                "next_token": 4,
                "teacher_logits": [0.1, 0.2, 0.3, 0.4, 0.7, 0.1, -0.2, -0.4],
                "switch_target": 1,
            }
        ],
        context_width=4,
    )

    losses = run_training_step(model=model, optimizer=optimizer, batch=batch, features=features)

    assert set(losses) >= {"total_loss", "distillation_loss", "ce_loss", "switch_loss"}


def test_export_checkpoint_saves_model_and_metadata(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    metadata = _metadata()

    export = export_token_channel_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        metadata=metadata,
    )

    assert export.checkpoint_path.exists()
    assert export.metadata_path.exists()
    assert export.metadata_path.read_text(encoding="utf-8")


def test_training_evidence_tracks_switch_counts_and_losses() -> None:
    evidence = TokenChannelTrainingEvidence(
        switch_target_positive_count=3,
        switch_target_negative_count=5,
        train_loss=0.8,
        validation_loss=0.6,
        epochs=(
            TokenChannelEpochMetrics(epoch=1, train_loss=0.8, validation_loss=0.6, switch_loss=0.3),
            TokenChannelEpochMetrics(epoch=2, train_loss=0.4, validation_loss=0.5, switch_loss=0.2),
        ),
    )

    payload = evidence.to_dict()

    assert payload["switch_target_positive_count"] == 3
    assert payload["switch_target_negative_count"] == 5
    assert payload["train_loss"] == pytest.approx(0.8)
    assert payload["validation_loss"] == pytest.approx(0.6)
    assert payload["epochs"][1]["switch_loss"] == pytest.approx(0.2)


def test_build_training_evidence_counts_switch_targets() -> None:
    evidence = build_training_evidence(
        rows=[
            {"switch_target": 1},
            {"switch_target": 0},
            {"switch_target": 1},
        ],
        epochs=[
            TokenChannelEpochMetrics(epoch=1, train_loss=0.7, validation_loss=0.5, switch_loss=0.2)
        ],
    )

    assert evidence.switch_target_positive_count == 2
    assert evidence.switch_target_negative_count == 1
    assert evidence.train_loss == pytest.approx(0.7)
    assert evidence.validation_loss == pytest.approx(0.5)


def test_save_token_channel_training_artifacts_writes_evidence(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "artifact"
    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    evidence = TokenChannelTrainingEvidence(
        switch_target_positive_count=1,
        switch_target_negative_count=2,
        train_loss=0.7,
        validation_loss=0.5,
        epochs=(
            TokenChannelEpochMetrics(epoch=1, train_loss=0.7, validation_loss=0.5, switch_loss=0.2),
        ),
    )

    paths = save_token_channel_training_artifacts(
        checkpoint_dir=checkpoint_dir,
        model=model,
        metadata=_metadata(),
        evidence=evidence,
    )

    assert paths["checkpoint_path"].exists()
    assert paths["metadata_path"].exists()
    assert paths["evidence_path"].exists()
    assert '"switch_target_positive_count": 1' in paths["evidence_path"].read_text(encoding="utf-8")


def test_metadata_save_load_and_dataclass_roundtrip(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _metadata()

    save_token_channel_artifact_metadata(metadata_path, metadata)

    loaded = load_token_channel_artifact_metadata(metadata_path)
    parsed = TokenChannelArtifactMetadata.from_mapping(loaded)

    assert loaded == metadata
    assert parsed.context_width == 4
    assert parsed.tokenizer_vocab_size == 8


def test_load_metadata_rejects_missing_required_keys(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    invalid = _metadata()
    invalid.pop("training_config")

    save_token_channel_artifact_metadata(metadata_path, invalid)

    with pytest.raises(ValueError, match="training_config"):
        load_token_channel_artifact_metadata(metadata_path)


def test_load_metadata_rejects_unexpected_schema_version(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    invalid = _metadata()
    invalid["schema_version"] = "token-channel/v2"

    save_token_channel_artifact_metadata(metadata_path, invalid)

    with pytest.raises(ValueError, match="schema_version"):
        load_token_channel_artifact_metadata(metadata_path)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("schema_version", None),
        ("schema_version", True),
        ("tokenizer_name", None),
        ("tokenizer_name", False),
        ("tokenizer_vocab_size", None),
        ("tokenizer_vocab_size", True),
        ("tokenizer_vocab_size", "8"),
        ("context_width", None),
        ("context_width", False),
        ("context_width", "4"),
        ("feature_version", None),
        ("feature_version", 1),
    ],
)
def test_load_metadata_rejects_malformed_scalar_field_types(
    tmp_path: Path,
    field_name: str,
    field_value: object,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    invalid = _metadata()
    invalid[field_name] = field_value

    save_token_channel_artifact_metadata(metadata_path, invalid)

    with pytest.raises(ValueError, match=field_name):
        load_token_channel_artifact_metadata(metadata_path)


def test_compatibility_check_reports_mismatched_context_width() -> None:
    compatibility = check_token_channel_compatibility(
        TokenChannelArtifactMetadata.from_mapping(_metadata()),
        tokenizer_name="offline-tokenizer",
        tokenizer_vocab_size=8,
        context_width=6,
        feature_version="token-channel-features/v1",
    )

    assert compatibility.is_compatible is False
    assert any("context_width" in reason for reason in compatibility.reasons)


def test_compatibility_check_reports_mismatched_schema_version() -> None:
    metadata = object.__new__(TokenChannelArtifactMetadata)
    object.__setattr__(metadata, "schema_version", "token-channel/v2")
    object.__setattr__(metadata, "tokenizer_name", "offline-tokenizer")
    object.__setattr__(metadata, "tokenizer_vocab_size", 8)
    object.__setattr__(metadata, "context_width", 4)
    object.__setattr__(metadata, "feature_version", "token-channel-features/v1")
    object.__setattr__(metadata, "training_config", {"dropout": 0.0})

    compatibility = check_token_channel_compatibility(
        metadata,
        tokenizer_name="offline-tokenizer",
        tokenizer_vocab_size=8,
        context_width=4,
        feature_version="token-channel-features/v1",
    )

    assert compatibility.is_compatible is False
    assert any("schema_version" in reason for reason in compatibility.reasons)


def test_load_token_channel_artifact_restores_model_and_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "model.pt"

    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    save_token_channel_artifact_metadata(metadata_path, _metadata())
    torch.save(model.state_dict(), model_path)

    artifact = load_token_channel_artifact(artifact_dir)

    assert artifact.model_path == model_path
    assert artifact.metadata_path == metadata_path
    assert artifact.metadata.tokenizer_name == "offline-tokenizer"
    assert artifact.model.context_width == 4


def test_load_token_channel_artifact_uses_weights_only(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "model.pt"

    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    save_token_channel_artifact_metadata(metadata_path, _metadata())
    torch.save(model.state_dict(), model_path)

    with patch("wfcllm.watermark.token_channel.model.torch.load") as load_mock:
        load_mock.return_value = model.state_dict()
        load_token_channel_artifact(artifact_dir)

    assert load_mock.call_args.kwargs["weights_only"] is True


def test_load_token_channel_artifact_falls_back_without_weights_only(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "model.pt"

    model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    save_token_channel_artifact_metadata(metadata_path, _metadata())
    torch.save(model.state_dict(), model_path)

    with patch("wfcllm.watermark.token_channel.model.torch.load") as load_mock:
        load_mock.side_effect = [TypeError("weights_only"), model.state_dict()]

        artifact = load_token_channel_artifact(artifact_dir)

    assert load_mock.call_count == 2
    assert load_mock.call_args_list[0].kwargs["weights_only"] is True
    assert "weights_only" not in load_mock.call_args_list[1].kwargs
    assert artifact.model.context_width == 4


def test_load_token_channel_artifact_rejects_non_state_dict_payload(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "model.pt"

    save_token_channel_artifact_metadata(metadata_path, _metadata())
    torch.save([1, 2, 3], model_path)

    with pytest.raises(ValueError, match="state_dict"):
        load_token_channel_artifact(artifact_dir)


def test_metadata_rejects_non_mapping_payload_and_bad_direct_types() -> None:
    with pytest.raises(ValueError, match="mapping"):
        TokenChannelArtifactMetadata.from_mapping([1, 2, 3])

    with pytest.raises(ValueError, match="tokenizer_name"):
        TokenChannelArtifactMetadata(
            schema_version="token-channel/v1",
            tokenizer_name=None,
            tokenizer_vocab_size=8,
            context_width=4,
            feature_version="token-channel-features/v1",
            training_config={"dropout": 0.0},
        )

    with pytest.raises(ValueError, match="tokenizer_vocab_size"):
        TokenChannelArtifactMetadata(
            schema_version="token-channel/v1",
            tokenizer_name="offline-tokenizer",
            tokenizer_vocab_size="8",
            context_width=4,
            feature_version="token-channel-features/v1",
            training_config={"dropout": 0.0},
        )
