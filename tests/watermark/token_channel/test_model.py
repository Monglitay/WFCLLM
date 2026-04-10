"""Tests for the token-channel model and artifact metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.model import (
    TOKEN_CHANNEL_METADATA_REQUIRED_KEYS,
    TokenChannelArtifactMetadata,
    TokenChannelModel,
    check_token_channel_compatibility,
    load_token_channel_artifact,
    load_token_channel_artifact_metadata,
    save_token_channel_artifact_metadata,
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
