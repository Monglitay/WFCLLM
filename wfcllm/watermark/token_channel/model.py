"""Model and artifact helpers for the token-channel runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from wfcllm.watermark.token_channel.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.features import TokenChannelFeatures

TOKEN_CHANNEL_SCHEMA_VERSION = "token-channel/v1"
TOKEN_CHANNEL_METADATA_REQUIRED_KEYS = {
    "schema_version",
    "tokenizer_name",
    "tokenizer_vocab_size",
    "context_width",
    "feature_version",
    "training_config",
}
TOKEN_CHANNEL_METADATA_FILENAME = "metadata.json"
TOKEN_CHANNEL_MODEL_FILENAME = "model.pt"


@dataclass(frozen=True)
class TokenChannelModelOutput:
    """Outputs needed by token-channel generation and detection."""

    switch_logit: torch.Tensor
    preference_logits: torch.Tensor


@dataclass(frozen=True)
class TokenChannelArtifactMetadata:
    """Validated persisted metadata for a token-channel artifact."""

    schema_version: str
    tokenizer_name: str
    tokenizer_vocab_size: int
    context_width: int
    feature_version: str
    training_config: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must be a non-empty string")
        if self.tokenizer_vocab_size <= 0:
            raise ValueError("tokenizer_vocab_size must be > 0")
        if self.context_width <= 0:
            raise ValueError("context_width must be > 0")
        if not self.feature_version:
            raise ValueError("feature_version must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "context_width": self.context_width,
            "feature_version": self.feature_version,
            "training_config": self.training_config,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TokenChannelArtifactMetadata:
        missing_keys = sorted(TOKEN_CHANNEL_METADATA_REQUIRED_KEYS - payload.keys())
        if missing_keys:
            raise ValueError(f"Missing required token-channel metadata keys: {', '.join(missing_keys)}")

        training_config = payload["training_config"]
        if not isinstance(training_config, dict):
            raise ValueError("training_config must be a JSON object")

        return cls(
            schema_version=str(payload["schema_version"]),
            tokenizer_name=str(payload["tokenizer_name"]),
            tokenizer_vocab_size=int(payload["tokenizer_vocab_size"]),
            context_width=int(payload["context_width"]),
            feature_version=str(payload["feature_version"]),
            training_config=dict(training_config),
        )


@dataclass(frozen=True)
class TokenChannelCompatibility:
    """Compatibility check result for runtime and extraction preflight."""

    is_compatible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TokenChannelArtifact:
    """Loaded token-channel artifact bundle."""

    model: TokenChannelModel
    metadata: TokenChannelArtifactMetadata
    model_path: Path
    metadata_path: Path


class TokenChannelModel(nn.Module):
    """Small dual-head network for lexical channel scoring."""

    def __init__(self, vocab_size: int, context_width: int, hidden_size: int = 64) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if context_width <= 0:
            raise ValueError("context_width must be > 0")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")

        self.vocab_size = vocab_size
        self.context_width = context_width
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.node_embedding = nn.Embedding(2048, hidden_size)
        self.parent_embedding = nn.Embedding(2048, hidden_size)
        self.language_embedding = nn.Embedding(64, hidden_size)
        self.numeric_projection = nn.Linear(3, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.switch_head = nn.Linear(hidden_size, 1)
        self.preference_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        prefix_ids: torch.Tensor,
        features: TokenChannelFeatures,
    ) -> TokenChannelModelOutput:
        if prefix_ids.ndim != 1:
            raise ValueError("prefix_ids must be a 1D tensor")

        prefix_ids = prefix_ids.to(dtype=torch.long)
        device = prefix_ids.device

        trimmed_prefix = prefix_ids[-self.context_width :]
        if trimmed_prefix.numel() == 0:
            token_summary = torch.zeros(self.hidden_size, dtype=torch.float32, device=device)
        else:
            token_summary = self.token_embedding(trimmed_prefix).mean(dim=0)

        feature_summary = self._encode_features(features=features, device=device)
        hidden = torch.tanh(
            self.hidden_projection(torch.cat((token_summary, feature_summary), dim=0))
        )
        switch_logit = self.switch_head(hidden).squeeze(-1)
        preference_logits = self.preference_head(hidden)
        return TokenChannelModelOutput(
            switch_logit=switch_logit,
            preference_logits=preference_logits,
        )

    def _encode_features(self, features: TokenChannelFeatures, device: torch.device) -> torch.Tensor:
        numeric_features = torch.tensor(
            [
                float(features.block_relative_offset),
                float(features.in_code_body),
                float(features.structure_mask),
            ],
            dtype=torch.float32,
            device=device,
        )
        return (
            self.node_embedding(_stable_feature_index(features.node_type, modulo=2048, device=device))
            + self.parent_embedding(
                _stable_feature_index(features.parent_node_type, modulo=2048, device=device)
            )
            + self.language_embedding(
                _stable_feature_index(features.language, modulo=64, device=device)
            )
            + self.numeric_projection(numeric_features)
        )


def save_token_channel_artifact_metadata(path: str | Path, metadata: dict[str, object]) -> None:
    metadata_path = Path(path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_token_channel_artifact_metadata(path: str | Path) -> dict[str, object]:
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("token-channel metadata must be a JSON object")
    validated = TokenChannelArtifactMetadata.from_mapping(payload)
    return validated.to_dict()


def check_token_channel_compatibility(
    metadata: TokenChannelArtifactMetadata,
    *,
    tokenizer_name: str,
    tokenizer_vocab_size: int,
    context_width: int,
    feature_version: str = FEATURE_VERSION,
) -> TokenChannelCompatibility:
    reasons: list[str] = []
    if metadata.tokenizer_name != tokenizer_name:
        reasons.append(
            f"tokenizer_name mismatch: expected {tokenizer_name!r}, got {metadata.tokenizer_name!r}"
        )
    if metadata.tokenizer_vocab_size != tokenizer_vocab_size:
        reasons.append(
            "tokenizer_vocab_size mismatch: "
            f"expected {tokenizer_vocab_size}, got {metadata.tokenizer_vocab_size}"
        )
    if metadata.context_width != context_width:
        reasons.append(
            f"context_width mismatch: expected {context_width}, got {metadata.context_width}"
        )
    if metadata.feature_version != feature_version:
        reasons.append(
            f"feature_version mismatch: expected {feature_version!r}, got {metadata.feature_version!r}"
        )
    return TokenChannelCompatibility(
        is_compatible=not reasons,
        reasons=tuple(reasons),
    )


def require_token_channel_compatibility(
    metadata: TokenChannelArtifactMetadata,
    *,
    tokenizer_name: str,
    tokenizer_vocab_size: int,
    context_width: int,
    feature_version: str = FEATURE_VERSION,
) -> None:
    compatibility = check_token_channel_compatibility(
        metadata,
        tokenizer_name=tokenizer_name,
        tokenizer_vocab_size=tokenizer_vocab_size,
        context_width=context_width,
        feature_version=feature_version,
    )
    if not compatibility.is_compatible:
        raise ValueError("Incompatible token-channel artifact: " + "; ".join(compatibility.reasons))


def load_token_channel_artifact(path: str | Path, map_location: str | torch.device = "cpu") -> TokenChannelArtifact:
    artifact_dir = Path(path)
    metadata_path = artifact_dir / TOKEN_CHANNEL_METADATA_FILENAME
    model_path = artifact_dir / TOKEN_CHANNEL_MODEL_FILENAME

    metadata = TokenChannelArtifactMetadata.from_mapping(load_token_channel_artifact_metadata(metadata_path))
    state_dict = torch.load(model_path, map_location=map_location)
    hidden_size = _infer_hidden_size_from_state_dict(state_dict)
    model = TokenChannelModel(
        vocab_size=metadata.tokenizer_vocab_size,
        context_width=metadata.context_width,
        hidden_size=hidden_size,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return TokenChannelArtifact(
        model=model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def _stable_feature_index(value: str, modulo: int, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % modulo
    return torch.tensor(index, dtype=torch.long, device=device)


def _infer_hidden_size_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    embedding = state_dict.get("token_embedding.weight")
    if embedding is None or embedding.ndim != 2:
        raise ValueError("Token-channel checkpoint is missing token_embedding.weight")
    return int(embedding.shape[1])
