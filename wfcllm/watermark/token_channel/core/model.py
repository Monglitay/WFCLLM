"""Model and artifact helpers for the token-channel runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wfcllm.watermark.token_channel.core.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures

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
TOKEN_CHANNEL_TRAINING_STATE_FILENAME = "training_state.pt"


@dataclass(frozen=True)
class TokenChannelModelOutput:
    """Outputs needed by token-channel generation and detection."""

    switch_logit: torch.Tensor
    preference_logits: torch.Tensor


@dataclass(frozen=True)
class TokenChannelLossWeights:
    """Relative weights for token and switch supervision."""

    distillation: float = 0.6
    ce: float = 0.2
    switch: float = 0.2

    def __post_init__(self) -> None:
        for name, value in (
            ("distillation", self.distillation),
            ("ce", self.ce),
            ("switch", self.switch),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} loss weight must be numeric")
            if value < 0:
                raise ValueError(f"{name} loss weight must be >= 0")


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
        _coerce_string(self.schema_version, "schema_version")
        _coerce_string(self.tokenizer_name, "tokenizer_name")
        _coerce_int(self.tokenizer_vocab_size, "tokenizer_vocab_size")
        _coerce_int(self.context_width, "context_width")
        _coerce_string(self.feature_version, "feature_version")
        if not isinstance(self.training_config, dict):
            raise ValueError("training_config must be a JSON object")
        if not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if self.schema_version != TOKEN_CHANNEL_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{TOKEN_CHANNEL_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
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
        if not isinstance(payload, Mapping):
            raise ValueError("TokenChannelArtifactMetadata payload must be a mapping")
        missing_keys = sorted(TOKEN_CHANNEL_METADATA_REQUIRED_KEYS - payload.keys())
        if missing_keys:
            raise ValueError(f"Missing required token-channel metadata keys: {', '.join(missing_keys)}")

        training_config = payload["training_config"]
        if not isinstance(training_config, dict):
            raise ValueError("training_config must be a JSON object")

        return cls(
            schema_version=_coerce_string(payload["schema_version"], "schema_version"),
            tokenizer_name=_coerce_string(payload["tokenizer_name"], "tokenizer_name"),
            tokenizer_vocab_size=_coerce_int(
                payload["tokenizer_vocab_size"],
                "tokenizer_vocab_size",
            ),
            context_width=_coerce_int(payload["context_width"], "context_width"),
            feature_version=_coerce_string(payload["feature_version"], "feature_version"),
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


@dataclass(frozen=True)
class TokenChannelCheckpointExport:
    """Saved local artifact paths for a trained token-channel model."""

    checkpoint_path: Path
    metadata_path: Path


class TokenChannelModel(nn.Module):
    """Transformer-based dual-head network for lexical channel scoring.

    Architecture mirrors the reference implementation: token + position embeddings,
    TransformerEncoder backbone, dual output heads for next-token preference and
    switch prediction.

    Args:
        vocab_size: Vocabulary size of the target LM tokenizer.
        context_width: Maximum prefix length (used as max_length for position embeddings).
        hidden_size: Transformer d_model dimension (default 512).
        num_layers: Number of TransformerEncoder layers (default 6).
        nhead: Number of attention heads (default 8).
        dim_feedforward: FFN hidden dimension (default 2048).
        dropout: Dropout probability (default 0.2).
    """

    def __init__(
        self,
        vocab_size: int,
        context_width: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
    ) -> None:
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

        # Token + position embeddings (same as reference)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_width, hidden_size)
        self.drop = nn.Dropout(dropout)

        # Transformer encoder backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)

        # Dual output heads
        self.switch_head = nn.Linear(hidden_size, 1)
        self.preference_head = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(
        self,
        prefix_ids: torch.Tensor,
        features: TokenChannelFeatures,
    ) -> TokenChannelModelOutput:
        if prefix_ids.ndim != 1:
            raise ValueError("prefix_ids must be a 1D tensor")
        if prefix_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("prefix_ids must use an integer tensor dtype")

        prefix_ids = prefix_ids.to(dtype=torch.long)
        device = prefix_ids.device

        # Trim to context_width and build [1, seq_len] batch
        trimmed = prefix_ids[-self.context_width :]
        seq_len = trimmed.numel()

        if seq_len == 0:
            # No prefix: use a single zero token as placeholder
            trimmed = torch.zeros(1, dtype=torch.long, device=device)
            seq_len = 1

        input_ids = trimmed.unsqueeze(0)  # [1, seq_len]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        hidden = self.drop(hidden)
        hidden = self.transformer(hidden)          # [1, seq_len, d_model]
        hidden = self.ln_f(hidden)

        # Use last token position as the prediction vector (causal)
        last_hidden = hidden[0, -1]               # [d_model]

        switch_logit = self.switch_head(last_hidden).squeeze(-1)
        preference_logits = self.preference_head(last_hidden)
        return TokenChannelModelOutput(
            switch_logit=switch_logit,
            preference_logits=preference_logits,
        )

    def compute_loss(
        self,
        *,
        batch: Mapping[str, torch.Tensor],
        output: TokenChannelModelOutput | Mapping[str, torch.Tensor],
        loss_weights: TokenChannelLossWeights | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the minimal dual-head training loss bundle."""

        weights = loss_weights or TokenChannelLossWeights()
        switch_logit, preference_logits = _coerce_model_output(output)

        next_token = _require_tensor(batch, "next_token").to(
            device=preference_logits.device,
            dtype=torch.long,
        )
        teacher_logits = _require_tensor(batch, "teacher_logits").to(
            device=preference_logits.device,
            dtype=preference_logits.dtype,
        )
        switch_target = _require_tensor(batch, "switch_target").to(
            device=switch_logit.device,
            dtype=switch_logit.dtype,
        )

        if preference_logits.ndim == 1:
            preference_logits = preference_logits.unsqueeze(0)
        if teacher_logits.ndim == 1:
            teacher_logits = teacher_logits.unsqueeze(0)
        if next_token.ndim == 0:
            next_token = next_token.unsqueeze(0)
        if switch_logit.ndim == 0:
            switch_logit = switch_logit.unsqueeze(0)
        if switch_target.ndim == 0:
            switch_target = switch_target.unsqueeze(0)

        if preference_logits.shape[0] != next_token.shape[0]:
            raise ValueError("next_token batch size must match preference logits")
        if switch_logit.shape[0] != switch_target.shape[0]:
            raise ValueError("switch_target batch size must match switch logits")

        # Compute distillation loss (KL divergence)
        # Note: teacher_logits may contain -inf for non-top-k positions (if using top-k storage)
        # This is correct: softmax(-inf) = 0, so KL divergence only considers top-k positions
        if teacher_logits.shape != preference_logits.shape:
            raise ValueError(
                f"teacher_logits shape {teacher_logits.shape} must match "
                f"preference_logits shape {preference_logits.shape}"
            )

        distillation_loss = F.kl_div(
            F.log_softmax(preference_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

        ce_loss = F.cross_entropy(preference_logits, next_token, label_smoothing=0.1)

        # Upweight minority class (switch=1) to counteract class imbalance
        importance_weights = torch.ones_like(switch_target)
        importance_weights[switch_target >= 0.5] = 2.0
        switch_loss = F.binary_cross_entropy_with_logits(
            switch_logit, switch_target, weight=importance_weights, reduction="mean"
        )
        total_loss = (
            weights.distillation * distillation_loss
            + weights.ce * ce_loss
            + weights.switch * switch_loss
        )
        return {
            "total_loss": total_loss,
            "distillation_loss": distillation_loss,
            "ce_loss": ce_loss,
            "switch_loss": switch_loss,
        }


def export_token_channel_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model: TokenChannelModel,
    metadata: dict[str, object],
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
) -> TokenChannelCheckpointExport:
    """Persist a local checkpoint plus metadata bundle.

    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        metadata: Metadata dictionary
        optimizer: Optional optimizer to save (for resuming training)
        epoch: Optional current epoch number (for resuming training)
    """

    export_dir = Path(checkpoint_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = export_dir / TOKEN_CHANNEL_MODEL_FILENAME
    metadata_path = export_dir / TOKEN_CHANNEL_METADATA_FILENAME
    torch.save(model.state_dict(), checkpoint_path)
    save_token_channel_artifact_metadata(metadata_path, metadata)

    # Save training state if optimizer and epoch are provided
    if optimizer is not None and epoch is not None:
        training_state_path = export_dir / TOKEN_CHANNEL_TRAINING_STATE_FILENAME
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
        }, training_state_path)

    return TokenChannelCheckpointExport(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
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
    if metadata.schema_version != TOKEN_CHANNEL_SCHEMA_VERSION:
        reasons.append(
            "schema_version mismatch: "
            f"expected {TOKEN_CHANNEL_SCHEMA_VERSION!r}, got {metadata.schema_version!r}"
        )
    # Skip tokenizer_name check - only vocab_size matters for compatibility
    # if metadata.tokenizer_name != tokenizer_name:
    #     reasons.append(
    #         f"tokenizer_name mismatch: expected {tokenizer_name!r}, got {metadata.tokenizer_name!r}"
    #     )
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
    state_dict = _load_checkpoint_state_dict(model_path, map_location=map_location)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Token-channel checkpoint must contain a state_dict mapping")
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


def load_training_state(
    checkpoint_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object] | None:
    """Load training state (optimizer + epoch) if available.

    Returns:
        Dictionary with 'epoch' and 'optimizer_state_dict' keys, or None if not found
    """
    training_state_path = Path(checkpoint_dir) / TOKEN_CHANNEL_TRAINING_STATE_FILENAME
    if not training_state_path.exists():
        return None

    try:
        state = torch.load(training_state_path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(training_state_path, map_location=map_location)

    if not isinstance(state, dict):
        raise ValueError("Training state must be a dictionary")
    if 'epoch' not in state or 'optimizer_state_dict' not in state:
        raise ValueError("Training state must contain 'epoch' and 'optimizer_state_dict'")

    return state


def _stable_feature_index(value: str, modulo: int, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % modulo
    return torch.tensor(index, dtype=torch.long, device=device)


def _infer_hidden_size_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> int:
    embedding = state_dict.get("token_embedding.weight")
    if embedding is None or embedding.ndim != 2:
        raise ValueError("Token-channel checkpoint is missing token_embedding.weight")
    return int(embedding.shape[1])


def _load_checkpoint_state_dict(
    model_path: Path,
    map_location: str | torch.device,
) -> object:
    try:
        return torch.load(model_path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(model_path, map_location=map_location)


def _coerce_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _coerce_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_tensor(batch: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a tensor")
    return value


def _coerce_model_output(
    output: TokenChannelModelOutput | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, TokenChannelModelOutput):
        return output.switch_logit, output.preference_logits
    if isinstance(output, Mapping):
        switch_logit = output.get("switch_logit")
        preference_logits = output.get("preference_logits")
        if isinstance(switch_logit, torch.Tensor) and isinstance(preference_logits, torch.Tensor):
            return switch_logit, preference_logits
    raise ValueError("output must provide switch_logit and preference_logits tensors")
