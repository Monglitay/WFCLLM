"""Runtime wrapper for token-channel scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from wfcllm.watermark.token_channel.config import TokenChannelConfig
from wfcllm.watermark.token_channel.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.model import TokenChannelArtifactMetadata
from wfcllm.watermark.token_channel.model import TokenChannelModel
from wfcllm.watermark.token_channel.model import require_token_channel_compatibility


@dataclass(frozen=True)
class TokenChannelDecision:
    """Scored token-channel decision for one prefix position."""

    prefix_ids: tuple[int, ...]
    truncated_prefix_ids: tuple[int, ...]
    features: TokenChannelFeatures
    switch_logit: float
    preference_logits: torch.Tensor
    should_switch: bool


class TokenChannelRuntime:
    """Thin runtime wrapper around the token-channel model."""

    def __init__(
        self,
        model: TokenChannelModel,
        config: TokenChannelConfig,
        artifact_metadata: TokenChannelArtifactMetadata | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._context_width = config.context_width
        self._model.eval()

        if artifact_metadata is not None:
            require_token_channel_compatibility(
                artifact_metadata,
                tokenizer_name=artifact_metadata.tokenizer_name,
                tokenizer_vocab_size=model.vocab_size,
                context_width=config.context_width,
                feature_version=FEATURE_VERSION,
            )

    def score_prefix(
        self,
        prefix_ids: Sequence[int] | torch.Tensor,
        features: TokenChannelFeatures,
    ) -> TokenChannelDecision:
        normalized_prefix = _normalize_prefix_ids(prefix_ids)
        truncated_prefix = normalized_prefix[-self._context_width :]
        prefix_tensor = torch.tensor(truncated_prefix, dtype=torch.long)

        with torch.no_grad():
            output = self._model(prefix_tensor, features)

        switch_logit = float(output.switch_logit.item())
        preference_logits = output.preference_logits.detach().cpu()
        return TokenChannelDecision(
            prefix_ids=normalized_prefix,
            truncated_prefix_ids=truncated_prefix,
            features=features,
            switch_logit=switch_logit,
            preference_logits=preference_logits,
            should_switch=switch_logit >= self._config.switch_threshold,
        )


def _normalize_prefix_ids(prefix_ids: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(prefix_ids, torch.Tensor):
        if prefix_ids.ndim != 1:
            raise ValueError("prefix_ids tensor must be 1D")
        return tuple(int(token_id) for token_id in prefix_ids.tolist())
    return tuple(int(token_id) for token_id in prefix_ids)
