"""Tests for the token-channel runtime wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wfcllm.watermark.token_channel.config import TokenChannelConfig
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.model import TokenChannelArtifactMetadata
from wfcllm.watermark.token_channel.model import TokenChannelModel
from wfcllm.watermark.token_channel.model import TokenChannelModelOutput
from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime


def _features() -> TokenChannelFeatures:
    return TokenChannelFeatures(
        node_type="if_statement",
        parent_node_type="block",
        block_relative_offset=2,
        in_code_body=True,
        structure_mask=True,
    )


def _metadata() -> TokenChannelArtifactMetadata:
    return TokenChannelArtifactMetadata.from_mapping(
        {
            "schema_version": "token-channel/v1",
            "tokenizer_name": "offline-tokenizer",
            "tokenizer_vocab_size": 8,
            "context_width": 4,
            "feature_version": "token-channel-features/v1",
            "training_config": {"dropout": 0.0},
        }
    )


def test_runtime_returns_gate_and_partition_logits() -> None:
    runtime = TokenChannelRuntime(
        model=TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12),
        config=TokenChannelConfig(context_width=4, switch_threshold=0.0),
    )

    decision = runtime.score_prefix(prefix_ids=[1, 2, 3, 4, 5], features=_features())

    assert decision.truncated_prefix_ids == (2, 3, 4, 5)
    assert isinstance(decision.switch_logit, float)
    assert decision.preference_logits.shape == (8,)
    assert decision.should_switch is True or decision.should_switch is False


def test_runtime_rejects_incompatible_artifact_metadata() -> None:
    runtime_model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)
    config = TokenChannelConfig(context_width=4)
    metadata = TokenChannelArtifactMetadata.from_mapping(
        {
            **_metadata().to_dict(),
            "tokenizer_vocab_size": 9,
        }
    )

    with pytest.raises(ValueError, match="tokenizer_vocab_size"):
        TokenChannelRuntime(model=runtime_model, config=config, artifact_metadata=metadata)


def test_runtime_rejects_mismatched_runtime_tokenizer_name() -> None:
    runtime_model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)

    with pytest.raises(ValueError, match="tokenizer_name"):
        TokenChannelRuntime(
            model=runtime_model,
            config=TokenChannelConfig(context_width=4),
            artifact_metadata=_metadata(),
            tokenizer=SimpleNamespace(name_or_path="other-runtime-tokenizer"),
        )


def test_runtime_rejects_same_name_but_different_tokenizer_vocab() -> None:
    runtime_model = TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)

    class MatchingNameTokenizer:
        name_or_path = "offline-tokenizer"
        vocab_size = 9

        def __len__(self) -> int:
            return 9

    with pytest.raises(ValueError, match="tokenizer_vocab_size"):
        TokenChannelRuntime(
            model=runtime_model,
            config=TokenChannelConfig(context_width=4),
            artifact_metadata=_metadata(),
            tokenizer=MatchingNameTokenizer(),
        )


def test_runtime_accepts_tensor_prefix_ids() -> None:
    runtime = TokenChannelRuntime(
        model=TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12),
        config=TokenChannelConfig(context_width=4),
        artifact_metadata=_metadata(),
    )

    decision = runtime.score_prefix(prefix_ids=torch.tensor([3, 4, 5]), features=_features())

    assert decision.prefix_ids == (3, 4, 5)


def test_runtime_places_prefix_tensor_on_model_device() -> None:
    class MetaDeviceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.empty(0, device="meta"))
            self.vocab_size = 8
            self.seen_device: torch.device | None = None

        def forward(
            self,
            prefix_ids: torch.Tensor,
            features: TokenChannelFeatures,
        ) -> TokenChannelModelOutput:
            self.seen_device = prefix_ids.device
            if prefix_ids.device != self.anchor.device:
                raise ValueError("prefix_ids device mismatch")
            return TokenChannelModelOutput(
                switch_logit=torch.zeros(()),
                preference_logits=torch.zeros(self.vocab_size),
            )

    model = MetaDeviceModel()
    runtime = TokenChannelRuntime(model=model, config=TokenChannelConfig(context_width=4))

    decision = runtime.score_prefix(prefix_ids=[1, 2, 3], features=_features())

    assert model.seen_device == torch.device("meta")
    assert decision.prefix_ids == (1, 2, 3)


@pytest.mark.parametrize("prefix_ids", [[1, 2.5], [1, "2"], [1, True], torch.tensor([1.0, 2.0])])
def test_runtime_rejects_non_integer_prefix_ids(prefix_ids) -> None:
    runtime = TokenChannelRuntime(
        model=TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12),
        config=TokenChannelConfig(context_width=4),
    )

    with pytest.raises(ValueError, match="token"):
        runtime.score_prefix(prefix_ids=prefix_ids, features=_features())
