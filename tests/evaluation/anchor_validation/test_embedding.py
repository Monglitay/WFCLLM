from __future__ import annotations

import pytest
import torch

from wfcllm.evaluation.anchor_validation.embedding import (
    EncoderEmbeddingProvider,
    HashEmbeddingProvider,
)


def test_hash_embedding_provider_is_deterministic_and_normalized():
    provider = HashEmbeddingProvider(embed_dim=8)

    left = provider.embed("slot|return")
    right = provider.embed("slot|return")

    assert torch.allclose(left, right)
    assert torch.linalg.vector_norm(left).item() == pytest.approx(1.0)


def test_encoder_embedding_provider_uses_encoder_config_embed_dim():
    class FakeConfig:
        embed_dim = 64

    class FakeEncoder:
        config = FakeConfig()

    provider = EncoderEmbeddingProvider(
        encoder=FakeEncoder(),
        tokenizer=None,
    )

    assert provider.embed_dim == 64
