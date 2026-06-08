"""Embedding providers for anchor validation diagnostics."""

from __future__ import annotations

import hashlib
from typing import Protocol

import torch
import torch.nn.functional as F


class EmbeddingProvider(Protocol):
    embed_dim: int

    def embed(self, text: str) -> torch.Tensor:
        ...


class HashEmbeddingProvider:
    """Deterministic normalized embeddings for tests and smoke diagnostics."""

    def __init__(self, embed_dim: int = 128) -> None:
        self.embed_dim = embed_dim

    def embed(self, text: str) -> torch.Tensor:
        values: list[float] = []
        counter = 0
        while len(values) < self.embed_dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.embed_dim:
                    break
            counter += 1
        vector = torch.tensor(values, dtype=torch.float32)
        return F.normalize(vector.unsqueeze(0), dim=1).squeeze(0)


class EncoderEmbeddingProvider:
    """Adapter for encoder/tokenizer pairs used by real offline diagnostics."""

    def __init__(
        self,
        encoder,
        tokenizer,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self.embed_dim = _resolve_encoder_embed_dim(encoder)

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        return self._encoder(input_ids, attention_mask).squeeze(0).detach().cpu()


def _resolve_encoder_embed_dim(encoder) -> int:
    direct = getattr(encoder, "embed_dim", None)
    if direct is not None:
        return int(direct)
    config = getattr(encoder, "config", None)
    config_dim = getattr(config, "embed_dim", None)
    if config_dim is not None:
        return int(config_dim)
    projection = getattr(encoder, "projection", None)
    out_features = getattr(projection, "out_features", None)
    if out_features is not None:
        return int(out_features)
    return 128
