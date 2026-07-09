"""Shared test fixtures for watermark tests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from wfcllm.watermark.config import WatermarkConfig


@dataclass
class MockModelOutput:
    logits: torch.Tensor
    past_key_values: tuple


class MockLM:
    """Deterministic mock language model for testing.

    Routes token generation based on (base_seq_len, round) where:
    - base_seq_len: the KV cache seq_len at the start of the branch
    - round: how many times we've visited this seq_len (tracks retries)

    Usage:
        lm = MockLM(vocab_size=100, num_layers=2)
        lm.register_branch(base_seq_len=10, token_ids=[5, 6, 7, 2])  # round=1
        lm.register_branch(base_seq_len=10, token_ids=[8, 9, 2], round=2)  # after rollback
    """

    def __init__(self, vocab_size: int = 100, num_layers: int = 2):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self._branches: dict[tuple[int, int], list[int]] = {}
        self._seq_len_visits: dict[int, int] = defaultdict(int)
        self._current_branch_idx: int = 0

    def register_branch(
        self, base_seq_len: int, token_ids: list[int], round: int = 1
    ):
        self._branches[(base_seq_len, round)] = token_ids

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        if past_key_values is None:
            # Prefill: create initial KV cache
            seq_len = input_ids.shape[1]
            past_kv = self._make_kv(seq_len)
            logits = torch.zeros(1, seq_len, self.vocab_size)
            return MockModelOutput(logits=logits, past_key_values=past_kv)

        # Decode step
        current_seq_len = past_key_values[0][0].shape[2]
        new_seq_len = current_seq_len + 1
        past_kv = self._make_kv(new_seq_len)

        # Find the right branch
        # Look for a branch whose base_seq_len is <= current_seq_len
        # and compute offset to determine which token to emit
        token_id = 0  # default
        for (base_sl, rnd), ids in self._branches.items():
            offset = current_seq_len - base_sl
            if offset >= 0 and offset < len(ids):
                # Check round
                if rnd <= self._seq_len_visits.get(base_sl, 0) + 1:
                    token_id = ids[offset]
                    break

        logits = torch.full((1, 1, self.vocab_size), -10.0)
        logits[0, 0, token_id] = 10.0  # Make this token overwhelmingly likely

        return MockModelOutput(logits=logits, past_key_values=past_kv)

    def _make_kv(self, seq_len: int) -> tuple:
        batch, heads, head_dim = 1, 4, 32
        return tuple(
            (
                torch.randn(batch, heads, seq_len, head_dim),
                torch.randn(batch, heads, seq_len, head_dim),
            )
            for _ in range(self.num_layers)
        )

    def notify_rollback(self, base_seq_len: int):
        """Called by test infrastructure when a rollback occurs to this seq_len."""
        self._seq_len_visits[base_seq_len] = self._seq_len_visits.get(base_seq_len, 0) + 1


class MockEncoder:
    """Mock semantic encoder that returns configurable embeddings."""

    def __init__(self, embed_dim: int = 128, default_embedding: torch.Tensor | None = None):
        self._embed_dim = embed_dim
        self._default = default_embedding if default_embedding is not None else torch.randn(embed_dim)
        self._text_map: dict[str, torch.Tensor] = {}

    def register_embedding(self, text: str, embedding: torch.Tensor):
        self._text_map[text] = embedding

    def __call__(self, input_ids, attention_mask=None):
        # Return batch of embeddings
        return self._default.unsqueeze(0)


class MockTokenizer:
    """Simple tokenizer that maps characters to integer IDs."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next_id = 10  # reserve 0-9

    def _ensure_char(self, ch: str) -> int:
        if ch not in self._char_to_id:
            self._char_to_id[ch] = self._next_id
            self._id_to_char[self._next_id] = ch
            self._next_id += 1
        return self._char_to_id[ch]

    def encode(self, text: str, add_special_tokens: bool = True, return_tensors=None) -> list[int]:
        ids = [self._ensure_char(ch) for ch in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, list):
            return "".join(self._id_to_char.get(i, "") for i in ids)
        return self._id_to_char.get(ids, "")

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        ids = self.encode(text)
        result = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
        return result


@pytest.fixture
def watermark_config():
    """Standard test config."""
    return WatermarkConfig(
        secret_key="test-key",
        max_new_tokens=50,
        max_retries=3,
        encoder_device="cpu",
        temperature=0.0,  # greedy for determinism
        top_k=0,
        top_p=1.0,
    )


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def mock_model():
    return MockLM()


@pytest.fixture
def mock_encoder():
    return MockEncoder()
