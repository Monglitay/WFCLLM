from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wfcllm.dynamic_semantic.encoder import (
    SemanticEncoderRuntime,
    verify_file_sha256,
)


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        extra = 2 if add_special_tokens else 0
        return list(range(len(text.split()) + extra))

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        max_length: int | None = None,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding in (True, "max_length")
        assert truncation is False
        assert return_tensors == "pt"
        rows = [self.encode(text) for text in texts]
        width = max_length if padding == "max_length" else max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeModel(torch.nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).to(torch.float64)
        return torch.stack((lengths, lengths + 1, lengths + 2), dim=1)


def test_runtime_batches_without_truncation_and_normalizes_float32() -> None:
    runtime = SemanticEncoderRuntime(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        max_tokens=8,
    )

    embeddings = runtime.encode(("one two", "one two three"))

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == torch.float32
    assert torch.linalg.vector_norm(embeddings, dim=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert runtime.encoder_calls == 1
    assert runtime.encoded_contexts == 2


def test_runtime_rejects_oversize_context_instead_of_truncating() -> None:
    runtime = SemanticEncoderRuntime(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        max_tokens=4,
    )

    with pytest.raises(ValueError, match="token budget"):
        runtime.encode(("one two three",))

    assert runtime.encoder_calls == 0
    assert runtime.encoded_contexts == 0


def test_runtime_rejects_empty_batch() -> None:
    runtime = SemanticEncoderRuntime(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        max_tokens=8,
    )

    with pytest.raises(ValueError, match="non-empty"):
        runtime.encode(())


def test_verify_file_sha256_accepts_exact_hash_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    expected = hashlib.sha256(b"checkpoint").hexdigest()

    verify_file_sha256(path, expected)

    with pytest.raises(ValueError, match="SHA-256"):
        verify_file_sha256(path, "0" * 64)


def test_fixed_batch_shape_pads_and_slices_without_counting_dummy_rows() -> None:
    runtime = SemanticEncoderRuntime(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        max_tokens=8,
        fixed_batch_size=4,
        fixed_sequence_length=True,
    )

    embeddings = runtime.encode(("one", "one two"))

    assert embeddings.shape == (2, 3)
    assert runtime.encoder_calls == 1
    assert runtime.encoded_contexts == 2
