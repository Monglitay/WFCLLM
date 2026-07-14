from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from wfcllm.dynamic_semantic.config import EncoderConfig


def verify_file_sha256(path: str | Path, expected_sha256: str) -> None:
    file_path = Path(path)
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"failed to read checkpoint: {file_path}") from exc
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {file_path}")


class SemanticEncoderRuntime:
    """Counted, no-truncation inference wrapper for frozen semantic evidence."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str | torch.device,
        max_tokens: int,
        fixed_batch_size: int | None = None,
        fixed_sequence_length: bool = False,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        if fixed_batch_size is not None and fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size must be positive or None")
        self._fixed_batch_size = fixed_batch_size
        self._fixed_sequence_length = fixed_sequence_length
        self.encoder_calls = 0
        self.encoded_contexts = 0

    @classmethod
    def load(
        cls,
        config: EncoderConfig,
        *,
        max_tokens: int,
        device: str | torch.device = "cuda",
        fixed_batch_size: int | None = None,
        fixed_sequence_length: bool = True,
    ) -> SemanticEncoderRuntime:
        verify_file_sha256(config.checkpoint_path, config.checkpoint_sha256)
        from transformers import AutoTokenizer

        from wfcllm.encoder.config import EncoderConfig as ProjectEncoderConfig
        from wfcllm.encoder.model import SemanticEncoder

        project_config = ProjectEncoderConfig(
            model_name=config.model_path,
            embed_dim=config.embedding_dimensions,
            use_lora=True,
            use_bf16=False,
            max_seq_length=max_tokens,
        )
        model = SemanticEncoder(project_config)
        try:
            checkpoint = torch.load(
                config.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("failed to load semantic encoder checkpoint") from exc
        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        try:
            model.load_state_dict(state_dict)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("semantic encoder checkpoint is incompatible") from exc
        model = model.float()
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_tokens=max_tokens,
            fixed_batch_size=fixed_batch_size,
            fixed_sequence_length=fixed_sequence_length,
        )

    def token_count(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text, add_special_tokens=True)
        return len(token_ids)

    def encode(self, contexts: Sequence[str]) -> torch.Tensor:
        if isinstance(contexts, (str, bytes)) or not contexts:
            raise ValueError("contexts must be a non-empty sequence of strings")
        normalized = tuple(contexts)
        if any(not isinstance(context, str) or not context for context in normalized):
            raise ValueError("contexts must contain non-empty strings")
        lengths = [self.token_count(context) for context in normalized]
        if any(length > self._max_tokens for length in lengths):
            raise ValueError("context exceeds frozen token budget; truncation is forbidden")
        chunk_size = self._fixed_batch_size or len(normalized)
        chunks: list[torch.Tensor] = []
        physical_calls = 0
        for start in range(0, len(normalized), chunk_size):
            real_chunk = normalized[start : start + chunk_size]
            padded_chunk = real_chunk
            if self._fixed_batch_size is not None and len(real_chunk) < chunk_size:
                padded_chunk = real_chunk + (real_chunk[0],) * (chunk_size - len(real_chunk))
            batch = self._tokenizer(
                list(padded_chunk),
                padding="max_length" if self._fixed_sequence_length else True,
                max_length=self._max_tokens if self._fixed_sequence_length else None,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"].to(self._device)
            attention_mask = batch["attention_mask"].to(self._device)
            with torch.no_grad():
                output = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("semantic encoder must return a rank-2 tensor")
            chunks.append(output[: len(real_chunk)].float())
            physical_calls += 1
        embeddings = F.normalize(torch.cat(chunks, dim=0), p=2, dim=1)
        if not torch.isfinite(embeddings).all():
            raise ValueError("semantic encoder produced non-finite embeddings")
        self.encoder_calls += physical_calls
        self.encoded_contexts += len(normalized)
        return embeddings.detach()
