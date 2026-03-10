"""LSH-based semantic verification for watermark embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wfcllm.watermark.lsh_space import LSHSpace


@dataclass
class VerifyResult:
    """Result of a watermark LSH verification."""

    passed: bool
    min_margin: float
    lsh_signature: tuple[int, ...] = ()


class ProjectionVerifier:
    """Verify if a code block's LSH signature falls in the valid set G."""

    def __init__(self, encoder, tokenizer, lsh_space: LSHSpace, device: str = "cuda"):
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._lsh_space = lsh_space
        self._device = device

    @torch.no_grad()
    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> VerifyResult:
        """Check if the block's LSH signature is in valid_set with sufficient margin.

        Args:
            code_text: Source code of the statement block.
            valid_set: Set of valid LSH signatures G for this block's position.
            margin: Minimum min_margin required (0.0 at extraction time).

        Returns:
            VerifyResult with passed=True iff sign ∈ valid_set AND min_margin > margin.
        """
        inputs = self._tokenizer(
            code_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        u = self._encoder(input_ids, attention_mask).squeeze(0).cpu()

        sig = self._lsh_space.sign(u)
        mm = self._lsh_space.min_margin(u)

        passed = (sig in valid_set) and (mm > margin)
        return VerifyResult(passed=passed, min_margin=mm, lsh_signature=sig)
