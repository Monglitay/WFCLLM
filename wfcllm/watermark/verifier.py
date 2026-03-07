"""Semantic projection verification for watermark embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VerifyResult:
    """Result of a watermark projection verification."""

    passed: bool
    projection: float
    target_sign: int
    margin: float


class ProjectionVerifier:
    """Verify if a code block's semantic projection matches the watermark target."""

    def __init__(self, encoder, tokenizer, device: str = "cuda"):
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._device = device

    @torch.no_grad()
    def verify(
        self, code_text: str, v: torch.Tensor, t: int, margin: float
    ) -> VerifyResult:
        """Check semantic projection of code against target direction.

        Args:
            code_text: Source code of the statement block.
            v: Direction vector (embed_dim,).
            t: Target bit in {0, 1}.
            margin: Minimum absolute projection required.

        Returns:
            VerifyResult with pass/fail and diagnostic values.
        """
        # Encode
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

        # Cosine projection
        v = v.float()
        u = u.float()
        p = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()

        # Target sign: {0,1} -> {-1,+1}
        target_sign = 2 * t - 1

        # Check: sign matches AND absolute value exceeds margin
        sign_match = (p > 0 and target_sign == 1) or (p < 0 and target_sign == -1)
        passed = sign_match and abs(p) > margin

        return VerifyResult(
            passed=passed,
            projection=p,
            target_sign=target_sign,
            margin=margin,
        )
