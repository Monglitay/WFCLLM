"""Z-score hypothesis testing for watermark detection."""

from __future__ import annotations

import math

from scipy.stats import norm

from wfcllm.extract.config import BlockScore, DetectionResult


class HypothesisTester:
    """One-sided Z-test for watermark presence."""

    def __init__(self, z_threshold: float = 3.0):
        self._z_threshold = z_threshold

    def test(
        self,
        selected_scores: list[BlockScore],
        total_blocks: int,
    ) -> DetectionResult:
        """Run hypothesis test on independent block scores.

        Args:
            selected_scores: Scores of DP-selected independent blocks.
            total_blocks: Total number of statement blocks in the code.

        Returns:
            DetectionResult with Z-score, p-value, and verdict.
        """
        m = len(selected_scores)
        if m == 0:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                total_blocks=total_blocks,
                independent_blocks=0,
                hit_blocks=0,
                block_details=list(selected_scores),
            )

        x = sum(1 for s in selected_scores if s.score == 1)
        z_score = (x - m / 2) / math.sqrt(m / 4)
        p_value = float(norm.sf(z_score))

        return DetectionResult(
            is_watermarked=z_score > self._z_threshold,
            z_score=z_score,
            p_value=p_value,
            total_blocks=total_blocks,
            independent_blocks=m,
            hit_blocks=x,
            block_details=list(selected_scores),
        )
