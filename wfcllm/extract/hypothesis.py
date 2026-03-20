"""Z-score hypothesis testing for watermark detection."""

from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

from wfcllm.extract.config import BlockScore, DetectionResult


class HypothesisTester:
    """One-sided Z-test for watermark presence."""

    def __init__(
        self,
        fpr_threshold: float = 3.0,
        gamma: float = 0.5,
        mode: Literal["fixed", "adaptive"] = "fixed",
    ):
        self._fpr_threshold = fpr_threshold
        self._gamma = gamma
        self._mode = mode

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
        expected_hits, variance = self._distribution_parameters(selected_scores)
        if m == 0:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                total_blocks=total_blocks,
                independent_blocks=0,
                hit_blocks=0,
                expected_hits=expected_hits,
                variance=variance,
                hypothesis_mode=self._mode,
                block_details=list(selected_scores),
            )

        x = sum(1 for s in selected_scores if s.score == 1)
        z_score = self._compute_z_score(x, expected_hits, variance)
        p_value = float(norm.sf(z_score))

        return DetectionResult(
            is_watermarked=z_score >= self._fpr_threshold,
            z_score=z_score,
            p_value=p_value,
            total_blocks=total_blocks,
            independent_blocks=m,
            hit_blocks=x,
            expected_hits=expected_hits,
            variance=variance,
            hypothesis_mode=self._mode,
            block_details=list(selected_scores),
        )

    def _distribution_parameters(
        self,
        selected_scores: list[BlockScore],
    ) -> tuple[float, float]:
        if self._mode == "adaptive":
            expected_hits = sum(score.gamma_effective for score in selected_scores)
            variance = sum(
                score.gamma_effective * (1 - score.gamma_effective)
                for score in selected_scores
            )
            return expected_hits, variance

        m = len(selected_scores)
        expected_hits = m * self._gamma
        variance = m * self._gamma * (1 - self._gamma)
        return expected_hits, variance

    @staticmethod
    def _compute_z_score(
        observed_hits: int,
        expected_hits: float,
        variance: float,
    ) -> float:
        if variance <= 0.0:
            if observed_hits > expected_hits:
                return math.inf
            if observed_hits < expected_hits:
                return -math.inf
            return 0.0

        return (observed_hits - expected_hits) / math.sqrt(variance)
