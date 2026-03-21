"""Offline FPR threshold calibration for watermark detection."""

from __future__ import annotations

import math
from typing import Callable, Literal

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.config import BlockScore
from wfcllm.extract.scorer import BlockScorer


class ThresholdCalibrator:
    """Compute FPR-based detection threshold M_r from a negative corpus.

    Uses BlockScorer to compute Z scores for each sample's simple blocks,
    then returns the (1-fpr) percentile as the threshold M_r.
    """

    def __init__(
        self,
        scorer: BlockScorer,
        gamma: float = 0.5,
        mode: Literal["fixed", "adaptive"] = "fixed",
        block_contract_builder: Callable[[str], dict[str, dict]] | None = None,
    ):
        self._scorer = scorer
        self._gamma = gamma
        self._mode = mode
        self._block_contract_builder = block_contract_builder

    def calibrate(self, corpus: list[dict], fpr: float) -> dict:
        """Compute M_r from a list of negative-sample records.

        Args:
            corpus: List of dicts with key "generated_code" (str).
            fpr: Target false positive rate, e.g. 0.01 for 1%.

        Returns:
            Dict with keys: fpr, fpr_threshold (M_r), n_samples.

        Raises:
            ValueError: If corpus is empty or no valid Z scores collected.
        """
        if not corpus:
            raise ValueError("corpus is empty")

        z_scores: list[float] = []
        for record in corpus:
            code = record.get("generated_code", "")
            blocks = extract_statement_blocks(code)
            if not blocks:
                continue

            # Only simple blocks carry watermark signal
            simple_blocks = [b for b in blocks if b.block_type == "simple"]
            if not simple_blocks:
                continue

            block_contracts_by_id = None
            if self._mode == "adaptive" and self._block_contract_builder is not None:
                block_contracts_by_id = self._block_contract_builder(code)

            scores = self._scorer.score_all(
                simple_blocks,
                blocks,
                block_contracts_by_id=block_contracts_by_id,
            )

            m = len(scores)
            if m == 0:
                continue

            x = sum(1 for s in scores if s.score == 1)
            expected_hits, variance = self._distribution_parameters(scores)
            z = self._compute_z_score(x, expected_hits, variance)
            z_scores.append(z)

        fpr_threshold: float
        if not z_scores:
            fpr_threshold = 0.0
        else:
            fpr_threshold = self._percentile_threshold(z_scores, fpr)

        return {
            "fpr": fpr,
            "fpr_threshold": fpr_threshold,
            "n_samples": len(corpus),
        }

    def _distribution_parameters(
        self,
        scores: list[BlockScore],
    ) -> tuple[float, float]:
        if self._mode == "adaptive":
            expected_hits = sum(score.gamma_effective for score in scores)
            variance = sum(
                score.gamma_effective * (1 - score.gamma_effective)
                for score in scores
            )
            return expected_hits, variance

        m = len(scores)
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

    @staticmethod
    def _percentile_threshold(z_scores: list[float], fpr: float) -> float:
        """Return (1-fpr) percentile of z_scores via linear interpolation."""
        sorted_z = sorted(z_scores)
        n = len(sorted_z)
        p = 1.0 - fpr
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_z[lo] + (sorted_z[hi] - sorted_z[lo]) * (idx - lo)
