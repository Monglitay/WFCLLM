"""Offline FPR threshold calibration for watermark detection."""

from __future__ import annotations

import math

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.dp_selector import DPSelector
from wfcllm.extract.scorer import BlockScorer


class ThresholdCalibrator:
    """Compute FPR-based detection threshold M_r from a negative corpus.

    Uses BlockScorer + DPSelector to compute SEMSTAMP Z scores for each
    sample, then returns the (1-fpr) percentile as the threshold M_r.

    Does NOT depend on HypothesisTester to avoid circular logic.
    """

    def __init__(self, scorer: BlockScorer, gamma: float = 0.5):
        self._scorer = scorer
        self._gamma = gamma
        self._dp = DPSelector()

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

            scores = self._scorer.score_all(blocks)
            selected_ids = set(self._dp.select(blocks, scores))
            selected = [s for s in scores if s.block_id in selected_ids]

            m = len(selected)
            if m == 0:
                continue

            x = sum(1 for s in selected if s.score == 1)
            gamma = self._gamma
            z = (x - m * gamma) / math.sqrt(m * gamma * (1 - gamma))
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
