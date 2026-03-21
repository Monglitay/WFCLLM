"""Tests for ThresholdCalibrator."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.calibrator import ThresholdCalibrator
from wfcllm.extract.config import BlockScore


def _make_block(block_id: str, depth: int = 0, parent_id: str | None = None) -> StatementBlock:
    return StatementBlock(
        block_id=block_id,
        block_type="simple",
        node_type="expression_statement",
        source="x = 1",
        start_line=1,
        end_line=1,
        depth=depth,
        parent_id=parent_id,
        children_ids=[],
    )


class TestThresholdCalibrator:
    @pytest.fixture
    def mock_scorer(self):
        scorer = MagicMock()
        return scorer

    def test_calibrate_returns_dict_with_required_keys(self, mock_scorer):
        """calibrate() returns dict with fpr, fpr_threshold, n_samples."""
        mock_scorer.score_all.return_value = [
            BlockScore(block_id="0", score=0, min_margin=0.1)
        ]

        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        corpus = [{"generated_code": "x = 1\n"}]
        result = calibrator.calibrate(corpus, fpr=0.05)

        assert "fpr" in result
        assert "fpr_threshold" in result
        assert "n_samples" in result
        assert result["fpr"] == 0.05
        assert result["n_samples"] == 1

    def test_calibrate_fpr_01_returns_99th_percentile(self, mock_scorer):
        """FPR=0.01 -> threshold is 99th percentile of Z scores."""
        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        z_scores = list(range(100))  # known distribution: 0,1,...,99
        threshold = calibrator._percentile_threshold(z_scores, fpr=0.01)

        # 99th percentile of 0..99: idx = 0.99 * 99 = 98.01, interpolated
        assert threshold == pytest.approx(99 * 0.99, rel=0.01)

    def test_calibrate_fpr_05_returns_95th_percentile(self, mock_scorer):
        """FPR=0.05 -> threshold is 95th percentile of Z scores."""
        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        z_scores = list(range(100))
        threshold = calibrator._percentile_threshold(z_scores, fpr=0.05)

        # 95th percentile of 0..99
        assert threshold == pytest.approx(99 * 0.95, rel=0.01)

    def test_calibrate_empty_corpus_raises(self, mock_scorer):
        """Empty corpus raises ValueError."""
        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        with pytest.raises(ValueError, match="empty"):
            calibrator.calibrate([], fpr=0.01)

    def test_calibrate_skips_no_block_samples(self, mock_scorer):
        """All corpus entries with no parseable blocks -> fpr_threshold=0.0 (no signal)."""
        mock_scorer.score_all.return_value = []

        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        corpus = [{"generated_code": ""}]  # empty code -> no blocks -> skip
        result = calibrator.calibrate(corpus, fpr=0.05)

        assert result["n_samples"] == 1
        # No Z scores collected -> fallback to 0.0 (accept all, user should know corpus is bad)
        assert result["fpr_threshold"] == 0.0

    def test_calibrate_filters_simple_blocks_only(self, mock_scorer):
        """Calibrator should only score simple blocks, not compound."""
        mock_scorer.score_all.return_value = [
            BlockScore(block_id="1", score=1, min_margin=0.5),
        ]
        calibrator = ThresholdCalibrator(mock_scorer)
        # Code with compound (for) + simple (x = 1) blocks
        corpus = [{"generated_code": "for i in range(10):\n    x = 1\n"}]
        result = calibrator.calibrate(corpus, fpr=0.01)
        # score_all should be called with simple blocks only as first arg
        call_args = mock_scorer.score_all.call_args
        target_blocks = call_args[0][0]
        assert all(b.block_type == "simple" for b in target_blocks)

    def test_calibrate_adaptive_mode_uses_gamma_sequence_mean_and_variance(
        self,
        mock_scorer,
    ):
        mock_scorer.score_all.return_value = [
            BlockScore(block_id="0", score=1, min_margin=0.1, gamma_effective=0.2),
            BlockScore(block_id="1", score=0, min_margin=0.1, gamma_effective=0.8),
        ]

        calibrator = ThresholdCalibrator(
            mock_scorer,
            gamma=0.5,
            mode="adaptive",
        )
        result = calibrator.calibrate([{"generated_code": "x = 1\ny = 2\n"}], fpr=0.05)

        expected_hits = 1.0
        variance = (0.2 * 0.8) + (0.8 * 0.2)
        expected_z = (1 - expected_hits) / math.sqrt(variance)
        assert result["fpr_threshold"] == pytest.approx(expected_z)

    def test_calibrate_adaptive_mode_passes_block_contracts_to_scorer(self, mock_scorer):
        mock_scorer.score_all.return_value = [
            BlockScore(block_id="0", score=1, min_margin=0.1, gamma_effective=0.375),
        ]
        calibrator = ThresholdCalibrator(
            mock_scorer,
            gamma=0.5,
            mode="adaptive",
            block_contract_builder=lambda code: {
                "0": {"k": 6, "gamma_effective": 0.375},
            },
        )

        calibrator.calibrate([{"generated_code": "x = 1\n"}], fpr=0.05)

        assert mock_scorer.score_all.call_args.kwargs["block_contracts_by_id"] == {
            "0": {"k": 6, "gamma_effective": 0.375},
        }
