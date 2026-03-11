"""Tests for HypothesisTester."""

from __future__ import annotations

import math

import pytest

from wfcllm.extract.config import BlockScore, DetectionResult
from wfcllm.extract.hypothesis import HypothesisTester


def _make_score(block_id: str, score: int) -> BlockScore:
    return BlockScore(
        block_id=block_id,
        score=score,
        min_margin=0.5 if score == 1 else 0.1,
        selected=True,
    )


class TestHypothesisTester:
    @pytest.fixture
    def tester(self):
        return HypothesisTester(z_threshold=3.0)

    def test_empty_blocks(self, tester):
        """No blocks -> not watermarked."""
        result = tester.test(selected_scores=[], total_blocks=0)
        assert result.is_watermarked is False
        assert result.z_score == 0.0
        assert result.p_value == 1.0
        assert result.independent_blocks == 0

    def test_all_hits(self, tester):
        """All 20 blocks hit -> high Z-score."""
        scores = [_make_score(str(i), 1) for i in range(20)]
        result = tester.test(selected_scores=scores, total_blocks=25)

        # Z = (20 - 10) / sqrt(5) ≈ 4.47
        expected_z = (20 - 10) / math.sqrt(5)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)
        assert result.is_watermarked is True
        assert result.independent_blocks == 20
        assert result.hit_blocks == 20
        assert result.total_blocks == 25

    def test_half_hits(self, tester):
        """Exactly half hits -> Z ≈ 0, not watermarked."""
        scores = [_make_score(str(i), 1) for i in range(10)]
        scores += [_make_score(str(i + 10), 0) for i in range(10)]
        result = tester.test(selected_scores=scores, total_blocks=20)

        assert result.z_score == pytest.approx(0.0)
        assert result.is_watermarked is False

    def test_custom_threshold(self):
        """Lower threshold makes detection easier."""
        tester = HypothesisTester(z_threshold=1.0)
        # 15 hits out of 20: Z = (15 - 10)/sqrt(5) ≈ 2.24
        scores = [_make_score(str(i), 1) for i in range(15)]
        scores += [_make_score(str(i + 15), 0) for i in range(5)]
        result = tester.test(selected_scores=scores, total_blocks=20)

        assert result.z_score > 1.0
        assert result.is_watermarked is True

    def test_p_value_decreases_with_z(self, tester):
        """Higher Z-score means lower p-value."""
        scores_high = [_make_score(str(i), 1) for i in range(20)]
        scores_low = [_make_score(str(i), 1) for i in range(12)]
        scores_low += [_make_score(str(i + 12), 0) for i in range(8)]

        r_high = tester.test(selected_scores=scores_high, total_blocks=20)
        r_low = tester.test(selected_scores=scores_low, total_blocks=20)

        assert r_high.p_value < r_low.p_value

    def test_result_includes_block_details(self, tester):
        """DetectionResult.block_details contains the input scores."""
        scores = [_make_score("0", 1), _make_score("1", 0)]
        result = tester.test(selected_scores=scores, total_blocks=5)
        assert len(result.block_details) == 2
        assert result.block_details[0].block_id == "0"


class TestHypothesisTesterGamma:
    def test_custom_gamma_z_score(self):
        """With gamma=0.25 and all hits, Z-score uses correct formula."""
        import math
        tester = HypothesisTester(z_threshold=3.0, gamma=0.25)
        scores = [_make_score(str(i), 1) for i in range(20)]
        result = tester.test(selected_scores=scores, total_blocks=20)
        # Z = (20 - 20*0.25) / sqrt(20 * 0.25 * 0.75) = 15 / sqrt(3.75)
        expected_z = 15 / math.sqrt(3.75)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)

    def test_default_gamma_is_half(self):
        """Default gamma=0.5 produces same result as original formula."""
        import math
        tester = HypothesisTester(z_threshold=3.0)
        scores = [_make_score(str(i), 1) for i in range(20)]
        result = tester.test(selected_scores=scores, total_blocks=20)
        expected_z = (20 - 10) / math.sqrt(5)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)
