"""Tests for HypothesisTester."""

from __future__ import annotations

import math

import pytest

import wfcllm.extract.hypothesis as hypothesis_module
from wfcllm.extract.config import BlockScore, DetectionResult
from wfcllm.extract.hypothesis import HypothesisTester


def _make_score(
    block_id: str,
    score: int,
    *,
    gamma_effective: float = 0.5,
) -> BlockScore:
    return BlockScore(
        block_id=block_id,
        score=score,
        min_margin=0.5 if score == 1 else 0.1,
        selected=True,
        gamma_effective=gamma_effective,
    )


class TestHypothesisTester:
    @pytest.fixture
    def tester(self):
        return HypothesisTester(fpr_threshold=3.0)

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
        tester = HypothesisTester(fpr_threshold=1.0)
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
    def test_compute_z_score_is_exposed_for_cross_module_reuse(self):
        """Shared z-score helper should remain importable by sibling modules."""
        z_score = hypothesis_module.compute_z_score(
            observed_hits=7,
            expected_hits=5.0,
            variance=4.0,
        )

        assert z_score == pytest.approx(1.0)

    def test_distribution_parameters_is_exposed_for_cross_module_reuse(self):
        """Shared distribution helper should remain importable by sibling modules."""
        scores = [
            _make_score("0", 1, gamma_effective=0.2),
            _make_score("1", 0, gamma_effective=0.8),
        ]

        expected_hits, variance = hypothesis_module.distribution_parameters(
            scores,
            gamma=0.5,
            mode="adaptive",
        )

        assert expected_hits == pytest.approx(1.0)
        assert variance == pytest.approx((0.2 * 0.8) + (0.8 * 0.2))

    def test_hypothesis_tester_uses_shared_compute_z_score_helper(self, monkeypatch):
        """Detector should route z-score math through the shared helper."""
        tester = HypothesisTester(fpr_threshold=3.0, gamma=0.25, mode="fixed")
        scores = [
            _make_score("0", 1),
            _make_score("1", 0),
            _make_score("2", 1),
            _make_score("3", 1),
        ]

        captured: dict[str, float] = {}

        def fake_compute_z_score(
            observed_hits: int,
            expected_hits: float,
            variance: float,
        ) -> float:
            captured["observed_hits"] = observed_hits
            captured["expected_hits"] = expected_hits
            captured["variance"] = variance
            return 12.34

        monkeypatch.setattr(
            hypothesis_module,
            "compute_z_score",
            fake_compute_z_score,
        )

        result = tester.test(selected_scores=scores, total_blocks=4)

        assert result.z_score == pytest.approx(12.34)
        assert captured == {
            "observed_hits": 3,
            "expected_hits": pytest.approx(1.0),
            "variance": pytest.approx(0.75),
        }

    def test_hypothesis_tester_uses_shared_distribution_helper(self, monkeypatch):
        """Detector should route distribution math through the shared helper."""
        tester = HypothesisTester(fpr_threshold=3.0, gamma=0.5, mode="adaptive")
        scores = [
            _make_score("0", 1, gamma_effective=0.2),
            _make_score("1", 0, gamma_effective=0.8),
        ]

        captured: dict[str, object] = {}
        original_distribution_parameters = hypothesis_module.distribution_parameters

        def fake_distribution_parameters(scores, gamma, mode):
            captured["scores"] = scores
            captured["gamma"] = gamma
            captured["mode"] = mode
            return original_distribution_parameters(scores, gamma=gamma, mode=mode)

        monkeypatch.setattr(
            hypothesis_module,
            "distribution_parameters",
            fake_distribution_parameters,
        )

        tester.test(selected_scores=scores, total_blocks=2)

        assert captured["gamma"] == pytest.approx(0.5)
        assert captured["mode"] == "adaptive"
        assert len(captured["scores"]) == 2

class TestHypothesisTesterGamma:
    def test_custom_gamma_z_score(self):
        """With gamma=0.25 and all hits, Z-score uses correct formula."""
        import math
        tester = HypothesisTester(fpr_threshold=3.0, gamma=0.25)
        scores = [_make_score(str(i), 1) for i in range(20)]
        result = tester.test(selected_scores=scores, total_blocks=20)
        # Z = (20 - 20*0.25) / sqrt(20 * 0.25 * 0.75) = 15 / sqrt(3.75)
        expected_z = 15 / math.sqrt(3.75)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)

    def test_default_gamma_is_half(self):
        """Default gamma=0.5 produces same result as original formula."""
        import math
        tester = HypothesisTester(fpr_threshold=3.0)
        scores = [_make_score(str(i), 1) for i in range(20)]
        result = tester.test(selected_scores=scores, total_blocks=20)
        expected_z = (20 - 10) / math.sqrt(5)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)

    def test_fixed_mode_matches_legacy_formula(self):
        """Fixed mode keeps the legacy binomial mean and variance."""
        tester = HypothesisTester(fpr_threshold=3.0, gamma=0.25, mode="fixed")
        scores = [_make_score(str(i), 1, gamma_effective=0.9) for i in range(15)]
        scores += [_make_score(str(i + 15), 0, gamma_effective=0.1) for i in range(5)]

        result = tester.test(selected_scores=scores, total_blocks=20)

        expected_hits = 20 * 0.25
        expected_variance = 20 * 0.25 * 0.75
        expected_z = (15 - expected_hits) / math.sqrt(expected_variance)

        assert result.hypothesis_mode == "fixed"
        assert result.expected_hits == pytest.approx(expected_hits)
        assert result.variance == pytest.approx(expected_variance)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)

    def test_adaptive_mode_uses_gamma_sequence_mean_and_variance(self):
        """Adaptive mode uses Poisson-binomial mean and variance from per-block gamma."""
        tester = HypothesisTester(fpr_threshold=3.0, gamma=0.5, mode="adaptive")
        gammas = [0.1, 0.4, 0.8, 0.6]
        hits = [1, 0, 1, 1]
        scores = [
            _make_score(str(i), hit, gamma_effective=gamma)
            for i, (hit, gamma) in enumerate(zip(hits, gammas, strict=True))
        ]

        result = tester.test(selected_scores=scores, total_blocks=6)

        expected_hits = sum(gammas)
        expected_variance = sum(gamma * (1 - gamma) for gamma in gammas)
        expected_z = (sum(hits) - expected_hits) / math.sqrt(expected_variance)

        assert result.hypothesis_mode == "adaptive"
        assert result.expected_hits == pytest.approx(expected_hits)
        assert result.variance == pytest.approx(expected_variance)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)

    def test_result_exposes_adaptive_statistics_fields(self):
        """Result and per-block details expose the adaptive stats surface."""
        tester = HypothesisTester(fpr_threshold=3.0, mode="adaptive")
        scores = [
            _make_score("0", 1, gamma_effective=0.2),
            _make_score("1", 0, gamma_effective=0.7),
        ]

        result = tester.test(selected_scores=scores, total_blocks=2)

        assert result.block_details[0].gamma_effective == pytest.approx(0.2)
        assert result.block_details[1].gamma_effective == pytest.approx(0.7)
        assert result.expected_hits == pytest.approx(0.9)
        assert result.variance == pytest.approx((0.2 * 0.8) + (0.7 * 0.3))
        assert result.hypothesis_mode == "adaptive"
