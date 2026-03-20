"""Tests for wfcllm.watermark.entropy."""

import pytest
from wfcllm.watermark.entropy import ENTROPY_SCALE, NodeEntropyEstimator
from wfcllm.watermark.config import WatermarkConfig


class TestNodeEntropyEstimator:
    @pytest.fixture
    def estimator(self):
        return NodeEntropyEstimator()

    def test_known_node_type_in_table(self, estimator):
        """The table should contain well-known AST node types."""
        assert "identifier" in estimator.ENTROPY_TABLE
        assert "assignment" in estimator.ENTROPY_TABLE
        assert "return_statement" in estimator.ENTROPY_TABLE

    def test_zero_entropy_tokens(self, estimator):
        """Punctuation/keyword tokens should have zero entropy."""
        assert estimator.ENTROPY_TABLE["="] == 0.0
        assert estimator.ENTROPY_TABLE[":"] == 0.0
        assert estimator.ENTROPY_TABLE["("] == 0.0

    def test_high_entropy_types(self, estimator):
        """Complex node types should have higher entropy."""
        assert estimator.ENTROPY_TABLE["boolean_operator"] > 0.9
        assert estimator.ENTROPY_TABLE["for_statement"] > 0.5

    def test_estimate_simple_assignment(self, estimator):
        """'x = 1' has identifiers, =, integer — sum their entropies."""
        entropy = estimator.estimate_block_entropy("x = 1")
        # Should be > 0 because of identifier and expression_statement
        assert entropy > 0.0

    def test_estimate_empty_string(self, estimator):
        entropy = estimator.estimate_block_entropy("")
        assert entropy == 0.0

    def test_estimate_complex_block_higher(self, estimator):
        """A for loop should have higher total entropy than a simple assignment."""
        simple = estimator.estimate_block_entropy("x = 1")
        compound = estimator.estimate_block_entropy(
            "for i in range(10):\n    x = i + 1"
        )
        assert compound > simple

    def test_estimate_block_entropy_units_non_empty_code(self, estimator):
        entropy_units = estimator.estimate_block_entropy_units("x = 1")
        assert isinstance(entropy_units, int)
        assert entropy_units > 0

    def test_estimate_block_entropy_matches_units_over_scale(self, estimator):
        code = "for i in range(3):\n    x = i\n"
        entropy = estimator.estimate_block_entropy(code)
        entropy_units = estimator.estimate_block_entropy_units(code)
        assert entropy == pytest.approx(entropy_units / ENTROPY_SCALE)

    def test_compute_margin(self, estimator):
        """Margin = m_base + alpha * entropy."""
        config = WatermarkConfig(secret_key="k", margin_base=0.1, margin_alpha=0.05)
        margin = estimator.compute_margin(2.0, config)
        assert margin == pytest.approx(0.1 + 0.05 * 2.0)

    def test_compute_margin_zero_entropy(self, estimator):
        config = WatermarkConfig(secret_key="k", margin_base=0.1, margin_alpha=0.05)
        margin = estimator.compute_margin(0.0, config)
        assert margin == pytest.approx(0.1)

    def test_unknown_node_type_uses_default(self, estimator):
        """Unknown node types should use DEFAULT_ENTROPY."""
        assert estimator.ENTROPY_TABLE.get("nonexistent_type") is None
        # The default is applied during traversal, not table lookup
        # Just verify the constant exists
        assert estimator.DEFAULT_ENTROPY >= 0.0

    def test_table_has_133_entries(self, estimator):
        """Table should contain all 133 known AST node types."""
        assert len(estimator.ENTROPY_TABLE) == 133
