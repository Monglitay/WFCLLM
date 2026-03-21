"""Tests for BlockScorer (LSH version)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import VerifyResult


def _make_block(
    block_id: str,
    node_type: str = "expression_statement",
    source: str = "x = 1",
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    depth: int = 0,
) -> StatementBlock:
    return StatementBlock(
        block_id=block_id,
        block_type="simple",
        node_type=node_type,
        source=source,
        start_line=1,
        end_line=1,
        depth=depth,
        parent_id=parent_id,
        children_ids=children_ids or [],
    )


class TestBlockScorer:
    @pytest.fixture
    def keying(self):
        return WatermarkKeying(secret_key="test-key", d=3, gamma=0.5)

    @pytest.fixture
    def mock_verifier(self):
        return MagicMock()

    def test_score_single_block_hit(self, keying, mock_verifier):
        """passed=True from verifier -> score = +1."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.4)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        result = scorer.score_block(block, blocks=[block])

        assert isinstance(result, BlockScore)
        assert result.block_id == "0"
        assert result.score == 1
        assert result.min_margin == 0.4

    def test_score_single_block_miss(self, keying, mock_verifier):
        """passed=False from verifier -> score = 0."""
        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.05)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        result = scorer.score_block(block, blocks=[block])

        assert result.score == 0
        assert result.min_margin == 0.05

    def test_verify_called_with_margin_zero(self, keying, mock_verifier):
        """Extraction always calls verify with margin=0.0."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        scorer.score_block(block, blocks=[block])

        call_args = mock_verifier.verify.call_args
        # verify(code_text, valid_set, margin)
        assert call_args[0][2] == 0.0

    def test_root_block_uses_module_parent(self, keying, mock_verifier):
        """Root-level block derives G from parent='module'."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0", parent_id=None)

        scorer.score_block(block, blocks=[block])

        expected_G = keying.derive("module")
        call_args = mock_verifier.verify.call_args
        actual_G = call_args[0][1]
        assert actual_G == expected_G

    def test_nested_block_uses_parent_node_type(self, keying, mock_verifier):
        """Nested block derives G from parent's node_type."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        parent = _make_block("0", node_type="for_statement", children_ids=["1"])
        child = _make_block("1", node_type="expression_statement", parent_id="0", depth=1)

        scorer.score_block(child, blocks=[parent, child])

        expected_G = keying.derive("for_statement")
        call_args = mock_verifier.verify.call_args
        actual_G = call_args[0][1]
        assert actual_G == expected_G

    def test_score_all_returns_all_blocks(self, keying, mock_verifier):
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        blocks = [_make_block("0"), _make_block("1")]
        # New: pass (target_blocks, all_blocks)
        results = scorer.score_all(blocks, blocks)

        assert len(results) == 2
        assert results[0].block_id == "0"
        assert results[1].block_id == "1"

    def test_score_block_uses_block_specific_k_when_present(self, mock_verifier):
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        keying = MagicMock()
        keying.derive.return_value = frozenset()
        scorer = BlockScorer(keying, mock_verifier, default_gamma=0.75)
        block = _make_block("0")

        result = scorer.score_block(
            block,
            blocks=[block],
            block_contract={"k": 6, "gamma_effective": 0.375},
        )

        keying.derive.assert_called_once_with("module", k=6)
        assert result.gamma_effective == pytest.approx(0.375)
