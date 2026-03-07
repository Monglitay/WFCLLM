"""Tests for BlockScorer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

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
        return WatermarkKeying(secret_key="test-key", embed_dim=128)

    @pytest.fixture
    def mock_verifier(self):
        verifier = MagicMock()
        return verifier

    def test_score_single_block_hit(self, keying, mock_verifier):
        """Block where projection sign matches target -> score = +1."""
        mock_verifier.verify.return_value = VerifyResult(
            passed=True, projection=0.5, target_sign=1, margin=0.0
        )
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        result = scorer.score_block(block, blocks=[block])

        assert isinstance(result, BlockScore)
        assert result.block_id == "0"
        assert result.score == 1
        assert result.projection == 0.5
        assert result.target_sign == 1
        # verify() called with margin=0.0
        _, kwargs = mock_verifier.verify.call_args
        assert kwargs.get("margin", mock_verifier.verify.call_args[0][3] if len(mock_verifier.verify.call_args[0]) > 3 else None) == 0.0 or mock_verifier.verify.call_args[0][3] == 0.0

    def test_score_single_block_miss(self, keying, mock_verifier):
        """Block where projection sign doesn't match -> score = -1."""
        mock_verifier.verify.return_value = VerifyResult(
            passed=False, projection=-0.3, target_sign=1, margin=0.0
        )
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        result = scorer.score_block(block, blocks=[block])

        assert result.score == -1
        assert result.projection == -0.3

    def test_root_block_uses_module_parent(self, keying, mock_verifier):
        """Root-level block (parent_id=None) should derive with parent='module'."""
        mock_verifier.verify.return_value = VerifyResult(
            passed=True, projection=0.5, target_sign=1, margin=0.0
        )
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0", parent_id=None)

        scorer.score_block(block, blocks=[block])

        # Check that verify was called with the direction vector derived
        # from ("module", "expression_statement")
        v_expected, t_expected = keying.derive("module", "expression_statement")
        call_args = mock_verifier.verify.call_args
        v_actual = call_args[0][1]
        t_actual = call_args[0][2]
        assert torch.allclose(v_actual, v_expected)
        assert t_actual == t_expected

    def test_nested_block_uses_parent_node_type(self, keying, mock_verifier):
        """Nested block should derive with parent's node_type."""
        mock_verifier.verify.return_value = VerifyResult(
            passed=True, projection=0.4, target_sign=1, margin=0.0
        )
        scorer = BlockScorer(keying, mock_verifier)
        parent = _make_block("0", node_type="for_statement", children_ids=["1"])
        child = _make_block(
            "1", node_type="expression_statement", parent_id="0", depth=1
        )
        blocks = [parent, child]

        scorer.score_block(child, blocks=blocks)

        v_expected, t_expected = keying.derive("for_statement", "expression_statement")
        call_args = mock_verifier.verify.call_args
        v_actual = call_args[0][1]
        assert torch.allclose(v_actual, v_expected)

    def test_score_all(self, keying, mock_verifier):
        """score_all returns a BlockScore for every block."""
        mock_verifier.verify.return_value = VerifyResult(
            passed=True, projection=0.5, target_sign=1, margin=0.0
        )
        scorer = BlockScorer(keying, mock_verifier)
        blocks = [_make_block("0"), _make_block("1")]
        results = scorer.score_all(blocks)

        assert len(results) == 2
        assert results[0].block_id == "0"
        assert results[1].block_id == "1"
