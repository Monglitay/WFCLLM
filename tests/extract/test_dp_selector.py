"""Tests for DPSelector."""

from __future__ import annotations

import pytest

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore
from wfcllm.extract.dp_selector import DPSelector


def _make_block(
    block_id: str,
    node_type: str = "expression_statement",
    block_type: str = "simple",
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    depth: int = 0,
) -> StatementBlock:
    return StatementBlock(
        block_id=block_id,
        block_type=block_type,
        node_type=node_type,
        source="x = 1",
        start_line=1,
        end_line=1,
        depth=depth,
        parent_id=parent_id,
        children_ids=children_ids or [],
    )


def _make_score(block_id: str, score: int) -> BlockScore:
    return BlockScore(
        block_id=block_id,
        score=score,
        projection=0.5 if score == 1 else -0.3,
        target_sign=1,
        selected=False,
    )


class TestDPSelector:
    @pytest.fixture
    def selector(self):
        return DPSelector()

    def test_single_leaf_block(self, selector):
        """Single block with no children -> selected."""
        blocks = [_make_block("0")]
        scores = [_make_score("0", score=1)]
        selected = selector.select(blocks, scores)
        assert selected == ["0"]

    def test_two_independent_roots(self, selector):
        """Two root blocks, both selected."""
        blocks = [_make_block("0"), _make_block("1")]
        scores = [_make_score("0", 1), _make_score("1", -1)]
        selected = selector.select(blocks, scores)
        assert set(selected) == {"0", "1"}

    def test_parent_beats_children(self, selector):
        """Parent score (+1) > sum of children scores (-1 + -1 = -2) -> select parent."""
        parent = _make_block(
            "0", node_type="for_statement", block_type="compound",
            children_ids=["1", "2"],
        )
        child1 = _make_block("1", parent_id="0", depth=1)
        child2 = _make_block("2", parent_id="0", depth=1)
        blocks = [parent, child1, child2]
        scores = [
            _make_score("0", 1),   # parent: +1
            _make_score("1", -1),  # child1: -1
            _make_score("2", -1),  # child2: -1
        ]
        selected = selector.select(blocks, scores)
        assert selected == ["0"]

    def test_children_beat_parent(self, selector):
        """Children sum (+1 + +1 = +2) > parent score (-1) -> select children."""
        parent = _make_block(
            "0", node_type="for_statement", block_type="compound",
            children_ids=["1", "2"],
        )
        child1 = _make_block("1", parent_id="0", depth=1)
        child2 = _make_block("2", parent_id="0", depth=1)
        blocks = [parent, child1, child2]
        scores = [
            _make_score("0", -1),  # parent: -1
            _make_score("1", 1),   # child1: +1
            _make_score("2", 1),   # child2: +1
        ]
        selected = selector.select(blocks, scores)
        assert set(selected) == {"1", "2"}

    def test_three_level_nesting(self, selector):
        """Three levels: grandparent -> parent -> child.

        grandparent (id=0, score=-1) has child parent (id=1, score=-1)
        which has child leaf (id=2, score=+1).

        Bottom-up:
          OPT(2) = +1
          OPT(1) = max(-1, OPT(2)) = max(-1, +1) = +1 -> use children
          OPT(0) = max(-1, OPT(1)) = max(-1, +1) = +1 -> use children

        Traceback: 0 -> children -> 1 -> children -> 2 (selected)
        """
        gp = _make_block(
            "0", node_type="for_statement", block_type="compound",
            children_ids=["1"],
        )
        p = _make_block(
            "1", node_type="if_statement", block_type="compound",
            parent_id="0", children_ids=["2"], depth=1,
        )
        c = _make_block("2", parent_id="1", depth=2)
        blocks = [gp, p, c]
        scores = [
            _make_score("0", -1),
            _make_score("1", -1),
            _make_score("2", 1),
        ]
        selected = selector.select(blocks, scores)
        assert selected == ["2"]

    def test_empty_blocks(self, selector):
        """No blocks -> empty selection."""
        assert selector.select([], []) == []

    def test_tie_prefers_children(self, selector):
        """When parent score == children sum, prefer children (more samples)."""
        parent = _make_block(
            "0", node_type="for_statement", block_type="compound",
            children_ids=["1"],
        )
        child = _make_block("1", parent_id="0", depth=1)
        blocks = [parent, child]
        scores = [_make_score("0", 1), _make_score("1", 1)]
        # parent=+1, children_sum=+1 -> tie -> prefer children
        selected = selector.select(blocks, scores)
        assert selected == ["1"]
