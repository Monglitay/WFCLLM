# Phase 3 Extract & Verify Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the watermark extraction and verification pipeline (`wfcllm/extract/`) that detects watermarks in code via AST parsing, semantic projection scoring, DP deduplication, and Z-score hypothesis testing.

**Architecture:** Pipeline of 4 independent components (BlockScorer → DPSelector → HypothesisTester) orchestrated by WatermarkDetector. Reuses Phase 2's `WatermarkKeying` and `ProjectionVerifier` directly. All new code lives in `wfcllm/extract/`.

**Tech Stack:** Python 3.10+, dataclasses, torch, scipy.stats, tree-sitter (via `wfcllm.common.ast_parser`), pytest + MagicMock for testing.

**Design doc:** `docs/plans/2026-03-07-phase3-extract-design.md`

---

### Task 1: Config and Data Structures

**Files:**
- Create: `wfcllm/extract/__init__.py` (empty placeholder)
- Create: `wfcllm/extract/config.py`
- Create: `tests/extract/__init__.py` (empty)
- Create: `tests/extract/test_config.py`

**Step 1: Write the failing test**

Create `tests/extract/__init__.py` (empty) and `tests/extract/test_config.py`:

```python
"""Tests for extract config and data structures."""

from __future__ import annotations

from wfcllm.extract.config import BlockScore, DetectionResult, ExtractConfig


class TestExtractConfig:
    def test_defaults(self):
        cfg = ExtractConfig(secret_key="test-key")
        assert cfg.secret_key == "test-key"
        assert cfg.embed_dim == 128
        assert cfg.z_threshold == 3.0

    def test_custom_threshold(self):
        cfg = ExtractConfig(secret_key="k", z_threshold=2.5)
        assert cfg.z_threshold == 2.5


class TestBlockScore:
    def test_fields(self):
        bs = BlockScore(
            block_id="0",
            score=1,
            projection=0.42,
            target_sign=1,
            selected=False,
        )
        assert bs.block_id == "0"
        assert bs.score == 1
        assert bs.projection == 0.42
        assert bs.target_sign == 1
        assert bs.selected is False


class TestDetectionResult:
    def test_watermarked(self):
        dr = DetectionResult(
            is_watermarked=True,
            z_score=3.5,
            p_value=0.0002,
            total_blocks=10,
            independent_blocks=6,
            hit_blocks=5,
            block_details=[],
        )
        assert dr.is_watermarked is True
        assert dr.z_score == 3.5
        assert dr.independent_blocks == 6
        assert dr.hit_blocks == 5
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/extract/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.extract.config'`

**Step 3: Write minimal implementation**

Create `wfcllm/extract/__init__.py` (empty placeholder):

```python
```

Create `wfcllm/extract/config.py`:

```python
"""Configuration and data structures for watermark extraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractConfig:
    """Configuration for the watermark extraction pipeline."""

    secret_key: str
    embed_dim: int = 128
    z_threshold: float = 3.0


@dataclass
class BlockScore:
    """Score result for a single statement block."""

    block_id: str
    score: int  # +1 (hit) or -1 (miss)
    projection: float
    target_sign: int  # -1 or +1
    selected: bool = False


@dataclass
class DetectionResult:
    """Final watermark detection result."""

    is_watermarked: bool
    z_score: float
    p_value: float
    total_blocks: int
    independent_blocks: int
    hit_blocks: int
    block_details: list[BlockScore] = field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/extract/test_config.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add wfcllm/extract/__init__.py wfcllm/extract/config.py tests/extract/__init__.py tests/extract/test_config.py
git commit -m "feat(extract): add config and data structures"
```

---

### Task 2: BlockScorer

**Files:**
- Create: `wfcllm/extract/scorer.py`
- Create: `tests/extract/test_scorer.py`

**Context:**
- `BlockScorer` takes a list of `StatementBlock` (from `wfcllm.common.ast_parser`) and produces a `BlockScore` for each.
- It reuses `WatermarkKeying` (from `wfcllm.watermark.keying`) to derive `(v, t)` and `ProjectionVerifier` (from `wfcllm.watermark.verifier`) to compute cosine projection.
- Extraction phase only checks projection **sign**, not margin. Pass `margin=0.0` to `verify()`.
- Root-level blocks (`parent_id is None`) use `"module"` as parent_node_type.
- Non-root blocks: look up parent's `node_type` from the blocks list via `parent_id`.

**Step 1: Write the failing test**

Create `tests/extract/test_scorer.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/extract/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.extract.scorer'`

**Step 3: Write minimal implementation**

Create `wfcllm/extract/scorer.py`:

```python
"""Semantic feature scoring for statement blocks."""

from __future__ import annotations

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import ProjectionVerifier


class BlockScorer:
    """Score each statement block for watermark hit/miss."""

    def __init__(self, keying: WatermarkKeying, verifier: ProjectionVerifier):
        self._keying = keying
        self._verifier = verifier

    def score_block(
        self, block: StatementBlock, blocks: list[StatementBlock]
    ) -> BlockScore:
        """Score a single block.

        Args:
            block: The statement block to score.
            blocks: All blocks (needed to resolve parent_node_type).
        """
        parent_node_type = self._resolve_parent_type(block, blocks)
        v, t = self._keying.derive(parent_node_type, block.node_type)
        result = self._verifier.verify(block.source, v, t, 0.0)

        target_sign = 2 * t - 1
        sign_match = (result.projection > 0 and target_sign == 1) or (
            result.projection < 0 and target_sign == -1
        )
        score = 1 if sign_match else -1

        return BlockScore(
            block_id=block.block_id,
            score=score,
            projection=result.projection,
            target_sign=target_sign,
            selected=False,
        )

    def score_all(self, blocks: list[StatementBlock]) -> list[BlockScore]:
        """Score all blocks."""
        return [self.score_block(b, blocks) for b in blocks]

    @staticmethod
    def _resolve_parent_type(
        block: StatementBlock, blocks: list[StatementBlock]
    ) -> str:
        if block.parent_id is None:
            return "module"
        block_map = {b.block_id: b for b in blocks}
        return block_map[block.parent_id].node_type
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/extract/test_scorer.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add wfcllm/extract/scorer.py tests/extract/test_scorer.py
git commit -m "feat(extract): add BlockScorer for semantic projection scoring"
```

---

### Task 3: DPSelector

**Files:**
- Create: `wfcllm/extract/dp_selector.py`
- Create: `tests/extract/test_dp_selector.py`

**Context:**
This is the core new algorithm. It eliminates nesting-induced score duplication via bottom-up DP, then traces back top-down to select a non-overlapping independent set. The `StatementBlock` dataclass has `parent_id` (str | None), `children_ids` (list[str]), and `depth` (int) which provide the tree structure.

Algorithm:
- `OPT(i) = max(S_i, Σ OPT(j) for j in children(i))`
- Leaf nodes: `OPT(i) = S_i`
- Bottom-up: process by descending depth
- Top-down traceback: start from roots, if `use_self` add to set, else recurse into children

**Step 1: Write the failing test**

Create `tests/extract/test_dp_selector.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/extract/test_dp_selector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.extract.dp_selector'`

**Step 3: Write minimal implementation**

Create `wfcllm/extract/dp_selector.py`:

```python
"""Dynamic programming deduplication for nested AST blocks."""

from __future__ import annotations

from dataclasses import dataclass

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore


@dataclass
class DPNode:
    """DP state for a single block."""

    block_id: str
    opt_score: float
    use_self: bool


class DPSelector:
    """Select non-overlapping independent blocks via bottom-up DP."""

    def select(
        self, blocks: list[StatementBlock], scores: list[BlockScore]
    ) -> list[str]:
        """Return block_ids of the optimal non-overlapping set.

        Args:
            blocks: All statement blocks with parent/child references.
            scores: Corresponding scores for each block.

        Returns:
            List of block_ids forming the independent sample set.
        """
        if not blocks:
            return []

        score_map = {s.block_id: s.score for s in scores}
        block_map = {b.block_id: b for b in blocks}
        dp: dict[str, DPNode] = {}

        # Bottom-up: process deeper nodes first
        for block in sorted(blocks, key=lambda b: b.depth, reverse=True):
            bid = block.block_id
            self_score = score_map[bid]
            children_sum = sum(
                dp[cid].opt_score for cid in block.children_ids if cid in dp
            )

            # Prefer children on tie (more independent samples)
            if self_score > children_sum:
                dp[bid] = DPNode(bid, self_score, use_self=True)
            else:
                dp[bid] = DPNode(bid, children_sum, use_self=False)

        # Top-down traceback from roots
        selected: list[str] = []
        roots = [b for b in blocks if b.parent_id is None]
        self._traceback(roots, dp, block_map, selected)
        return selected

    def _traceback(
        self,
        nodes: list[StatementBlock],
        dp: dict[str, DPNode],
        block_map: dict[str, StatementBlock],
        selected: list[str],
    ) -> None:
        for node in nodes:
            dp_node = dp[node.block_id]
            if dp_node.use_self:
                selected.append(node.block_id)
            else:
                children = [
                    block_map[cid]
                    for cid in node.children_ids
                    if cid in block_map
                ]
                self._traceback(children, dp, block_map, selected)
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/extract/test_dp_selector.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add wfcllm/extract/dp_selector.py tests/extract/test_dp_selector.py
git commit -m "feat(extract): add DPSelector for nested block deduplication"
```

---

### Task 4: HypothesisTester

**Files:**
- Create: `wfcllm/extract/hypothesis.py`
- Create: `tests/extract/test_hypothesis.py`

**Context:**
- `M` = number of independent blocks, `X` = hit count (score == +1)
- `Z = (X - M/2) / sqrt(M/4)` — one-sided normal test
- `p_value = scipy.stats.norm.sf(z_score)` (survival function)
- `is_watermarked = Z > z_threshold`
- Edge case: M == 0 → `is_watermarked=False, z_score=0.0, p_value=1.0`

**Step 1: Write the failing test**

Create `tests/extract/test_hypothesis.py`:

```python
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
        projection=0.5 if score == 1 else -0.3,
        target_sign=1,
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
        scores += [_make_score(str(i + 10), -1) for i in range(10)]
        result = tester.test(selected_scores=scores, total_blocks=20)

        assert result.z_score == pytest.approx(0.0)
        assert result.is_watermarked is False

    def test_custom_threshold(self):
        """Lower threshold makes detection easier."""
        tester = HypothesisTester(z_threshold=1.0)
        # 15 hits out of 20: Z = (15 - 10)/sqrt(5) ≈ 2.24
        scores = [_make_score(str(i), 1) for i in range(15)]
        scores += [_make_score(str(i + 15), -1) for i in range(5)]
        result = tester.test(selected_scores=scores, total_blocks=20)

        assert result.z_score > 1.0
        assert result.is_watermarked is True

    def test_p_value_decreases_with_z(self, tester):
        """Higher Z-score means lower p-value."""
        scores_high = [_make_score(str(i), 1) for i in range(20)]
        scores_low = [_make_score(str(i), 1) for i in range(12)]
        scores_low += [_make_score(str(i + 12), -1) for i in range(8)]

        r_high = tester.test(selected_scores=scores_high, total_blocks=20)
        r_low = tester.test(selected_scores=scores_low, total_blocks=20)

        assert r_high.p_value < r_low.p_value

    def test_result_includes_block_details(self, tester):
        """DetectionResult.block_details contains the input scores."""
        scores = [_make_score("0", 1), _make_score("1", -1)]
        result = tester.test(selected_scores=scores, total_blocks=5)
        assert len(result.block_details) == 2
        assert result.block_details[0].block_id == "0"
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/extract/test_hypothesis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.extract.hypothesis'`

**Step 3: Write minimal implementation**

Create `wfcllm/extract/hypothesis.py`:

```python
"""Z-score hypothesis testing for watermark detection."""

from __future__ import annotations

import math

from scipy.stats import norm

from wfcllm.extract.config import BlockScore, DetectionResult


class HypothesisTester:
    """One-sided Z-test for watermark presence."""

    def __init__(self, z_threshold: float = 3.0):
        self._z_threshold = z_threshold

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
        if m == 0:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                total_blocks=total_blocks,
                independent_blocks=0,
                hit_blocks=0,
                block_details=list(selected_scores),
            )

        x = sum(1 for s in selected_scores if s.score == 1)
        z_score = (x - m / 2) / math.sqrt(m / 4)
        p_value = float(norm.sf(z_score))

        return DetectionResult(
            is_watermarked=z_score > self._z_threshold,
            z_score=z_score,
            p_value=p_value,
            total_blocks=total_blocks,
            independent_blocks=m,
            hit_blocks=x,
            block_details=list(selected_scores),
        )
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/extract/test_hypothesis.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add wfcllm/extract/hypothesis.py tests/extract/test_hypothesis.py
git commit -m "feat(extract): add HypothesisTester for Z-score significance testing"
```

---

### Task 5: WatermarkDetector

**Files:**
- Create: `wfcllm/extract/detector.py`
- Create: `tests/extract/test_detector.py`

**Context:**
- `WatermarkDetector` is the high-level entry point that wires together all components.
- It takes `ExtractConfig`, an encoder, and a tokenizer.
- Internally constructs `WatermarkKeying`, `ProjectionVerifier`, `BlockScorer`, `DPSelector`, `HypothesisTester`.
- The `detect(code)` method runs the full pipeline.
- `block_details` in the result should contain ALL blocks (not just selected ones), with the `selected` field set by DP.

**Step 1: Write the failing test**

Create `tests/extract/test_detector.py`:

```python
"""Tests for WatermarkDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from wfcllm.extract.config import DetectionResult, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector


class TestWatermarkDetector:
    @pytest.fixture
    def config(self):
        return ExtractConfig(secret_key="test-key", embed_dim=128, z_threshold=3.0)

    @pytest.fixture
    def mock_encoder(self):
        encoder = MagicMock()
        vec = torch.randn(1, 128)
        vec = vec / vec.norm()
        encoder.return_value = vec
        encoder.eval = MagicMock(return_value=encoder)
        return encoder

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        return tokenizer

    def test_detect_returns_detection_result(
        self, config, mock_encoder, mock_tokenizer
    ):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "x = 1\ny = 2\n"
        result = detector.detect(code)

        assert isinstance(result, DetectionResult)
        assert result.total_blocks >= 2
        assert result.independent_blocks >= 0
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value, float)

    def test_detect_empty_code(self, config, mock_encoder, mock_tokenizer):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        result = detector.detect("")

        assert result.is_watermarked is False
        assert result.total_blocks == 0
        assert result.independent_blocks == 0

    def test_block_details_include_all_blocks(
        self, config, mock_encoder, mock_tokenizer
    ):
        """block_details should contain all blocks, not just selected."""
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
        result = detector.detect(code)

        # Should have compound (for) + 2 simple children = 3 blocks
        assert result.total_blocks == len(result.block_details)
        assert result.total_blocks >= 3

    def test_selected_flag_set(self, config, mock_encoder, mock_tokenizer):
        """At least some blocks should have selected=True after DP."""
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "x = 1\ny = 2\nz = 3\n"
        result = detector.detect(code)

        selected_count = sum(1 for b in result.block_details if b.selected)
        assert selected_count == result.independent_blocks
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/extract/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.extract.detector'`

**Step 3: Write minimal implementation**

Create `wfcllm/extract/detector.py`:

```python
"""High-level watermark detection entry point."""

from __future__ import annotations

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.config import DetectionResult, ExtractConfig
from wfcllm.extract.dp_selector import DPSelector
from wfcllm.extract.hypothesis import HypothesisTester
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import ProjectionVerifier


class WatermarkDetector:
    """One-call watermark detection pipeline."""

    def __init__(
        self,
        config: ExtractConfig,
        encoder,
        tokenizer,
        device: str = "cuda",
    ):
        keying = WatermarkKeying(config.secret_key, config.embed_dim)
        verifier = ProjectionVerifier(encoder, tokenizer, device=device)
        self._scorer = BlockScorer(keying, verifier)
        self._dp = DPSelector()
        self._tester = HypothesisTester(config.z_threshold)

    def detect(self, code: str) -> DetectionResult:
        """Run the full extraction and verification pipeline.

        Args:
            code: Python source code to test for watermarks.

        Returns:
            DetectionResult with verdict, statistics, and per-block details.
        """
        blocks = extract_statement_blocks(code)
        if not blocks:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                total_blocks=0,
                independent_blocks=0,
                hit_blocks=0,
                block_details=[],
            )

        scores = self._scorer.score_all(blocks)
        selected_ids = set(self._dp.select(blocks, scores))

        # Mark selected blocks
        for s in scores:
            s.selected = s.block_id in selected_ids

        selected_scores = [s for s in scores if s.selected]
        result = self._tester.test(selected_scores, total_blocks=len(blocks))

        # Replace block_details with ALL blocks (not just selected)
        result.block_details = scores
        return result
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/extract/test_detector.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add wfcllm/extract/detector.py tests/extract/test_detector.py
git commit -m "feat(extract): add WatermarkDetector high-level entry point"
```

---

### Task 6: Public API exports

**Files:**
- Modify: `wfcllm/extract/__init__.py`

**Step 1: Write the failing test**

No separate test file — test the imports directly:

Add to `tests/extract/test_config.py` at the top of the file, add a new test class:

```python
class TestPublicAPI:
    def test_high_level_imports(self):
        from wfcllm.extract import (
            DetectionResult,
            ExtractConfig,
            WatermarkDetector,
        )

    def test_low_level_imports(self):
        from wfcllm.extract import (
            BlockScore,
            BlockScorer,
            DPSelector,
            HypothesisTester,
        )
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/extract/test_config.py::TestPublicAPI -v`
Expected: FAIL with `ImportError: cannot import name 'WatermarkDetector' from 'wfcllm.extract'`

**Step 3: Write minimal implementation**

Update `wfcllm/extract/__init__.py`:

```python
"""Watermark extraction and verification module."""

from wfcllm.extract.config import BlockScore, DetectionResult, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector
from wfcllm.extract.dp_selector import DPSelector
from wfcllm.extract.hypothesis import HypothesisTester
from wfcllm.extract.scorer import BlockScorer

__all__ = [
    "ExtractConfig",
    "DetectionResult",
    "BlockScore",
    "WatermarkDetector",
    "BlockScorer",
    "DPSelector",
    "HypothesisTester",
]
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/extract/test_config.py::TestPublicAPI -v`
Expected: 2 PASSED

**Step 5: Run full test suite and commit**

Run: `conda run -n WFCLLM pytest tests/extract/ -v`
Expected: ALL PASSED (17 tests total across 4 test files)

```bash
git add wfcllm/extract/__init__.py tests/extract/test_config.py
git commit -m "feat(extract): complete Phase 3 module with public API exports"
```

---

### Task 7: Final verification

**Step 1: Run the complete project test suite**

Run: `conda run -n WFCLLM pytest tests/ -v`
Expected: ALL PASSED — no regressions in Phase 1 or Phase 2 tests.

**Step 2: Verify no cross-phase import violations**

Run: `grep -r "from experiment" wfcllm/`
Expected: No output (no imports from experiment/).

**Step 3: No commit needed — verification only.**
