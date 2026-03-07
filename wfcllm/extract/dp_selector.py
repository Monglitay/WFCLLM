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
            children_in_dp = [cid for cid in block.children_ids if cid in dp]

            if not children_in_dp:
                # Leaf node: must use self (no alternative)
                dp[bid] = DPNode(bid, self_score, use_self=True)
            else:
                children_sum = sum(dp[cid].opt_score for cid in children_in_dp)
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
