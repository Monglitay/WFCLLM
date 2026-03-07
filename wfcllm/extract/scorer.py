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
