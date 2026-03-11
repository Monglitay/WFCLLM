"""Semantic feature scoring for statement blocks (LSH version)."""

from __future__ import annotations

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import ProjectionVerifier


class BlockScorer:
    """Score each statement block for watermark hit/miss via LSH."""

    def __init__(self, keying: WatermarkKeying, verifier: ProjectionVerifier):
        self._keying = keying
        self._verifier = verifier

    def score_block(
        self, block: StatementBlock, blocks: list[StatementBlock]
    ) -> BlockScore:
        parent_node_type = self._resolve_parent_type(block, blocks)
        valid_set = self._keying.derive(parent_node_type)
        result = self._verifier.verify(block.source, valid_set, 0.0)

        score = 1 if result.passed else 0
        return BlockScore(
            block_id=block.block_id,
            score=score,
            min_margin=result.min_margin,
        )

    def score_all(
        self,
        target_blocks: list[StatementBlock],
        all_blocks: list[StatementBlock],
    ) -> list[BlockScore]:
        """Score target blocks. all_blocks needed for parent_id → node_type lookup."""
        return [self.score_block(b, all_blocks) for b in target_blocks]

    @staticmethod
    def _resolve_parent_type(block: StatementBlock, blocks: list[StatementBlock]) -> str:
        if block.parent_id is None:
            return "module"
        block_map = {b.block_id: b for b in blocks}
        return block_map[block.parent_id].node_type
