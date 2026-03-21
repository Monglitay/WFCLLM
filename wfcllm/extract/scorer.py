"""Semantic feature scoring for statement blocks (LSH version)."""

from __future__ import annotations

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import ProjectionVerifier


class BlockScorer:
    """Score each statement block for watermark hit/miss via LSH."""

    def __init__(
        self,
        keying: WatermarkKeying,
        verifier: ProjectionVerifier,
        default_gamma: float = 0.5,
    ):
        self._keying = keying
        self._verifier = verifier
        self._default_gamma = default_gamma

    def score_block(
        self,
        block: StatementBlock,
        blocks: list[StatementBlock],
        block_contract: dict | None = None,
    ) -> BlockScore:
        parent_node_type = self._resolve_parent_type(block, blocks)
        k = self._resolve_k(block_contract)
        if k is None:
            valid_set = self._keying.derive(parent_node_type)
        else:
            valid_set = self._keying.derive(parent_node_type, k=k)
        result = self._verifier.verify(block.source, valid_set, 0.0)

        score = 1 if result.passed else 0
        return BlockScore(
            block_id=block.block_id,
            score=score,
            min_margin=result.min_margin,
            gamma_effective=self._resolve_gamma_effective(block_contract),
        )

    def score_all(
        self,
        target_blocks: list[StatementBlock],
        all_blocks: list[StatementBlock],
        block_contracts_by_id: dict[str, dict] | None = None,
    ) -> list[BlockScore]:
        """Score target blocks. all_blocks needed for parent_id → node_type lookup."""
        return [
            self.score_block(
                b,
                all_blocks,
                block_contract=(
                    block_contracts_by_id.get(b.block_id)
                    if block_contracts_by_id is not None
                    else None
                ),
            )
            for b in target_blocks
        ]

    @staticmethod
    def _resolve_parent_type(block: StatementBlock, blocks: list[StatementBlock]) -> str:
        if block.parent_id is None:
            return "module"
        block_map = {b.block_id: b for b in blocks}
        return block_map[block.parent_id].node_type

    def _resolve_gamma_effective(self, block_contract: dict | None) -> float:
        if isinstance(block_contract, dict) and "gamma_effective" in block_contract:
            return float(block_contract["gamma_effective"])
        return self._default_gamma

    @staticmethod
    def _resolve_k(block_contract: dict | None) -> int | None:
        if not isinstance(block_contract, dict):
            return None
        k = block_contract.get("k")
        if isinstance(k, int) and k > 0:
            return k
        return None
