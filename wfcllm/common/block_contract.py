"""Canonical block contracts shared by watermark embed and extract paths."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from wfcllm.common.ast_parser import extract_statement_blocks


@dataclass(frozen=True)
class BlockContract:
    ordinal: int
    block_id: str
    node_type: str
    parent_node_type: str | None
    block_text_hash: str
    start_line: int
    end_line: int
    entropy_units: int
    gamma_target: float = 0.0
    k: int = 0
    gamma_effective: float = 0.0


def build_block_contracts(code: str) -> list[BlockContract]:
    """Extract statement blocks and return canonical block contracts."""
    from wfcllm.watermark.entropy import NodeEntropyEstimator

    all_blocks = extract_statement_blocks(code)
    blocks = [
        block for block in all_blocks if block.block_type == "simple"
    ]
    block_type_by_id = {block.block_id: block.node_type for block in all_blocks}
    estimator = NodeEntropyEstimator()

    contracts: list[BlockContract] = []
    for ordinal, block in enumerate(blocks):
        parent_node_type = (
            block_type_by_id.get(block.parent_id)
            if block.parent_id is not None
            else "module"
        )
        contracts.append(
            BlockContract(
                ordinal=ordinal,
                block_id=block.block_id,
                node_type=block.node_type,
                parent_node_type=parent_node_type,
                block_text_hash=sha256(block.source.encode("utf-8")).hexdigest(),
                start_line=block.start_line,
                end_line=block.end_line,
                entropy_units=estimator.estimate_block_entropy_units(block.source),
            )
        )
    return contracts
