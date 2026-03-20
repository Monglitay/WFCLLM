"""Extract-side helpers for block contract rebuilding and comparison."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from wfcllm.common.block_contract import build_block_contracts

_STRUCTURAL_FIELDS = (
    "ordinal",
    "block_id",
    "node_type",
    "parent_node_type",
    "block_text_hash",
    "start_line",
    "end_line",
)
_NUMERIC_FIELDS = (
    "entropy_units",
    "gamma_target",
    "k",
    "gamma_effective",
)


@dataclass(frozen=True)
class AlignmentReport:
    """Comparison result between embedded and rebuilt block contracts."""

    embedded_block_count: int
    rebuilt_block_count: int
    block_count_mismatch: bool = False
    structure_mismatch: bool = False
    numeric_mismatch: bool = False
    structure_mismatches: list[dict[str, object]] = field(default_factory=list)
    numeric_mismatches: list[dict[str, object]] = field(default_factory=list)

    @property
    def contract_valid(self) -> bool:
        return not self.structure_mismatch

    @property
    def is_aligned(self) -> bool:
        return not self.structure_mismatch and not self.numeric_mismatch

    @property
    def status(self) -> str:
        if self.structure_mismatch and self.numeric_mismatch:
            return "structure_and_numeric_mismatch"
        if self.structure_mismatch:
            return "structure_mismatch"
        if self.numeric_mismatch:
            return "numeric_mismatch"
        return "aligned"

    def to_dict(self) -> dict[str, object]:
        return {
            "embedded_block_count": self.embedded_block_count,
            "rebuilt_block_count": self.rebuilt_block_count,
            "block_count_mismatch": self.block_count_mismatch,
            "structure_mismatch": self.structure_mismatch,
            "numeric_mismatch": self.numeric_mismatch,
            "contract_valid": self.contract_valid,
            "is_aligned": self.is_aligned,
            "status": self.status,
            "structure_mismatches": list(self.structure_mismatches),
            "numeric_mismatches": list(self.numeric_mismatches),
        }


def rebuild_block_contracts(code: str) -> list[dict[str, object]]:
    """Rebuild canonical simple-block contracts from final code."""
    return [asdict(contract) for contract in build_block_contracts(code)]


def compare_block_contracts(
    embedded_contracts: list[object],
    rebuilt_contracts: list[object],
) -> AlignmentReport:
    """Compare embedded block metadata against rebuilt final-code contracts."""
    embedded = [_normalize_contract(contract) for contract in embedded_contracts]
    rebuilt = [_normalize_contract(contract) for contract in rebuilt_contracts]

    structure_mismatches: list[dict[str, object]] = []
    numeric_mismatches: list[dict[str, object]] = []
    block_count_mismatch = len(embedded) != len(rebuilt)

    if block_count_mismatch:
        structure_mismatches.append(
            {
                "field": "block_count",
                "embedded": len(embedded),
                "rebuilt": len(rebuilt),
            }
        )

    for embedded_contract, rebuilt_contract in zip(embedded, rebuilt):
        ordinal = rebuilt_contract.get("ordinal", embedded_contract.get("ordinal"))
        for field_name in _STRUCTURAL_FIELDS:
            embedded_value = embedded_contract.get(field_name)
            rebuilt_value = rebuilt_contract.get(field_name)
            if embedded_value != rebuilt_value:
                structure_mismatches.append(
                    {
                        "ordinal": ordinal,
                        "field": field_name,
                        "embedded": embedded_value,
                        "rebuilt": rebuilt_value,
                    }
                )

        for field_name in _NUMERIC_FIELDS:
            embedded_value = embedded_contract.get(field_name)
            rebuilt_value = rebuilt_contract.get(field_name)
            if embedded_value != rebuilt_value:
                numeric_mismatches.append(
                    {
                        "ordinal": ordinal,
                        "field": field_name,
                        "embedded": embedded_value,
                        "rebuilt": rebuilt_value,
                    }
                )

    return AlignmentReport(
        embedded_block_count=len(embedded),
        rebuilt_block_count=len(rebuilt),
        block_count_mismatch=block_count_mismatch,
        structure_mismatch=bool(structure_mismatches),
        numeric_mismatch=bool(numeric_mismatches),
        structure_mismatches=structure_mismatches,
        numeric_mismatches=numeric_mismatches,
    )


def _normalize_contract(contract: object) -> dict[str, object]:
    if isinstance(contract, dict):
        normalized = dict(contract)
    else:
        normalized = {
            field_name: getattr(contract, field_name)
            for field_name in (
                *_STRUCTURAL_FIELDS,
                *_NUMERIC_FIELDS,
            )
            if hasattr(contract, field_name)
        }

    normalized.setdefault("gamma_target", 0.0)
    normalized.setdefault("k", 0)
    normalized.setdefault("gamma_effective", 0.0)
    return normalized
