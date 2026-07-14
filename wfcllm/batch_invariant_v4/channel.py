from __future__ import annotations

from dataclasses import dataclass

from wfcllm.batch_invariant_v4.context import StructuralContext
from wfcllm.batch_invariant_v4.keying import V4SecretKey, derive_bytes


def _bits(material: bytes, count: int) -> tuple[int, ...]:
    values: list[int] = []
    for byte in material:
        values.extend((byte >> shift) & 1 for shift in range(8))
    return tuple(values[:count])


@dataclass(frozen=True)
class UnitEvidence:
    unit_id: str
    canonical_context_sha256: str
    representation: tuple[int, ...]
    quantized_values: tuple[int, ...]
    erasure_mask: tuple[bool, ...]
    signature_bits: tuple[int, ...]
    target_bits: tuple[int, ...]
    matches: int
    numerator: int
    denominator: int


class StructuralChannel:
    def __init__(
        self,
        key: V4SecretKey,
        *,
        bit_count: int,
        minimum_independent_units: int,
    ) -> None:
        if bit_count <= 0 or minimum_independent_units <= 0:
            raise ValueError("channel counts must be positive")
        self.key = key
        self.bit_count = bit_count
        self.minimum_independent_units = minimum_independent_units

    def score(self, context: StructuralContext) -> UnitEvidence:
        byte_count = (self.bit_count + 7) // 8
        signature = _bits(
            derive_bytes(
                self.key,
                domain="v4-formal/structural-signature",
                message=bytes(context.representation_bytes),
                length=byte_count,
            ),
            self.bit_count,
        )
        target = _bits(
            derive_bytes(
                self.key,
                domain="v4-formal/unit-target",
                message=context.unit_id.encode("ascii"),
                length=byte_count,
            ),
            self.bit_count,
        )
        matches = sum(
            left == right for left, right in zip(signature, target, strict=True)
        )
        representation = context.representation_bytes
        return UnitEvidence(
            unit_id=context.unit_id,
            canonical_context_sha256=context.context_sha256,
            representation=representation,
            quantized_values=representation,
            erasure_mask=(False,) * self.bit_count,
            signature_bits=signature,
            target_bits=target,
            matches=matches,
            numerator=2 * matches - self.bit_count,
            denominator=self.bit_count,
        )
