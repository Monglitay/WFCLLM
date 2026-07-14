from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from wfcllm.dynamic_semantic.config import ChannelConfig
from wfcllm.dynamic_semantic.keying import SecretKey, derive_bytes


BitTuple = tuple[int, ...]


def _validate_bits(bits: Sequence[int], length: int, name: str) -> tuple[int, ...]:
    normalized = tuple(bits)
    if len(normalized) != length or any(bit not in (0, 1) for bit in normalized):
        raise ValueError(f"{name} must contain exactly {length} binary bits")
    return normalized


def hamming_7_4_encode(data_bits: Sequence[int]) -> BitTuple:
    d1, d2, d3, d4 = _validate_bits(data_bits, 4, "data_bits")
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return (p1, p2, d1, p4, d2, d3, d4)


def hamming_7_4_syndrome(codeword: Sequence[int]) -> int:
    b1, b2, b3, b4, b5, b6, b7 = _validate_bits(
        codeword,
        7,
        "codeword",
    )
    s1 = b1 ^ b3 ^ b5 ^ b7
    s2 = b2 ^ b3 ^ b6 ^ b7
    s4 = b4 ^ b5 ^ b6 ^ b7
    return s1 + 2 * s2 + 4 * s4


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def quantize_vector(values: Iterable[float], scale: int) -> tuple[int, ...]:
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    quantized: list[int] = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("embedding values must be finite")
        quantized.append(_round_half_away_from_zero(numeric * scale))
    return tuple(quantized)


@dataclass(frozen=True)
class UnitEvidence:
    unit_id: str
    context_sha256: str
    quantized_embedding: tuple[int, ...]
    signature_bits: BitTuple
    target_bits: BitTuple
    matches: int
    numerator: int
    denominator: int = 7

    @classmethod
    def synthetic(cls, unit_id: str, *, matches: int) -> UnitEvidence:
        if not 0 <= matches <= 7:
            raise ValueError("matches must be in [0, 7]")
        return cls(
            unit_id=unit_id,
            context_sha256="synthetic",
            quantized_embedding=(),
            signature_bits=(0,) * 7,
            target_bits=(0,) * 7,
            matches=matches,
            numerator=2 * matches - 7,
        )


@dataclass(frozen=True)
class AggregateEvidence:
    numerator: int
    denominator: int
    independent_units: int
    eligible: bool

    @property
    def score(self) -> float:
        if self.denominator == 0:
            return 0.0
        return self.numerator / self.denominator


class SemanticChannel:
    """Secret post-encoder projection and content-addressed target channel."""

    def __init__(self, key: SecretKey, config: ChannelConfig) -> None:
        self._key = key
        self._config = config

    def score(
        self,
        unit_id: str,
        context_sha256: str,
        embedding: Iterable[float],
    ) -> UnitEvidence:
        quantized = quantize_vector(embedding, self._config.quantization_scale)
        if len(quantized) != self._config.whitening_dimensions:
            raise ValueError(
                "embedding dimensions must equal whitening_dimensions"
            )
        signature = self._signature_bits(quantized)
        target = self._target_bits(unit_id)
        matches = sum(left == right for left, right in zip(signature, target, strict=True))
        return UnitEvidence(
            unit_id=unit_id,
            context_sha256=context_sha256,
            quantized_embedding=quantized,
            signature_bits=signature,
            target_bits=target,
            matches=matches,
            numerator=2 * matches - self._config.projection_rows,
            denominator=self._config.projection_rows,
        )

    def _signature_bits(self, quantized: tuple[int, ...]) -> BitTuple:
        bits: list[int] = []
        for row_index in range(self._config.projection_rows):
            row_bytes = derive_bytes(
                self._key,
                domain="projection-row",
                message=row_index.to_bytes(4, "big"),
                length=len(quantized),
            )
            dot_product = sum(
                value * (1 if row_byte & 1 else -1)
                for value, row_byte in zip(quantized, row_bytes, strict=True)
            )
            bits.append(1 if dot_product >= 0 else 0)
        return tuple(bits)

    def _target_bits(self, unit_id: str) -> BitTuple:
        material = derive_bytes(
            self._key,
            domain="unit-target",
            message=unit_id.encode("ascii"),
            length=self._config.target_data_bits,
        )
        data_bits = tuple(byte & 1 for byte in material)
        return hamming_7_4_encode(data_bits)


def aggregate_unit_evidence(
    evidence: Iterable[UnitEvidence],
    *,
    minimum_independent_units: int,
) -> AggregateEvidence:
    if (
        isinstance(minimum_independent_units, bool)
        or not isinstance(minimum_independent_units, int)
        or minimum_independent_units <= 0
    ):
        raise ValueError("minimum_independent_units must be a positive integer")
    items = tuple(evidence)
    unit_ids = [item.unit_id for item in items]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("duplicate unit evidence is forbidden")
    numerator = sum(item.numerator for item in items)
    denominator = sum(item.denominator for item in items)
    return AggregateEvidence(
        numerator=numerator,
        denominator=denominator,
        independent_units=len(items),
        eligible=len(items) >= minimum_independent_units,
    )
