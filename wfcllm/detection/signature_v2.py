from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from typing import Any

from wfcllm.detection.canonical_v2 import CanonicalUnit, extract_canonical_units
from wfcllm.semantic.lsh import load_semantic_lsh_components

_TARGET_DOMAIN = b"wfcllm-aligned-canonical-signature/v2\x00"

TRIMMED_UNIT_MEAN = "trimmed_unit_mean"
STANDARDIZED_BIT_SUM = "standardized_bit_sum"
SUPPORTED_AGGREGATIONS = frozenset({TRIMMED_UNIT_MEAN, STANDARDIZED_BIT_SUM})


@dataclass(frozen=True)
class V2UnitEvidence:
    unit_id: str
    normalized_text_hash: str
    node_type: str
    parent_node_type: str
    ordinal: int
    actual_signature: tuple[int, ...]
    target_signature: tuple[int, ...]
    matched_bits: int
    total_bits: int
    unit_score: float


@dataclass(frozen=True)
class V2CodeScore:
    raw_score: float
    unit_count: int
    duplicate_count: int
    total_bits: int
    matched_bits: int
    unit_evidence: tuple[V2UnitEvidence, ...]


def derive_target_signature(
    unit: CanonicalUnit,
    *,
    secret_key: str,
    bits: int,
) -> tuple[int, ...]:
    _validate_signature_bits(bits)
    if not isinstance(secret_key, str) or not secret_key:
        raise ValueError("secret_key must be non-empty")
    payload = json.dumps(
        {
            "schema_version": unit.schema_version,
            "node_type": unit.node_type,
            "parent_node_type": unit.parent_node_type,
            "ordinal": unit.ordinal,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hmac.new(
        secret_key.encode("utf-8"),
        _TARGET_DOMAIN + payload,
        hashlib.sha256,
    ).digest()
    return tuple(
        (digest[index // 8] >> (7 - index % 8)) & 1
        for index in range(bits)
    )


def robust_unit_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) >= 5:
        ordered = ordered[1:-1]
    return sum(ordered) / len(ordered)


def standardized_bit_sum(*, matched_bits: int, total_bits: int) -> float:
    """Return a null-standardized sum over unique canonical signature bits."""

    if isinstance(total_bits, bool) or not isinstance(total_bits, int) or total_bits < 0:
        raise ValueError("total_bits must be a non-negative integer")
    if (
        isinstance(matched_bits, bool)
        or not isinstance(matched_bits, int)
        or not 0 <= matched_bits <= total_bits
    ):
        raise ValueError("matched_bits must be an integer in [0, total_bits]")
    if total_bits == 0:
        return 0.0
    expected = total_bits / 2.0
    null_standard_deviation = math.sqrt(total_bits / 4.0)
    return (matched_bits - expected) / null_standard_deviation


class CanonicalSignatureScorer:
    def __init__(
        self,
        *,
        verifier: Any,
        secret_key: str,
        signature_bits: int = 16,
        aggregation: str = TRIMMED_UNIT_MEAN,
    ) -> None:
        _validate_signature_bits(signature_bits)
        if not isinstance(secret_key, str) or not secret_key:
            raise ValueError("secret_key must be non-empty")
        self._verifier = verifier
        self._secret_key = secret_key
        self._signature_bits = signature_bits
        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise ValueError(f"unsupported V2 aggregation: {aggregation!r}")
        self._aggregation = aggregation

    def score_code(self, final_code: str) -> V2CodeScore:
        extraction = extract_canonical_units(final_code)
        evidence: list[V2UnitEvidence] = []
        for unit in extraction.units:
            target = derive_target_signature(
                unit,
                secret_key=self._secret_key,
                bits=self._signature_bits,
            )
            result = self._verifier.verify(
                unit.normalized_text,
                frozenset({target}),
                0.0,
            )
            actual = tuple(int(bit) for bit in result.lsh_signature)
            if len(actual) != self._signature_bits:
                raise ValueError(
                    "verifier signature length does not match signature_bits"
                )
            matched_bits = sum(
                1 for actual_bit, target_bit in zip(actual, target)
                if actual_bit == target_bit
            )
            evidence.append(
                V2UnitEvidence(
                    unit_id=unit.unit_id,
                    normalized_text_hash=unit.normalized_text_hash,
                    node_type=unit.node_type,
                    parent_node_type=unit.parent_node_type,
                    ordinal=unit.ordinal,
                    actual_signature=actual,
                    target_signature=target,
                    matched_bits=matched_bits,
                    total_bits=self._signature_bits,
                    unit_score=matched_bits / self._signature_bits,
                )
            )

        matched_bits = sum(item.matched_bits for item in evidence)
        total_bits = len(evidence) * self._signature_bits
        if self._aggregation == STANDARDIZED_BIT_SUM:
            raw_score = standardized_bit_sum(
                matched_bits=matched_bits,
                total_bits=total_bits,
            )
        else:
            raw_score = robust_unit_mean([item.unit_score for item in evidence])
        return V2CodeScore(
            raw_score=raw_score,
            unit_count=len(evidence),
            duplicate_count=extraction.duplicate_count,
            total_bits=total_bits,
            matched_bits=matched_bits,
            unit_evidence=tuple(evidence),
        )


def load_v2_signature_scorer(
    *,
    encoder_model_path: str,
    encoder_checkpoint_path: str | None,
    embed_dim: int,
    device: str,
    use_lora: bool,
    use_bf16: bool,
    secret_key: str,
    signature_bits: int,
    whitening_path: str | None = None,
    aggregation: str = TRIMMED_UNIT_MEAN,
) -> CanonicalSignatureScorer:
    """Load the exact encoder/LSH stack shared by V2 generation and detection."""

    _validate_signature_bits(signature_bits)
    components = load_semantic_lsh_components(
        encoder_model_path=encoder_model_path,
        encoder_checkpoint_path=encoder_checkpoint_path,
        embed_dim=embed_dim,
        device=device,
        use_lora=use_lora,
        use_bf16=use_bf16,
        secret_key=secret_key,
        lsh_d=signature_bits,
        whitening_path=whitening_path,
    )
    return CanonicalSignatureScorer(
        verifier=components.verifier,
        secret_key=secret_key,
        signature_bits=signature_bits,
        aggregation=aggregation,
    )


def _validate_signature_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 128:
        raise ValueError("signature_bits must be an integer in [2, 128]")
