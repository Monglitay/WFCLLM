from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from wfcllm.batch_invariant_v4.calibration import CalibrationArtifact
from wfcllm.batch_invariant_v4.channel import StructuralChannel, UnitEvidence
from wfcllm.batch_invariant_v4.context import StructuralContextExtractor


@dataclass(frozen=True)
class DetectorPayload:
    final_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.final_code, str):
            raise ValueError("final_code must be a string")

    @classmethod
    def from_dict(cls, payload: Any) -> DetectorPayload:
        if not isinstance(payload, dict) or set(payload) != {"final_code"}:
            raise ValueError("R3 detector payload must contain exactly final_code")
        return cls(final_code=payload["final_code"])


@dataclass(frozen=True)
class CodeEvidence:
    units: tuple[UnitEvidence, ...]
    erasure_counts: dict[str, int]
    aggregate_numerator: int
    aggregate_denominator: int
    independent_units: int
    eligible: bool
    p_value: float
    decision: bool

    @property
    def score(self) -> float:
        return (
            self.aggregate_numerator / self.aggregate_denominator
            if self.aggregate_denominator
            else 0.0
        )


@dataclass(frozen=True)
class UnthresholdedEvidence:
    units: tuple[UnitEvidence, ...]
    erasure_counts: dict[str, int]
    aggregate_numerator: int
    aggregate_denominator: int
    independent_units: int
    eligible: bool

    @property
    def score(self) -> float:
        return (
            self.aggregate_numerator / self.aggregate_denominator
            if self.aggregate_denominator
            else 0.0
        )


def reconstruct_unthresholded(
    *,
    extractor: StructuralContextExtractor,
    channel: StructuralChannel,
    final_code: str,
) -> UnthresholdedEvidence:
    extraction = extractor.extract(final_code)
    units = tuple(
        sorted(
            (channel.score(context) for context in extraction.contexts),
            key=lambda unit: unit.unit_id,
        )
    )
    return UnthresholdedEvidence(
        units=units,
        erasure_counts=extraction.erasure_counts,
        aggregate_numerator=sum(unit.numerator for unit in units),
        aggregate_denominator=sum(unit.denominator for unit in units),
        independent_units=len(units),
        eligible=len(units) >= channel.minimum_independent_units,
    )


class V4Detector:
    def __init__(
        self,
        *,
        extractor: StructuralContextExtractor,
        channel: StructuralChannel,
        calibration: CalibrationArtifact,
    ) -> None:
        if (
            channel.minimum_independent_units
            != calibration.minimum_independent_units
        ):
            raise ValueError("channel and calibration minimum units differ")
        self.extractor = extractor
        self.channel = channel
        self.calibration = calibration

    def detect(self, payload: DetectorPayload) -> CodeEvidence:
        if not isinstance(payload, DetectorPayload):
            raise ValueError("payload must be DetectorPayload")
        raw = reconstruct_unthresholded(
            extractor=self.extractor,
            channel=self.channel,
            final_code=payload.final_code,
        )
        score = raw.score
        p_value = self.calibration.empirical_p_value(score)
        if not math.isfinite(p_value):
            raise ValueError("calibration produced non-finite p-value")
        return CodeEvidence(
            units=raw.units,
            erasure_counts=raw.erasure_counts,
            aggregate_numerator=raw.aggregate_numerator,
            aggregate_denominator=raw.aggregate_denominator,
            independent_units=raw.independent_units,
            eligible=raw.eligible,
            p_value=p_value,
            decision=self.calibration.decide(
                score=score,
                independent_units=raw.independent_units,
            ),
        )


_UNIT_FIELDS = (
    "unit_id",
    "canonical_context_sha256",
    "representation",
    "quantized_values",
    "erasure_mask",
    "signature_bits",
    "target_bits",
    "matches",
    "numerator",
    "denominator",
)
_CODE_FIELDS = (
    "erasure_counts",
    "aggregate_numerator",
    "aggregate_denominator",
    "independent_units",
    "eligible",
    "p_value",
    "decision",
)


def exact_code_evidence_mismatches(
    reference: CodeEvidence,
    candidate: CodeEvidence,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if len(reference.units) != len(candidate.units):
        mismatches.append("units.length")
    for index, (left, right) in enumerate(
        zip(reference.units, candidate.units, strict=False)
    ):
        for field in _UNIT_FIELDS:
            if getattr(left, field) != getattr(right, field):
                mismatches.append(f"units[{index}].{field}")
    for field in _CODE_FIELDS:
        if getattr(reference, field) != getattr(candidate, field):
            mismatches.append(field)
    return tuple(mismatches)
