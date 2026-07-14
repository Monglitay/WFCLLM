from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from wfcllm.dynamic_semantic.calibration import ConformalCalibration
from wfcllm.dynamic_semantic.channel import (
    AggregateEvidence,
    SemanticChannel,
    UnitEvidence,
    aggregate_unit_evidence,
)
from wfcllm.dynamic_semantic.context import DynamicContextExtractor


@dataclass(frozen=True)
class R3DetectionResult:
    final_code_sha256: str
    parse_ok: bool
    evidence: tuple[UnitEvidence, ...]
    aggregate: AggregateEvidence
    p_value: float
    decision: bool
    erasure_counts: dict[str, int]

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "wfcllm-dynamic-semantic-r3-result/v3",
            "final_code_sha256": self.final_code_sha256,
            "parse_ok": self.parse_ok,
            "unit_ids": [item.unit_id for item in self.evidence],
            "context_sha256": [item.context_sha256 for item in self.evidence],
            "independent_units": self.aggregate.independent_units,
            "statistic_numerator": self.aggregate.numerator,
            "statistic_denominator": self.aggregate.denominator,
            "score": self.aggregate.score,
            "eligible": self.aggregate.eligible,
            "p_value": self.p_value,
            "decision": self.decision,
            "erasure_counts": dict(self.erasure_counts),
        }


@dataclass(frozen=True)
class ExactReplayAudit:
    exact: bool
    mismatches: tuple[str, ...]


class R3Detector:
    """Official detector: final source code is the only per-sample input."""

    def __init__(
        self,
        *,
        extractor: DynamicContextExtractor,
        encoder: Any,
        whitening: Any,
        channel: SemanticChannel,
        calibration: ConformalCalibration,
        minimum_independent_units: int,
    ) -> None:
        self._extractor = extractor
        self._encoder = encoder
        self._whitening = whitening
        self._channel = channel
        self._calibration = calibration
        self._minimum_independent_units = minimum_independent_units

    def detect_payload(self, payload: Mapping[str, Any]) -> R3DetectionResult:
        if not isinstance(payload, Mapping) or set(payload) != {"final_code"}:
            raise ValueError("R3 detector payload must contain only final_code")
        final_code = payload["final_code"]
        if not isinstance(final_code, str):
            raise ValueError("final_code must be a string")
        return self.detect(final_code)

    def detect(self, final_code: str) -> R3DetectionResult:
        if not isinstance(final_code, str):
            raise ValueError("final_code must be a string")
        extraction = self._extractor.extract(final_code)
        evidence: tuple[UnitEvidence, ...] = ()
        if extraction.contexts:
            embeddings = self._encoder.encode(
                tuple(context.serialized for context in extraction.contexts)
            )
            whitened = self._whitening.transform(embeddings)
            if len(whitened) != len(extraction.contexts):
                raise ValueError("whitening output row count does not match contexts")
            evidence = tuple(
                self._channel.score(
                    context.unit_id,
                    context.context_sha256,
                    vector.tolist(),
                )
                for context, vector in zip(
                    extraction.contexts,
                    whitened,
                    strict=True,
                )
            )
        aggregate = aggregate_unit_evidence(
            evidence,
            minimum_independent_units=self._minimum_independent_units,
        )
        p_value = self._calibration.p_value(aggregate.score)
        decision = self._calibration.decide(
            aggregate.score,
            eligible=aggregate.eligible,
        )
        return R3DetectionResult(
            final_code_sha256=hashlib.sha256(final_code.encode("utf-8")).hexdigest(),
            parse_ok=extraction.parse_ok,
            evidence=evidence,
            aggregate=aggregate,
            p_value=p_value,
            decision=decision,
            erasure_counts=extraction.erasure_counts,
        )


def compare_exact_replay(
    generation: tuple[UnitEvidence, ...],
    replay: tuple[UnitEvidence, ...],
) -> ExactReplayAudit:
    mismatches: list[str] = []
    if len(generation) != len(replay):
        mismatches.append("unit_count")
    for index, (left, right) in enumerate(zip(generation, replay)):
        fields = (
            "unit_id",
            "context_sha256",
            "quantized_embedding",
            "signature_bits",
            "target_bits",
            "matches",
            "numerator",
            "denominator",
        )
        for field in fields:
            if getattr(left, field) != getattr(right, field):
                mismatches.append(f"unit[{index}].{field}")
    return ExactReplayAudit(
        exact=not mismatches,
        mismatches=tuple(mismatches),
    )
