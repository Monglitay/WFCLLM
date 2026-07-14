from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from wfcllm.batch_invariant_v4.calibration import CalibrationArtifact
from wfcllm.batch_invariant_v4.channel import StructuralChannel
from wfcllm.batch_invariant_v4.context import ContextConfig, StructuralContextExtractor
from wfcllm.batch_invariant_v4.detector import CodeEvidence, DetectorPayload, V4Detector
from wfcllm.batch_invariant_v4.keying import V4SecretKey


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RawCandidate:
    task_id: str
    attempt_index: int
    final_code: str
    final_code_sha256: str
    quality_tier: int
    valid: bool
    fallback_count: int = 0


@dataclass(frozen=True)
class CandidateSelection:
    selected: RawCandidate
    selected_generation_evidence: CodeEvidence
    evidence_by_attempt: dict[int, CodeEvidence]
    input_pool_sha256: str
    output_pool_sha256: str
    candidate_pool_match_rate: float


def _pool_sha256(candidates: Iterable[RawCandidate]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.task_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.attempt_index.to_bytes(4, "big"))
        digest.update(bytes.fromhex(candidate.final_code_sha256))
    return digest.hexdigest()


class CandidateRuntime:
    def __init__(self, detector: V4Detector, *, public_config_sha256: str) -> None:
        self.detector = detector
        self.public_config_sha256 = public_config_sha256
        self.selected_final_replay_count = 0
        self.eos_all_candidate_neural_rescore_count = 0

    @classmethod
    def for_test(
        cls,
        *,
        key: V4SecretKey,
        public_config_sha256: str,
        minimum_independent_units: int,
    ) -> CandidateRuntime:
        detector = V4Detector(
            extractor=StructuralContextExtractor(ContextConfig()),
            channel=StructuralChannel(
                key,
                bit_count=32,
                minimum_independent_units=minimum_independent_units,
            ),
            calibration=CalibrationArtifact.from_scores_for_test(
                tuple(index / 10 for index in range(-10, 11)),
                target_fpr=0.05,
                minimum_independent_units=minimum_independent_units,
            ),
        )
        return cls(detector, public_config_sha256=public_config_sha256)

    def _validate(
        self,
        candidates: Iterable[RawCandidate],
        *,
        retry: int,
    ) -> tuple[RawCandidate, ...]:
        items = tuple(candidates)
        if len(items) != retry:
            raise ValueError(f"candidate pool must contain exactly {retry} candidates")
        if not items:
            raise ValueError("candidate pool must not be empty")
        if len({item.task_id for item in items}) != 1:
            raise ValueError("candidate pool must contain one task")
        if tuple(item.attempt_index for item in items) != tuple(range(retry)):
            raise ValueError("candidate attempts must be ordered 0..retry-1")
        for item in items:
            actual = hashlib.sha256(item.final_code.encode("utf-8")).hexdigest()
            if not _SHA256.fullmatch(item.final_code_sha256) or actual != item.final_code_sha256:
                raise ValueError("candidate final-code SHA-256 mismatch")
            if item.fallback_count < 0:
                raise ValueError("candidate fallback_count must be non-negative")
        return items

    def select(
        self,
        candidates: Iterable[RawCandidate],
        *,
        retry: int,
    ) -> CandidateSelection:
        items = self._validate(candidates, retry=retry)
        evidence = {
            item.attempt_index: self.detector.detect(
                DetectorPayload(final_code=item.final_code)
            )
            for item in items
        }
        eligible = [
            item
            for item in items
            if item.valid and evidence[item.attempt_index].eligible
        ]
        valid = [item for item in items if item.valid]
        if eligible:
            selected = max(
                eligible,
                key=lambda item: (
                    item.quality_tier,
                    Fraction(
                        evidence[item.attempt_index].aggregate_numerator,
                        evidence[item.attempt_index].aggregate_denominator,
                    ),
                    evidence[item.attempt_index].independent_units,
                    -sum(evidence[item.attempt_index].erasure_counts.values()),
                    -item.attempt_index,
                ),
            )
        elif valid:
            selected = max(valid, key=lambda item: (item.quality_tier, -item.attempt_index))
        else:
            selected = max(
                items,
                key=lambda item: (
                    item.quality_tier,
                    -item.fallback_count,
                    -item.attempt_index,
                ),
            )
        input_sha = _pool_sha256(items)
        output_items = tuple(items)
        output_sha = _pool_sha256(output_items)
        return CandidateSelection(
            selected=selected,
            selected_generation_evidence=evidence[selected.attempt_index],
            evidence_by_attempt=evidence,
            input_pool_sha256=input_sha,
            output_pool_sha256=output_sha,
            candidate_pool_match_rate=sum(
                left == right for left, right in zip(items, output_items, strict=True)
            )
            / len(items),
        )

    def replay_selected(self, final_code: str) -> CodeEvidence:
        if self.selected_final_replay_count != 0:
            raise ValueError("selected final code must replay exactly once")
        self.selected_final_replay_count += 1
        return self.detector.detect(DetectorPayload(final_code=final_code))
