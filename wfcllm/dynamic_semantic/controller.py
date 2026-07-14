from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from wfcllm.dynamic_semantic.channel import UnitEvidence, aggregate_unit_evidence
from wfcllm.dynamic_semantic.context import DynamicContextExtractor
from wfcllm.dynamic_semantic.observer import AttemptContextObserver
from wfcllm.dynamic_semantic.selection import select_dynamic_attempt
from wfcllm.generation.selection_v2 import RetryAttempt


@dataclass(frozen=True)
class ControllerSelection:
    result: Any
    attempt_index: int
    ledger_rows: tuple[dict[str, Any], ...]


class DynamicSemanticController:
    """Own one shared scheduler across the frozen retry-20 attempt stream."""

    def __init__(
        self,
        *,
        extractor: DynamicContextExtractor,
        scheduler: Any,
        minimum_independent_units: int,
    ) -> None:
        self._extractor = extractor
        self._scheduler = scheduler
        self._minimum_independent_units = minimum_independent_units
        self._observers: dict[int, AttemptContextObserver] = {}
        self.selected_evidence: tuple[UnitEvidence, ...] = ()

    def observer_for_attempt(self, attempt_index: int) -> AttemptContextObserver:
        if attempt_index in self._observers:
            raise ValueError("observer already exists for attempt")
        observer = AttemptContextObserver(
            attempt_index=attempt_index,
            extractor=self._extractor,
            scheduler=self._scheduler,
        )
        self._observers[attempt_index] = observer
        return observer

    def attempt_completed(self, attempt_index: int) -> None:
        if attempt_index not in self._observers:
            raise ValueError("attempt has no dynamic semantic observer")
        self._scheduler.attempt_completed(attempt_index)

    def select(
        self,
        *,
        sample_id: str,
        prompt: str,
        attempts: tuple[RetryAttempt, ...],
    ) -> ControllerSelection:
        self._scheduler.flush_final()
        evidence_by_attempt = {
            attempt.attempt_index: self._scheduler.evidence_for_attempt(
                attempt.attempt_index
            )
            for attempt in attempts
        }
        selected = select_dynamic_attempt(
            prompt=prompt,
            attempts=attempts,
            evidence_by_attempt=evidence_by_attempt,
            erasures_by_attempt={attempt.attempt_index: 0 for attempt in attempts},
            minimum_independent_units=self._minimum_independent_units,
        )
        self.selected_evidence = evidence_by_attempt[selected.attempt_index]
        rows = tuple(
            self._ledger_row(
                sample_id=sample_id,
                attempt=attempt,
                evidence=evidence_by_attempt[attempt.attempt_index],
                selected=attempt.attempt_index == selected.attempt_index,
            )
            for attempt in attempts
        )
        return ControllerSelection(
            result=selected.result,
            attempt_index=selected.attempt_index,
            ledger_rows=rows,
        )

    def _ledger_row(
        self,
        *,
        sample_id: str,
        attempt: RetryAttempt,
        evidence: tuple[UnitEvidence, ...],
        selected: bool,
    ) -> dict[str, Any]:
        aggregate = aggregate_unit_evidence(
            evidence,
            minimum_independent_units=self._minimum_independent_units,
        )
        final_code = str(attempt.result.final_code)
        return {
            "artifact_type": "wfcllm_dynamic_semantic_v3_attempt",
            "schema_version": "wfcllm-dynamic-semantic-attempt/v3",
            "id": sample_id,
            "audit_only": True,
            "detector_input_allowed": False,
            "scientific_claims_enabled": False,
            "attempt_index": attempt.attempt_index,
            "seed": attempt.seed,
            "selected": selected,
            "statistic_numerator": aggregate.numerator,
            "statistic_denominator": aggregate.denominator,
            "score": aggregate.score,
            "independent_units": aggregate.independent_units,
            "eligible": aggregate.eligible,
            "context_sha256": [item.context_sha256 for item in evidence],
            "unit_ids": [item.unit_id for item in evidence],
            "v1_accepted_hit_count": int(attempt.result.accepted_hit_count),
            "v1_closed_without_hit_count": int(
                attempt.result.closed_without_hit_count
            ),
            "v1_fallback_count": int(attempt.result.fallback_count),
            "v1_candidate_count": int(attempt.result.candidate_count),
            "final_code_sha256": hashlib.sha256(
                final_code.encode("utf-8")
            ).hexdigest(),
            "final_code": final_code,
        }
