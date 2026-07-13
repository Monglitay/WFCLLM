from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from wfcllm.generation.quality_v2 import (
    StaticQualityAssessment,
    assess_static_quality,
)

V2_RETRY_LEDGER_ARTIFACT_TYPE = "wfcllm_v2_retry_attempt"
V2_RETRY_LEDGER_SCHEMA_VERSION = "wfcllm-v2-retry-ledger/v2"


@dataclass(frozen=True)
class RetryAttempt:
    attempt_index: int
    seed: int
    result: Any


@dataclass(frozen=True)
class _ScoredAttempt:
    attempt: RetryAttempt
    quality: StaticQualityAssessment
    code_score: Any


@dataclass(frozen=True)
class V2AttemptSelection:
    result: Any
    attempt_index: int
    generation_score: float
    recovered_score: float
    replay_equal: bool
    no_embedding: bool
    ledger_rows: tuple[dict[str, Any], ...]


class V2RetryAttemptSelector:
    def __init__(self, *, scorer: Any) -> None:
        self._scorer = scorer

    def select(
        self,
        *,
        sample_id: str,
        prompt: str,
        attempts: tuple[RetryAttempt, ...],
    ) -> V2AttemptSelection:
        if not attempts:
            raise ValueError("attempts must not be empty")
        scored = tuple(
            _ScoredAttempt(
                attempt=attempt,
                quality=assess_static_quality(
                    prompt=prompt,
                    final_code=str(attempt.result.final_code),
                ),
                code_score=self._scorer.score_code(str(attempt.result.final_code)),
            )
            for attempt in attempts
        )
        eligible = tuple(item for item in scored if item.quality.eligible)
        no_embedding = not eligible
        if eligible:
            selected = max(eligible, key=_eligible_ranking_key)
        else:
            selected = max(scored, key=_fallback_ranking_key)

        recovered = self._scorer.score_code(str(selected.attempt.result.final_code))
        generation_score = float(selected.code_score.raw_score)
        recovered_score = float(recovered.raw_score)
        replay_equal = generation_score == recovered_score
        ledger_rows = tuple(
            self._ledger_row(
                sample_id=sample_id,
                item=item,
                selected=item.attempt.attempt_index == selected.attempt.attempt_index,
                recovered_score=(
                    recovered_score
                    if item.attempt.attempt_index == selected.attempt.attempt_index
                    else float(item.code_score.raw_score)
                ),
            )
            for item in scored
        )
        return V2AttemptSelection(
            result=selected.attempt.result,
            attempt_index=selected.attempt.attempt_index,
            generation_score=generation_score,
            recovered_score=recovered_score,
            replay_equal=replay_equal,
            no_embedding=no_embedding,
            ledger_rows=ledger_rows,
        )

    @staticmethod
    def _ledger_row(
        *,
        sample_id: str,
        item: _ScoredAttempt,
        selected: bool,
        recovered_score: float,
    ) -> dict[str, Any]:
        result = item.attempt.result
        final_code = str(result.final_code)
        return {
            "artifact_type": V2_RETRY_LEDGER_ARTIFACT_TYPE,
            "schema_version": V2_RETRY_LEDGER_SCHEMA_VERSION,
            "id": sample_id,
            "audit_only": True,
            "detector_input_allowed": False,
            "scientific_claims_enabled": False,
            "attempt_index": item.attempt.attempt_index,
            "seed": item.attempt.seed,
            "selected": selected,
            "generation_score": float(item.code_score.raw_score),
            "recovered_score": float(recovered_score),
            "replay_equal": float(item.code_score.raw_score) == float(recovered_score),
            "unit_count": int(item.code_score.unit_count),
            "duplicate_units": int(item.code_score.duplicate_count),
            "total_signature_bits": int(item.code_score.total_bits),
            "matched_signature_bits": int(item.code_score.matched_bits),
            "quality": asdict(item.quality),
            "v1_accepted_hit_count": int(result.accepted_hit_count),
            "v1_closed_without_hit_count": int(result.closed_without_hit_count),
            "v1_fallback_count": int(result.fallback_count),
            "v1_candidate_count": int(result.candidate_count),
            "final_code_sha256": hashlib.sha256(final_code.encode("utf-8")).hexdigest(),
            "final_code": final_code,
        }


def _eligible_ranking_key(item: _ScoredAttempt) -> tuple[float, int, int, int]:
    return (
        float(item.code_score.raw_score),
        int(item.code_score.unit_count),
        -int(item.attempt.result.fallback_count),
        -int(item.attempt.attempt_index),
    )


def _fallback_ranking_key(item: _ScoredAttempt) -> tuple[int, int, int]:
    return (
        int(item.quality.quality_tier),
        -int(item.attempt.result.fallback_count),
        -int(item.attempt.attempt_index),
    )
