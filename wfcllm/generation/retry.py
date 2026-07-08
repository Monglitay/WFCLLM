from __future__ import annotations

from typing import Any


def evidence_retry_key(result: Any, attempt_index: int) -> tuple[int, int, int, int, int]:
    return (
        int(result.accepted_hit_count),
        -int(result.closed_without_hit_count),
        -int(result.fallback_count),
        int(result.candidate_count),
        -int(attempt_index),
    )
