from __future__ import annotations

from wfcllm.generation.retry import evidence_retry_key


class Result:
    accepted_hit_count = 1
    closed_without_hit_count = 2
    fallback_count = 3
    candidate_count = 4


def test_evidence_retry_key_matches_method_contract() -> None:
    assert evidence_retry_key(Result(), attempt_index=5) == (1, -2, -3, 4, -5)
