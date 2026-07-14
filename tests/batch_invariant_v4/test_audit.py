from __future__ import annotations

import hashlib
import json

import pytest

from wfcllm.batch_invariant_v4.audit import (
    compare_evidence_maps,
    evidence_sha256,
    load_retry_ledgers,
    replay_schedule,
    rows_to_candidates,
    scheduled_indices,
)
from wfcllm.batch_invariant_v4.runtime import CandidateRuntime
from wfcllm.batch_invariant_v4.keying import V4SecretKey


def _row(task_id: str, attempt: int) -> dict:
    code = f"def f():\n    value = {attempt}\n    return value\n"
    return {
        "schema_version": "wfcllm-v2-retry-ledger/v2",
        "id": task_id,
        "attempt_index": attempt,
        "final_code": code,
        "final_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "quality": {"quality_tier": 2, "eligible": True},
    }


def test_load_retry_ledgers_and_candidates_preserve_order(tmp_path) -> None:
    path = tmp_path / "retry.jsonl"
    rows = [_row("HumanEval/1", index) for index in range(3)]
    path.write_text("\n".join(json.dumps(row) for row in reversed(rows)) + "\n")

    grouped = load_retry_ledgers((path,))
    candidates = rows_to_candidates(grouped["HumanEval/1"], retry=3)

    assert [candidate.attempt_index for candidate in candidates] == [0, 1, 2]
    assert all(candidate.valid for candidate in candidates)


def test_retry_ledgers_reject_duplicate_attempt(tmp_path) -> None:
    path = tmp_path / "retry.jsonl"
    row = _row("HumanEval/1", 0)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_retry_ledgers((path,))


def test_all_schedule_shapes_orders_and_compositions_are_exact() -> None:
    runtime = CandidateRuntime.for_test(
        key=V4SecretKey.from_material_for_test(b"a" * 32),
        public_config_sha256="b" * 64,
        minimum_independent_units=1,
    )
    rows = tuple(_row("HumanEval/2", index) for index in range(6))
    candidates = rows_to_candidates(rows, retry=6)
    reference = replay_schedule(
        runtime.detector,
        candidates,
        scheduled_indices(candidates, batch_size=1, order="forward"),
    )

    for batch_size in (1, 2, 4, 8, 16, 32):
        for order in ("forward", "reverse", "permutation", "short_first", "long_first"):
            replay = replay_schedule(
                runtime.detector,
                candidates,
                scheduled_indices(candidates, batch_size=batch_size, order=order),
            )
            assert compare_evidence_maps(reference, replay) == ()
            assert {key: evidence_sha256(value) for key, value in replay.items()} == {
                key: evidence_sha256(value) for key, value in reference.items()
            }
