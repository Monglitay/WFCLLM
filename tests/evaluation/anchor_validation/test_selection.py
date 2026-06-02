from __future__ import annotations

from wfcllm.evaluation.anchor_validation.schema import CandidateBlock
from wfcllm.evaluation.anchor_validation.selection import simulate_retry_selection


def test_retry_selection_accepts_first_hit_within_budget():
    candidates = (
        CandidateBlock("c0", "return x", rank=0),
        CandidateBlock("c1", "return x + 1", rank=1),
        CandidateBlock("c2", "return 1 + x", rank=2),
    )
    signatures = {"c0": (0,), "c1": (1,), "c2": (1,)}

    row = simulate_retry_selection(
        context_id="ctx",
        method="slot_context",
        key_id="key-00",
        gamma=0.5,
        retry_budget=2,
        candidates=candidates,
        signatures_by_candidate_id=signatures,
        valid_set=frozenset({(1,)}),
    )

    assert row.selected_candidate_id == "c1"
    assert row.selected_rank == 1
    assert row.hit_acquired is True
    assert row.fallback is False


def test_retry_selection_falls_back_to_rank_zero_when_no_hit():
    candidates = (CandidateBlock("c0", "return x", rank=0),)

    row = simulate_retry_selection(
        context_id="ctx",
        method="vanilla",
        key_id="key-00",
        gamma=0.5,
        retry_budget=1,
        candidates=candidates,
        signatures_by_candidate_id={"c0": (0,)},
        valid_set=frozenset({(1,)}),
    )

    assert row.selected_candidate_id == "c0"
    assert row.hit_acquired is False
    assert row.fallback is True
    assert row.z_proxy < 0.0
