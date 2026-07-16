from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from wfcllm.generation.gated_state import (
    GatedWindowRetryController,
    RetryAction,
    WindowAttempt,
)


def _attempt(
    candidate_index: int,
    *,
    suitable: bool = True,
    structure_ok: bool = True,
    semantic_hit: bool = False,
    semantic_stable: bool = True,
    semantic_margin: float = 0.25,
) -> WindowAttempt:
    return WindowAttempt(
        candidate_index=candidate_index,
        suitable=suitable,
        structure_ok=structure_ok,
        semantic_hit=semantic_hit,
        semantic_stable=semantic_stable,
        semantic_margin=semantic_margin,
    )


def test_original_hit_is_accepted_without_rewrite() -> None:
    controller = GatedWindowRetryController(max_rewrites=3)
    controller.begin(_attempt(0, semantic_hit=True))

    assert controller.next_action() is RetryAction.ACCEPT_ORIGINAL
    assert controller.rewrite_count == 0
    assert controller.selected_candidate_index == 0


def test_unsuitable_original_is_kept_without_rewrite() -> None:
    controller = GatedWindowRetryController(max_rewrites=3)
    controller.begin(_attempt(0, suitable=False))

    assert controller.next_action() is RetryAction.SKIP_ORIGINAL
    assert controller.rewrite_count == 0
    assert controller.selected_candidate_index == 0


def test_suitable_miss_rewrites_whole_window_then_accepts_first_hit() -> None:
    controller = GatedWindowRetryController(max_rewrites=3)
    controller.begin(_attempt(0))
    assert controller.next_action() is RetryAction.REWRITE

    controller.observe(_attempt(1, semantic_hit=False))
    assert controller.next_action() is RetryAction.REWRITE
    controller.observe(_attempt(2, semantic_hit=True))

    assert controller.next_action() is RetryAction.ACCEPT_REWRITE
    assert controller.rewrite_count == 2
    assert controller.selected_candidate_index == 2


def test_budget_exhaustion_restores_candidate_zero() -> None:
    controller = GatedWindowRetryController(max_rewrites=3)
    controller.begin(_attempt(0))
    for index in (1, 2, 3):
        controller.observe(_attempt(index))

    assert controller.next_action() is RetryAction.RESTORE_ORIGINAL
    assert controller.rewrite_count == 3
    assert controller.selected_candidate_index == 0


@pytest.mark.parametrize(
    ("structure_ok", "semantic_stable"),
    [(False, True), (True, False), (False, False)],
)
def test_invalid_rewrite_consumes_budget_but_cannot_be_accepted(
    structure_ok: bool,
    semantic_stable: bool,
) -> None:
    controller = GatedWindowRetryController(max_rewrites=2)
    controller.begin(_attempt(0))
    controller.observe(
        _attempt(
            1,
            structure_ok=structure_ok,
            semantic_hit=True,
            semantic_stable=semantic_stable,
        )
    )

    assert controller.rewrite_count == 1
    assert controller.next_action() is RetryAction.REWRITE
    controller.observe(_attempt(2, semantic_hit=True))
    assert controller.next_action() is RetryAction.ACCEPT_REWRITE
    assert controller.selected_candidate_index == 2


def test_maximum_budget_counts_only_candidates_one_through_r() -> None:
    controller = GatedWindowRetryController(max_rewrites=1)
    controller.begin(_attempt(0))

    assert controller.rewrite_count == 0
    assert controller.next_action() is RetryAction.REWRITE
    controller.observe(_attempt(1))

    assert controller.rewrite_count == 1
    assert controller.next_action() is RetryAction.RESTORE_ORIGINAL


def test_zero_rewrite_budget_restores_original_suitable_miss() -> None:
    controller = GatedWindowRetryController(max_rewrites=0)
    controller.begin(_attempt(0))

    assert controller.next_action() is RetryAction.RESTORE_ORIGINAL
    assert controller.selected_candidate_index == 0


def test_new_window_requires_explicit_reset() -> None:
    controller = GatedWindowRetryController(max_rewrites=1)
    controller.begin(_attempt(0, semantic_hit=True))

    with pytest.raises(RuntimeError, match="reset"):
        controller.begin(_attempt(0))

    controller.reset()
    controller.begin(_attempt(0, suitable=False))
    assert controller.next_action() is RetryAction.SKIP_ORIGINAL


def test_attempt_contract_is_frozen_and_rejects_quality_signals() -> None:
    assert [field.name for field in fields(WindowAttempt)] == [
        "candidate_index",
        "suitable",
        "structure_ok",
        "semantic_hit",
        "semantic_stable",
        "semantic_margin",
    ]
    attempt = _attempt(0)
    with pytest.raises(FrozenInstanceError):
        attempt.semantic_hit = True  # type: ignore[misc]

    for forbidden in (
        "correctness",
        "test_result",
        "pass_result",
        "quality",
        "quality_rank",
        "reference_similarity",
    ):
        kwargs = {
            "candidate_index": 0,
            "suitable": True,
            "structure_ok": True,
            "semantic_hit": False,
            "semantic_stable": True,
            "semantic_margin": 0.0,
            forbidden: True,
        }
        with pytest.raises(TypeError):
            WindowAttempt(**kwargs)  # type: ignore[arg-type]


def test_controller_rejects_nonzero_original_and_out_of_order_rewrites() -> None:
    controller = GatedWindowRetryController(max_rewrites=2)
    with pytest.raises(ValueError, match="candidate_index=0"):
        controller.begin(_attempt(1))

    controller.begin(_attempt(0))
    with pytest.raises(ValueError, match="candidate_index=1"):
        controller.observe(_attempt(2))


def test_observe_is_only_allowed_while_rewrite_is_requested() -> None:
    controller = GatedWindowRetryController(max_rewrites=1)
    with pytest.raises(RuntimeError, match="begin"):
        controller.observe(_attempt(1))

    controller.begin(_attempt(0, semantic_hit=True))
    with pytest.raises(RuntimeError, match="not requested"):
        controller.observe(_attempt(1))


def test_invalid_configuration_and_attempt_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        GatedWindowRetryController(max_rewrites=-1)
    with pytest.raises(ValueError, match="non-negative"):
        _attempt(-1)
    with pytest.raises(ValueError, match="finite"):
        _attempt(0, semantic_margin=float("nan"))
