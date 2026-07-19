from __future__ import annotations

from dataclasses import replace
import random

import pytest

from wfcllm.gate.feasibility import (
    FEASIBILITY_CONTRACT_VERSION,
    FeasibilityGroup,
    evaluate_gate_data_feasibility,
)


def _groups(count: int = 100) -> tuple[FeasibilityGroup, ...]:
    return tuple(
        FeasibilityGroup(
            group_id=f"group-{index:05d}",
            suitable_target=index < 15,
            window_lengths=(1, 2, 3),
            statement_family=("assignment", "branch", "loop", "return")[index % 4],
            r1_success_rate=0.20,
            r3_success_rate=0.40,
            holdout_success_rate=0.35,
            split=("validation" if index % 10 == 0 else "test" if index % 10 == 1 else "train"),
            repository_id=f"repo-{index % 40}",
            task_id=f"task-{index % 200}",
            generation_model_id=f"model-{index % 4}",
            structural_invalid_rate=0.1,
            numeric_instability_rate=0.02,
            first_hit_candidate_position=index % 4,
        )
        for index in range(count)
    )


def test_pilot_feasibility_counts_independent_groups_not_derived_rows() -> None:
    summary = evaluate_gate_data_feasibility(_groups(), scale="pilot")
    assert summary.contract_version == FEASIBILITY_CONTRACT_VERSION
    assert summary.independent_group_count == 100
    assert summary.admissions["pilot_group_count"].passed is True
    assert summary.admissions["suitable_positive_groups"].observed == 15
    assert summary.admissions["suitable_negative_groups"].observed == 85
    assert summary.admissions["r3_minus_r1_bootstrap_lower_95"].passed is True
    assert summary.admissions["holdout_key_absolute_gap"].passed is True
    assert summary.passed is True
    assert summary.suitable_groups["positive_rate"] == 0.15
    assert summary.coverage["generation_model_count"] == 4
    assert summary.reliable_success["r3"]["mean"] == pytest.approx(0.4)
    assert summary.rewrite_health["structural_invalid_rate"] == pytest.approx(0.1)
    assert summary.rewrite_health["numeric_instability_rate"] == pytest.approx(0.02)
    assert sum(summary.rewrite_health["first_hit_candidate_position_distribution"].values()) == 100


def test_pilot_rejects_too_few_or_too_many_independent_starts() -> None:
    assert not evaluate_gate_data_feasibility(_groups(99), scale="pilot").passed
    assert not evaluate_gate_data_feasibility(_groups(301), scale="pilot").passed


def test_each_feasibility_admission_has_an_explicit_result() -> None:
    base = list(_groups())
    # Destroy one family, one window length, gain, and holdout transfer.
    bad = tuple(
        replace(
            item,
            window_lengths=(1, 2),
            statement_family="assignment",
            r3_success_rate=item.r1_success_rate,
            holdout_success_rate=0.0,
        )
        for item in base
    )
    summary = evaluate_gate_data_feasibility(bad, scale="pilot")
    assert summary.admissions["window_length_w3_groups"].passed is False
    assert summary.admissions["major_statement_families"].passed is False
    assert summary.admissions["r3_minus_r1_bootstrap_lower_95"].passed is False
    assert summary.admissions["holdout_key_absolute_gap"].passed is False


def test_duplicate_group_ids_cannot_inflate_feasibility() -> None:
    duplicated = _groups(99) + (_groups(1)[0],)
    try:
        evaluate_gate_data_feasibility(duplicated, scale="pilot")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("duplicate group IDs must be rejected")


def test_holdout_improvement_is_not_misclassified_as_a_decline() -> None:
    improved = tuple(replace(item, holdout_success_rate=0.9) for item in _groups())
    summary = evaluate_gate_data_feasibility(improved, scale="pilot")
    assert summary.admissions["holdout_key_absolute_gap"].observed == 0.0
    assert summary.admissions["holdout_key_absolute_gap"].passed is True


def test_feasibility_summary_is_canonical_for_any_group_permutation() -> None:
    groups = tuple(
        replace(
            group,
            r1_success_rate=(index % 19) / 100,
            r3_success_rate=0.25 + (index % 23) / 100,
            holdout_success_rate=0.30 + (index % 13) / 100,
            structural_invalid_rate=(index % 11) / 100,
            numeric_instability_rate=(index % 7) / 100,
        )
        for index, group in enumerate(_groups())
    )
    shuffled = list(groups)
    random.Random(918_273).shuffle(shuffled)

    expected = evaluate_gate_data_feasibility(groups, scale="pilot").to_json()

    assert evaluate_gate_data_feasibility(tuple(reversed(groups)), scale="pilot").to_json() == expected
    assert evaluate_gate_data_feasibility(tuple(shuffled), scale="pilot").to_json() == expected
