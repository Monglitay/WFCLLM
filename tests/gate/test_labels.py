from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import FrozenInstanceError, replace

import pytest

from wfcllm.gate.labels import (
    BudgetOutcome,
    GateLabels,
    LabelThresholds,
    build_gate_labels,
)
from wfcllm.gate.schema import (
    GATE_DATA_SCHEMA_VERSION,
    CandidateObservation,
    GateLabelRow,
)


TRAINING_KEY_COUNT = 32


@pytest.fixture
def observation_factory() -> Callable[..., CandidateObservation]:
    def make(
        *,
        index: int,
        hit_keys: Iterable[int] = (),
        unit_count: int = 1,
        parse_status: str = "ok",
        same_parent_scope: bool = True,
        precision_stable: bool = True,
        batch_stable: bool = True,
        per_key_stable: bool = True,
        margin: float = 0.25,
    ) -> CandidateObservation:
        hits = frozenset(hit_keys)
        structurally_usable = (
            parse_status == "ok"
            and 1 <= unit_count <= 3
            and same_parent_scope
        )
        return CandidateObservation(
            candidate_index=index,
            code=f"value_{index} = source",
            parse_status=parse_status,
            unit_count=unit_count,
            same_parent_scope=same_parent_scope,
            boundary_span=(0, 16),
            stable_across_precision_modes=(
                precision_stable if structurally_usable else False
            ),
            stable_across_batch_modes=(
                batch_stable if structurally_usable else False
            ),
            lsh_by_key_id={
                f"train-key-{key_index:03d}": {
                    "hit": key_index in hits,
                    "margin": margin,
                    "stable": per_key_stable,
                }
                for key_index in range(TRAINING_KEY_COUNT)
            } if structurally_usable else {},
            generation_seed_id=f"seed-v1:{index}",
            rewrite_config_id="rewrite-v1",
            lsh_signature=(1, 0, 1, 0) if structurally_usable else None,
            semantic_reference_cosine=(
                1.0 if index == 0 else 0.95
            ) if structurally_usable else None,
            semantic_preservation_passed=(True if structurally_usable else None),
        )

    return make


def _trajectory(
    observation_factory: Callable[..., CandidateObservation],
    *,
    hits_by_index: Mapping[int, Iterable[int]] | None = None,
    unit_count: int = 1,
) -> tuple[CandidateObservation, ...]:
    hits_by_index = hits_by_index or {}
    return tuple(
        observation_factory(
            index=index,
            hit_keys=hits_by_index.get(index, ()),
            unit_count=unit_count,
        )
        for index in range(4)
    )


def test_r1_and_r3_are_prefixes_of_one_candidate_trajectory(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = tuple(
        observation_factory(
            index=index,
            hit_keys=range(20) if index in {1, 3} else range(4),
        )
        for index in range(4)
    )

    labels = build_gate_labels(
        candidates,
        training_key_count=TRAINING_KEY_COUNT,
        thresholds=LabelThresholds(
            r3_hit_rate=0.60,
            structural_valid_rate=2 / 3,
            unstable_rate=0.10,
        ),
    )

    assert labels.budgets[1].candidate_indices == (0, 1)
    assert labels.budgets[3].candidate_indices == (0, 1, 2, 3)
    assert labels.suitable_target is True


def test_candidate_zero_is_checked_but_not_counted_as_rewrite(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = (
        observation_factory(index=0, hit_keys=range(TRAINING_KEY_COUNT)),
    ) + tuple(
        observation_factory(index=index, hit_keys=()) for index in range(1, 4)
    )

    labels = build_gate_labels(candidates, training_key_count=TRAINING_KEY_COUNT)

    assert labels.budgets[1].rewrite_count == 1
    assert labels.budgets[1].success_rate == 1.0
    assert labels.budgets[1].structural_valid_rate == 1.0
    assert labels.budgets[1].unstable_rate == 0.0


def test_suitable_is_false_at_a_fifty_nine_percent_r3_hit_rate(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = _trajectory(
        observation_factory,
        hits_by_index={1: range(19)},
    )

    labels = build_gate_labels(candidates, training_key_count=TRAINING_KEY_COUNT)

    assert labels.budgets[3].success_rate == 19 / 32
    assert labels.suitable_target is False


def test_suitable_is_false_below_the_structural_rewrite_threshold(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(
        _trajectory(
            observation_factory,
            hits_by_index={0: range(TRAINING_KEY_COUNT)},
        )
    )
    candidates[2] = observation_factory(
        index=2,
        hit_keys=range(TRAINING_KEY_COUNT),
        parse_status="parse_error",
    )
    candidates[3] = observation_factory(
        index=3,
        hit_keys=range(TRAINING_KEY_COUNT),
        same_parent_scope=False,
        parse_status="scope_changed",
    )

    labels = build_gate_labels(tuple(candidates), training_key_count=32)

    assert labels.budgets[3].structural_valid_rate == 1 / 3
    assert labels.suitable_target is False


@pytest.mark.parametrize("stability_field", ["precision", "batch"])
def test_suitable_is_false_above_the_unstable_rewrite_threshold(
    observation_factory: Callable[..., CandidateObservation],
    stability_field: str,
) -> None:
    candidates = list(
        _trajectory(
            observation_factory,
            hits_by_index={0: range(TRAINING_KEY_COUNT)},
        )
    )
    candidates[2] = observation_factory(
        index=2,
        precision_stable=stability_field != "precision",
        batch_stable=stability_field != "batch",
    )

    labels = build_gate_labels(tuple(candidates), training_key_count=32)

    assert labels.budgets[3].unstable_rate == 1 / 3
    assert labels.suitable_target is False


def test_structurally_invalid_rewrite_is_not_classified_as_numerically_unstable(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    candidates[1] = observation_factory(
        index=1,
        parse_status="parse_error",
        precision_stable=False,
        batch_stable=False,
        per_key_stable=False,
    )

    labels = build_gate_labels(tuple(candidates), training_key_count=32)

    # Both rates retain the R=3 denominator. Candidate 0 is in neither metric,
    # and a structurally unavailable rewrite is not numerical evidence.
    assert labels.budgets[3].structural_valid_rate == 2 / 3
    assert labels.budgets[3].unstable_rate == 0.0


def test_structurally_invalid_candidate_may_have_no_lsh_observations(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    candidates[1] = replace(
        candidates[1],
        parse_status="parse_error",
        unit_count=0,
        same_parent_scope=False,
        lsh_by_key_id={},
        lsh_signature=None,
        stable_across_precision_modes=False,
        stable_across_batch_modes=False,
        semantic_reference_cosine=None,
        semantic_preservation_passed=None,
    )

    labels = build_gate_labels(tuple(candidates), training_key_count=32)

    assert labels.budgets[3].structural_valid_rate == 2 / 3
    assert all(
        outcome.first_hit_by_key_id["train-key-000"] is None
        for outcome in labels.budgets.values()
    )


def test_structurally_invalid_nonempty_lsh_map_is_still_strictly_validated(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    with pytest.raises(ValueError, match="structurally invalid"):
        replace(
            candidates[1],
            parse_status="parse_error",
            lsh_by_key_id={"train-key-000": {"hit": True}},
        )


def test_candidate_zero_must_be_structurally_valid_with_complete_evidence(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    candidates[0] = replace(
        candidates[0],
        parse_status="parse_error",
        unit_count=0,
        same_parent_scope=False,
        boundary_span=(0, 0),
        lsh_by_key_id={},
        lsh_signature=None,
        stable_across_precision_modes=False,
        stable_across_batch_modes=False,
        semantic_reference_cosine=None,
        semantic_preservation_passed=None,
    )
    with pytest.raises(ValueError, match="candidate 0"):
        build_gate_labels(tuple(candidates), training_key_count=32)


def test_candidate_zero_blank_code_is_defensively_rejected(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    object.__setattr__(candidates[0], "code", "   \t")
    with pytest.raises(ValueError, match="candidate 0.*code"):
        build_gate_labels(tuple(candidates), training_key_count=32)


def test_labels_reject_pending_semantic_probe_even_with_keyed_evidence(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    candidates[1] = replace(
        candidates[1],
        semantic_probe_pending=True,
        semantic_reference_cosine=None,
        semantic_preservation_passed=None,
    )

    with pytest.raises(ValueError, match="semantic probe must be complete"):
        build_gate_labels(tuple(candidates), training_key_count=32)


def test_reliable_hit_requires_structure_stability_and_configured_margin(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    candidates[0] = observation_factory(index=0, hit_keys=(0,), margin=0.2)
    candidates[1] = observation_factory(
        index=1,
        hit_keys=(1,),
        per_key_stable=False,
        margin=0.5,
    )
    candidates[2] = observation_factory(
        index=2,
        hit_keys=(2,),
        margin=0.199,
    )
    candidates[3] = observation_factory(
        index=3,
        hit_keys=(3,),
        precision_stable=False,
        margin=0.5,
    )

    labels = build_gate_labels(
        tuple(candidates),
        training_key_count=32,
        thresholds=LabelThresholds(configured_margin=0.2),
    )

    first_hits = labels.budgets[3].first_hit_by_key_id
    assert first_hits["train-key-000"] == 0
    assert first_hits["train-key-001"] is None
    assert first_hits["train-key-002"] is None
    assert first_hits["train-key-003"] is None
    assert labels.budgets[3].success_rate == 1 / 32


def test_first_hit_indices_are_budget_prefix_specific(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = _trajectory(
        observation_factory,
        hits_by_index={0: (0,), 2: (1,)},
    )

    labels = build_gate_labels(candidates, training_key_count=32)

    assert labels.budgets[1].first_hit_by_key_id["train-key-000"] == 0
    assert labels.budgets[1].first_hit_by_key_id["train-key-001"] is None
    assert labels.budgets[3].first_hit_by_key_id["train-key-001"] == 2
    assert labels.budgets[3].first_hit_by_key_id["train-key-002"] is None


def test_close_target_is_true_for_a_hard_boundary(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    labels = build_gate_labels(
        _trajectory(observation_factory),
        training_key_count=32,
        hard_boundary=True,
    )

    assert labels.suitable_target is False
    assert labels.close_target is True


def test_close_target_is_true_at_three_units_even_when_unsuitable(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    labels = build_gate_labels(
        _trajectory(observation_factory, unit_count=3),
        training_key_count=32,
    )

    assert labels.suitable_target is False
    assert labels.close_target is True


def test_close_target_is_true_when_the_current_window_is_suitable(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    labels = build_gate_labels(
        _trajectory(
            observation_factory,
            hits_by_index={1: range(20)},
            unit_count=1,
        ),
        training_key_count=32,
    )

    assert labels.suitable_target is True
    assert labels.close_target is True


def test_outcomes_are_deeply_immutable_and_budget_objects_are_independent(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    labels = build_gate_labels(
        _trajectory(observation_factory),
        training_key_count=32,
    )

    assert labels.budgets[1] is not labels.budgets[3]
    with pytest.raises(TypeError):
        labels.budgets[1] = labels.budgets[3]  # type: ignore[index]
    with pytest.raises(TypeError):
        labels.budgets[1].first_hit_by_key_id[  # type: ignore[index]
            "train-key-000"
        ] = 1
    with pytest.raises(FrozenInstanceError):
        labels.close_target = True  # type: ignore[misc]


def test_labels_are_json_safe_and_fit_the_flattened_label_row(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    labels = build_gate_labels(
        _trajectory(observation_factory, hits_by_index={1: range(20)}),
        training_key_count=32,
    )
    row = GateLabelRow(
        schema_version=GATE_DATA_SCHEMA_VERSION,
        group_id="group-001",
        window_length=1,
        payload=labels.to_dict(),
    ).to_dict()

    serialized = json.dumps(row, allow_nan=False, sort_keys=True)
    assert json.loads(serialized)["payload"]["budgets"]["3"]["rewrite_count"] == 3
    for forbidden in ('"test"', '"pass"', "correctness", "quality"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("candidates_transform", "message"),
    [
        (lambda candidates: list(candidates), "tuple"),
        (lambda candidates: candidates[:-1], "0 through 3"),
        (
            lambda candidates: candidates[:2]
            + (replace(candidates[2], candidate_index=4),)
            + candidates[3:],
            "0 through 3",
        ),
        (
            lambda candidates: candidates[:3] + (object(),) + candidates[4:],
            "CandidateObservation",
        ),
    ],
)
def test_build_rejects_malformed_candidate_trajectories(
    observation_factory: Callable[..., CandidateObservation],
    candidates_transform: Callable[[tuple[CandidateObservation, ...]], object],
    message: str,
) -> None:
    candidates = _trajectory(observation_factory)

    with pytest.raises(ValueError, match=message):
        build_gate_labels(
            candidates_transform(candidates),  # type: ignore[arg-type]
            training_key_count=32,
        )


def test_reversed_training_key_mapping_has_identical_canonical_labels(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    original = candidates[2]
    reversed_lsh = dict(reversed(tuple(original.lsh_by_key_id.items())))
    candidates[2] = replace(original, lsh_by_key_id=reversed_lsh)

    canonical = build_gate_labels(
        _trajectory(observation_factory),
        training_key_count=32,
    )
    reversed_result = build_gate_labels(tuple(candidates), training_key_count=32)

    assert reversed_result == canonical
    assert tuple(reversed_result.budgets[3].first_hit_by_key_id) == tuple(
        f"train-key-{index:03d}" for index in range(32)
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong"])
def test_build_rejects_noncanonical_training_key_sets(
    observation_factory: Callable[..., CandidateObservation],
    mutation: str,
) -> None:
    candidates = list(_trajectory(observation_factory))
    original = candidates[2]
    lsh = dict(original.lsh_by_key_id)
    if mutation == "missing":
        lsh.pop("train-key-031")
    elif mutation == "extra":
        lsh["holdout-key-000"] = {
            "hit": False,
            "stable": True,
            "margin": 0.25,
        }
    else:
        lsh["holdout-key-000"] = lsh.pop("train-key-031")
    candidates[2] = replace(original, lsh_by_key_id=lsh)

    with pytest.raises(ValueError, match="training key IDs"):
        build_gate_labels(tuple(candidates), training_key_count=32)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("pass", "quality proxy"),
        ("correctness_result", "quality proxy"),
        ("benign_unknown", "exactly"),
    ],
)
def test_build_rejects_all_extra_per_key_evidence_fields(
    observation_factory: Callable[..., CandidateObservation],
    field: str,
    message: str,
) -> None:
    candidates = list(_trajectory(observation_factory))
    original = candidates[1]
    lsh = {key: dict(evidence) for key, evidence in original.lsh_by_key_id.items()}
    lsh["train-key-000"][field] = True
    candidates[1] = replace(original, lsh_by_key_id=lsh)

    with pytest.raises(ValueError, match=message):
        build_gate_labels(tuple(candidates), training_key_count=32)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hit", 1.0, "hit"),
        ("stable", 1.0, "stable"),
        ("margin", math.nan, "finite"),
        ("margin", math.inf, "finite"),
        ("margin", -0.1, "non-negative"),
    ],
)
def test_build_rejects_invalid_per_key_evidence(
    observation_factory: Callable[..., CandidateObservation],
    field: str,
    value: object,
    message: str,
) -> None:
    candidates = list(_trajectory(observation_factory))
    original = candidates[1]
    lsh = {key: dict(evidence) for key, evidence in original.lsh_by_key_id.items()}
    lsh["train-key-000"][field] = value  # type: ignore[assignment]

    with pytest.raises(ValueError, match=message):
        candidates[1] = replace(original, lsh_by_key_id=lsh)
        build_gate_labels(tuple(candidates), training_key_count=32)


def test_build_rejects_missing_per_key_evidence_field(
    observation_factory: Callable[..., CandidateObservation],
) -> None:
    candidates = list(_trajectory(observation_factory))
    original = candidates[1]
    lsh = {key: dict(evidence) for key, evidence in original.lsh_by_key_id.items()}
    del lsh["train-key-000"]["stable"]
    candidates[1] = replace(original, lsh_by_key_id=lsh)

    with pytest.raises(ValueError, match="stable"):
        build_gate_labels(tuple(candidates), training_key_count=32)


@pytest.mark.parametrize("training_key_count", [True, 31, 33, 32.0])
def test_build_rejects_training_key_count_contract_drift(
    observation_factory: Callable[..., CandidateObservation],
    training_key_count: object,
) -> None:
    with pytest.raises(ValueError, match="training_key_count"):
        build_gate_labels(
            _trajectory(observation_factory),
            training_key_count=training_key_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("hard_boundary", [0, 1, "yes", None])
def test_build_requires_a_boolean_hard_boundary(
    observation_factory: Callable[..., CandidateObservation],
    hard_boundary: object,
) -> None:
    with pytest.raises(ValueError, match="hard_boundary"):
        build_gate_labels(
            _trajectory(observation_factory),
            training_key_count=32,
            hard_boundary=hard_boundary,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("r3_hit_rate", math.nan),
        ("r3_hit_rate", -0.01),
        ("structural_valid_rate", 1.01),
        ("unstable_rate", True),
        ("configured_margin", math.inf),
        ("configured_margin", -0.01),
    ],
)
def test_label_thresholds_reject_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        LabelThresholds(**{field: value})


def _first_hits(
    *,
    values: Mapping[str, int | None] | None = None,
) -> dict[str, int | None]:
    output = {f"train-key-{index:03d}": None for index in range(32)}
    if values:
        output.update(values)
    return output


def _outcome(
    budget: int,
    *,
    first_hits: Mapping[str, int | None] | None = None,
    structural_valid_rate: float = 1.0,
    unstable_rate: float = 0.0,
) -> BudgetOutcome:
    first_hits = _first_hits(values=first_hits)
    return BudgetOutcome(
        budget=budget,
        candidate_indices=tuple(range(budget + 1)),
        rewrite_count=budget,
        success_rate=sum(value is not None for value in first_hits.values()) / 32,
        structural_valid_rate=structural_valid_rate,
        unstable_rate=unstable_rate,
        first_hit_by_key_id=first_hits,
    )


def test_budget_outcome_canonicalizes_and_snapshots_first_hit_mapping() -> None:
    source = dict(reversed(tuple(_first_hits(values={"train-key-000": 1}).items())))
    outcome = _outcome(1, first_hits=source)

    source["train-key-000"] = None

    assert tuple(outcome.first_hit_by_key_id) == tuple(
        f"train-key-{index:03d}" for index in range(32)
    )
    assert outcome.first_hit_by_key_id["train-key-000"] == 1


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong"])
def test_budget_outcome_rejects_noncanonical_first_hit_key_sets(
    mutation: str,
) -> None:
    first_hits = _first_hits()
    if mutation == "missing":
        first_hits.pop("train-key-031")
    elif mutation == "extra":
        first_hits["train-key-032"] = None
    else:
        first_hits["holdout-key-000"] = first_hits.pop("train-key-031")

    with pytest.raises(ValueError, match="training key IDs"):
        BudgetOutcome(
            budget=1,
            candidate_indices=(0, 1),
            rewrite_count=1,
            success_rate=0.0,
            structural_valid_rate=1.0,
            unstable_rate=0.0,
            first_hit_by_key_id=first_hits,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"success_rate": 0.5}, "success_rate"),
        ({"structural_valid_rate": 0.5}, "structural_valid_rate"),
        ({"unstable_rate": 0.5}, "unstable_rate"),
        (
            {"structural_valid_rate": 1 / 3, "unstable_rate": 2 / 3},
            "unstable_rate",
        ),
    ],
)
def test_budget_outcome_rejects_impossible_or_inconsistent_rates(
    updates: Mapping[str, float],
    message: str,
) -> None:
    values: dict[str, object] = {
        "budget": 3,
        "candidate_indices": (0, 1, 2, 3),
        "rewrite_count": 3,
        "success_rate": 0.0,
        "structural_valid_rate": 1.0,
        "unstable_rate": 0.0,
        "first_hit_by_key_id": _first_hits(),
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        BudgetOutcome(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("budget", True),
        ("candidate_indices", (0, True)),
        ("rewrite_count", True),
    ],
)
def test_budget_outcome_rejects_boolean_integer_aliases(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "budget": 1,
        "candidate_indices": (0, 1),
        "rewrite_count": 1,
        "success_rate": 0.0,
        "structural_valid_rate": 1.0,
        "unstable_rate": 0.0,
        "first_hit_by_key_id": _first_hits(),
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        BudgetOutcome(**values)  # type: ignore[arg-type]


def test_gate_labels_canonicalizes_and_snapshots_budget_mapping() -> None:
    source = {3: _outcome(3), 1: _outcome(1)}
    labels = GateLabels(close_target=False, suitable_target=False, budgets=source)

    source[1] = _outcome(1, first_hits={"train-key-000": 1})

    assert tuple(labels.budgets) == (1, 3)
    assert labels.budgets[1].first_hit_by_key_id["train-key-000"] is None


def test_gate_labels_rejects_suitable_without_close() -> None:
    with pytest.raises(ValueError, match="suitable_target"):
        GateLabels(
            close_target=False,
            suitable_target=True,
            budgets={1: _outcome(1), 3: _outcome(3)},
        )


@pytest.mark.parametrize(
    "budgets",
    [
        {1: _outcome(1)},
        {1: _outcome(1), 3: _outcome(3), 7: _outcome(3)},
    ],
)
def test_gate_labels_requires_exact_budget_set(
    budgets: Mapping[int, BudgetOutcome],
) -> None:
    with pytest.raises(ValueError, match="budgets"):
        GateLabels(close_target=False, suitable_target=False, budgets=budgets)


def test_gate_labels_rejects_boolean_budget_key_alias() -> None:
    with pytest.raises(ValueError, match="budgets"):
        GateLabels(
            close_target=False,
            suitable_target=False,
            budgets={True: _outcome(1), 3: _outcome(3)},
        )


@pytest.mark.parametrize(
    "budgets",
    [
        {
            1: _outcome(1, first_hits={"train-key-000": 1}),
            3: _outcome(3, first_hits={"train-key-000": 2}),
        },
        {
            1: _outcome(1),
            3: _outcome(3, first_hits={"train-key-000": 0}),
        },
    ],
)
def test_gate_labels_rejects_cross_budget_first_hit_inconsistency(
    budgets: Mapping[int, BudgetOutcome],
) -> None:
    with pytest.raises(ValueError, match="first-hit"):
        GateLabels(close_target=False, suitable_target=False, budgets=budgets)


def test_gate_labels_rejects_too_many_new_structural_rewrites() -> None:
    with pytest.raises(ValueError, match="structural prefix"):
        GateLabels(
            close_target=False,
            suitable_target=False,
            budgets={
                1: _outcome(1, structural_valid_rate=0.0),
                3: _outcome(3, structural_valid_rate=1.0),
            },
        )


def test_gate_labels_rejects_decreasing_unstable_rewrite_count() -> None:
    with pytest.raises(ValueError, match="unstable prefix"):
        GateLabels(
            close_target=False,
            suitable_target=False,
            budgets={
                1: _outcome(
                    1,
                    structural_valid_rate=1.0,
                    unstable_rate=1.0,
                ),
                3: _outcome(
                    3,
                    structural_valid_rate=1.0,
                    unstable_rate=0.0,
                ),
            },
        )


def test_gate_labels_accepts_legal_structural_and_unstable_prefix_growth() -> None:
    labels = GateLabels(
        close_target=False,
        suitable_target=False,
        budgets={
            1: _outcome(
                1,
                structural_valid_rate=1.0,
                unstable_rate=0.0,
            ),
            3: _outcome(
                3,
                structural_valid_rate=1.0,
                unstable_rate=2 / 3,
            ),
        },
    )

    assert labels.budgets[3].structural_valid_rate == 1.0
    assert labels.budgets[3].unstable_rate == 2 / 3


def test_aggregate_records_reject_huge_integers_as_values() -> None:
    huge = 10**10000
    with pytest.raises(ValueError, match="configured_margin"):
        LabelThresholds(configured_margin=huge)
    with pytest.raises(ValueError, match="success_rate"):
        BudgetOutcome(
            budget=1,
            candidate_indices=(0, 1),
            rewrite_count=1,
            success_rate=huge,
            structural_valid_rate=1.0,
            unstable_rate=0.0,
            first_hit_by_key_id=_first_hits(),
        )


def test_aggregate_to_dict_outputs_remain_strict_json() -> None:
    labels = GateLabels(
        close_target=False,
        suitable_target=False,
        budgets={3: _outcome(3), 1: _outcome(1)},
    )

    serialized = json.dumps(labels.to_dict(), allow_nan=False)

    assert tuple(json.loads(serialized)["budgets"]) == ("1", "3")
