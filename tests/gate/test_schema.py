from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
from typing import get_type_hints

import pytest

from wfcllm.gate.config import GateDataConfig, GateTrainConfig, GateValidateConfig
from wfcllm.gate.schema import (
    GATE_DATA_SCHEMA_VERSION,
    CandidateObservation,
    GateLabelRow,
    GateTrainingGroup,
)


def _observation(index: int = 0) -> CandidateObservation:
    return CandidateObservation(
        candidate_index=index,
        code="x += item",
        parse_status="ok",
        unit_count=1,
        same_parent_scope=True,
        boundary_span=(0, 9),
        stable_across_precision_modes=True,
        stable_across_batch_modes=True,
        lsh_by_key_id={
            "train-key-000": {"hit": True, "margin": 0.23, "stable": True}
        },
        generation_seed_id="seed-plan-v1:000",
        rewrite_config_id="rewrite-v1",
        lsh_signature=(1, 0, 1, 0),
        semantic_reference_cosine=1.0,
        semantic_preservation_passed=True,
    )


def _group() -> GateTrainingGroup:
    return GateTrainingGroup(
        schema_version=GATE_DATA_SCHEMA_VERSION,
        group_id="group-001",
        source_id="source-001",
        source_family="source-family",
        repository_group="repository-001",
        function_group="function-001",
        language="python",
        parser_contract_version="python-statement-units/v1",
        split="train",
        window_start_unit_id="unit-001",
        parent_descriptor="parent/v1:module/function_definition/block",
        candidate_window_lengths=(1,),
        previous_units=("x = 0",),
        candidates_by_length={
            "1": tuple(_observation(index) for index in range(4)),
        },
    )


def test_candidate_observation_uses_allowed_fields_only() -> None:
    row = _observation().to_dict()
    serialized = json.dumps(row, sort_keys=True)
    assert "syntax_valid" not in serialized
    assert '"pass"' not in serialized
    assert "correctness" not in serialized
    assert row["lsh_signature"] == [1, 0, 1, 0]


def test_candidate_observation_serializes_one_canonical_lsh_signature() -> None:
    observation = _observation()
    signed = CandidateObservation(
        **{**observation.__dict__, "lsh_signature": (1, 0, 1, 1)}
    )

    row = signed.to_dict()

    assert signed.lsh_signature == (1, 0, 1, 1)
    assert row["lsh_signature"] == [1, 0, 1, 1]
    json.dumps(row, allow_nan=False)


def test_semantic_preservation_failure_is_retained_but_lsh_evidence_free() -> None:
    observation = _observation()
    rejected = CandidateObservation(
        **{
            **observation.__dict__,
            "semantic_reference_cosine": 0.71,
            "semantic_preservation_passed": False,
            "stable_across_precision_modes": False,
            "stable_across_batch_modes": False,
            "lsh_by_key_id": {},
            "lsh_signature": None,
        }
    )
    row = rejected.to_dict()
    assert row["semantic_reference_cosine"] == pytest.approx(0.71)
    assert row["semantic_preservation_passed"] is False
    assert row["lsh_by_key_id"] == {}

    with pytest.raises(ValueError, match="semantic preservation"):
        CandidateObservation(
            **{
                **observation.__dict__,
                "semantic_reference_cosine": 0.71,
                "semantic_preservation_passed": False,
            }
        )


def test_final_structurally_usable_candidate_requires_semantic_probe_result() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="requires passed semantic preservation"):
        CandidateObservation(
            **{
                **observation.__dict__,
                "semantic_reference_cosine": None,
                "semantic_preservation_passed": None,
                "semantic_probe_pending": False,
                "stable_across_precision_modes": False,
                "stable_across_batch_modes": False,
                "lsh_by_key_id": {},
                "lsh_signature": None,
            }
        )


@pytest.mark.parametrize(
    ("cosine", "passed"), [(0.89, True), (0.90, False)]
)
def test_semantic_preservation_decision_must_match_frozen_threshold(
    cosine: float, passed: bool
) -> None:
    observation = _observation()
    values = {
        **observation.__dict__,
        "semantic_reference_cosine": cosine,
        "semantic_preservation_passed": passed,
    }
    if not passed:
        values.update(
            {
                "stable_across_precision_modes": False,
                "stable_across_batch_modes": False,
                "lsh_by_key_id": {},
                "lsh_signature": None,
            }
        )
    with pytest.raises(ValueError, match="cosine >= 0.9"):
        CandidateObservation(**values)


@pytest.mark.parametrize("signature", [(), (1, 2), (True, 0), [1, 0]])
def test_candidate_observation_rejects_invalid_lsh_signature(
    signature: object,
) -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="lsh_signature"):
        CandidateObservation(
            **{**observation.__dict__, "lsh_signature": signature}
        )


@pytest.mark.parametrize(
    "changes",
    [
        {
            "parse_status": "ok",
            "unit_count": 1,
            "same_parent_scope": True,
            "lsh_by_key_id": {},
            "lsh_signature": None,
            "stable_across_precision_modes": False,
            "stable_across_batch_modes": False,
        },
        {
            "parse_status": "parse_error",
            "unit_count": 0,
            "same_parent_scope": False,
            "lsh_by_key_id": {},
            "lsh_signature": (1, 0),
            "stable_across_precision_modes": False,
            "stable_across_batch_modes": False,
        },
        {
            "parse_status": "parse_error",
            "unit_count": 0,
            "same_parent_scope": False,
            "lsh_by_key_id": {},
            "lsh_signature": None,
            "stable_across_precision_modes": True,
            "stable_across_batch_modes": False,
        },
        {
            "lsh_signature": None,
        },
    ],
)
def test_candidate_observation_rejects_evidence_structure_contradictions(
    changes: dict[str, object],
) -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="evidence|signature|stability|semantic"):
        CandidateObservation(**{**observation.__dict__, **changes})


def test_structurally_invalid_empty_observation_is_the_only_empty_form() -> None:
    observation = _observation()
    invalid = CandidateObservation(
        **{
            **observation.__dict__,
            "parse_status": "parse_error",
            "unit_count": 0,
            "same_parent_scope": False,
            "boundary_span": (0, 0),
            "lsh_by_key_id": {},
            "lsh_signature": None,
            "stable_across_precision_modes": False,
            "stable_across_batch_modes": False,
            "semantic_reference_cosine": None,
            "semantic_preservation_passed": None,
        }
    )
    assert invalid.to_dict()["lsh_by_key_id"] == {}


def test_keyed_evidence_cannot_omit_semantic_preservation_fields() -> None:
    observation = _observation()

    with pytest.raises(ValueError, match="requires passed semantic preservation"):
        CandidateObservation(
            **{
                **observation.__dict__,
                "semantic_reference_cosine": None,
                "semantic_preservation_passed": None,
            }
        )


def test_structurally_usable_observation_requires_nonempty_boundary_span() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="boundary_span"):
        CandidateObservation(
            **{**observation.__dict__, "boundary_span": (0, 0)}
        )


@pytest.mark.parametrize("code", ["", "   ", "\t\r\n"])
def test_structurally_usable_observation_requires_nonblank_code(code: str) -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="code"):
        CandidateObservation(**{**observation.__dict__, "code": code})


def test_candidate_observation_deeply_snapshots_lsh_mappings() -> None:
    inner = {"hit": True, "margin": 0.23, "stable": True}
    source = {"train-key-000": inner}
    observation = _observation()
    values = {**observation.__dict__, "lsh_by_key_id": source}
    frozen = CandidateObservation(**values)

    inner["margin"] = 0.99
    source["train-key-001"] = {"hit": False}

    assert frozen.to_dict()["lsh_by_key_id"] == {
        "train-key-000": {"hit": True, "margin": 0.23, "stable": True}
    }
    with pytest.raises(TypeError):
        frozen.lsh_by_key_id["train-key-001"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen.lsh_by_key_id["train-key-000"]["margin"] = 0.99  # type: ignore[index]


def test_records_are_frozen_and_versioned() -> None:
    observation = _observation()
    group = _group()
    label = GateLabelRow(
        schema_version=GATE_DATA_SCHEMA_VERSION,
        group_id=group.group_id,
        window_length=1,
        payload={"close_target": False},
    )

    with pytest.raises(FrozenInstanceError):
        observation.code = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        group.group_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        label.group_id = "changed"  # type: ignore[misc]

    assert observation.to_dict()["schema_version"] == GATE_DATA_SCHEMA_VERSION
    assert group.to_dict()["schema_version"] == GATE_DATA_SCHEMA_VERSION
    assert label.to_dict()["schema_version"] == GATE_DATA_SCHEMA_VERSION


def test_read_only_schema_fields_are_publicly_annotated_as_mappings() -> None:
    assert get_type_hints(CandidateObservation)["lsh_by_key_id"].__origin__ is Mapping
    assert (
        get_type_hints(GateTrainingGroup)["candidates_by_length"].__origin__
        is Mapping
    )
    assert get_type_hints(GateLabelRow)["payload"].__origin__ is Mapping


@pytest.mark.parametrize(
    "parse_status",
    ["ok", "parse_error", "scope_changed", "unit_count_out_of_range"],
)
def test_parse_status_is_a_fixed_parser_fact_enum(parse_status: str) -> None:
    observation = _observation()
    values = {**observation.__dict__, "parse_status": parse_status}
    if parse_status != "ok":
        values.update(
            {
                "same_parent_scope": False,
                "lsh_by_key_id": {},
                "lsh_signature": None,
                "stable_across_precision_modes": False,
                "stable_across_batch_modes": False,
                "semantic_reference_cosine": None,
                "semantic_preservation_passed": None,
            }
        )
    assert CandidateObservation(**values).parse_status == parse_status


def test_unknown_parse_status_is_rejected() -> None:
    observation = _observation()
    values = {**observation.__dict__, "parse_status": "syntax_valid"}
    with pytest.raises(ValueError, match="parse_status"):
        CandidateObservation(**values)


def test_forbidden_quality_proxy_is_rejected_before_serialization() -> None:
    observation = _observation()
    values = {
        **observation.__dict__,
        "lsh_by_key_id": {"train-key-000": {"pass": True}},
    }
    with pytest.raises(ValueError, match="pass"):
        CandidateObservation(**values).to_dict()


def test_group_serializes_metadata_separately_from_candidate_attempts() -> None:
    group = _group()

    group_row = group.to_dict()
    attempt_rows = tuple(group.iter_candidate_attempts())

    assert "candidates_by_length" not in group_row
    assert "code" not in group_row
    assert len(attempt_rows) == 4
    assert {row["group_id"] for row in attempt_rows} == {group.group_id}
    assert {row["window_length"] for row in attempt_rows} == {1}
    assert [row["candidate_index"] for row in attempt_rows] == list(range(4))
    assert all("previous_units" not in row for row in attempt_rows)


def test_group_deeply_snapshots_trajectory_mapping() -> None:
    candidates = tuple(_observation(index) for index in range(4))
    source = {"1": candidates}
    group = _group()
    values = {**group.__dict__, "candidates_by_length": source}
    frozen = GateTrainingGroup(**values)

    source["1"] = tuple(reversed(candidates))
    source["2"] = candidates

    assert tuple(
        row["candidate_index"] for row in frozen.iter_candidate_attempts()
    ) == tuple(range(4))
    with pytest.raises(TypeError):
        frozen.candidates_by_length["2"] = candidates  # type: ignore[index]


def test_group_candidate_and_label_rows_join_only_by_group_id() -> None:
    group = _group()
    group_row = group.to_dict()
    candidate_row = next(group.iter_candidate_attempts())
    label_row = GateLabelRow(
        schema_version=GATE_DATA_SCHEMA_VERSION,
        group_id=group.group_id,
        window_length=1,
        payload={"close_target": False, "suitable_target": True},
    ).to_dict()

    assert {
        group_row["group_id"],
        candidate_row["group_id"],
        label_row["group_id"],
    } == {group.group_id}
    assert "candidates_by_length" not in label_row
    assert "code" not in label_row


def test_label_rows_are_independently_jsonl_serializable() -> None:
    rows = tuple(
        GateLabelRow(
            schema_version=GATE_DATA_SCHEMA_VERSION,
            group_id="group-001",
            window_length=window_length,
            payload={"close_target": window_length == 3},
        ).to_dict()
        for window_length in (1, 2, 3)
    )

    serialized_lines = [json.dumps(row, sort_keys=True) for row in rows]

    assert len(serialized_lines) == 3
    assert all("candidates_by_length" not in line for line in serialized_lines)
    assert [json.loads(line)["window_length"] for line in serialized_lines] == [
        1,
        2,
        3,
    ]


def test_label_row_deeply_snapshots_and_freezes_payload() -> None:
    nested = {"values": [1, {"stable": True}]}
    source = {"metrics": nested}
    row = GateLabelRow(
        schema_version=GATE_DATA_SCHEMA_VERSION,
        group_id="group-001",
        window_length=1,
        payload=source,
    )

    nested["values"].append(2)
    source["new"] = True

    assert row.to_dict()["payload"] == {
        "metrics": {"values": [1, {"stable": True}]}
    }
    with pytest.raises(TypeError):
        row.payload["new"] = True  # type: ignore[index]
    with pytest.raises(AttributeError):
        row.payload["metrics"]["values"].append(2)  # type: ignore[union-attr]


def test_label_row_rejects_nested_quality_proxy_before_serialization() -> None:
    row = GateLabelRow(
        schema_version=GATE_DATA_SCHEMA_VERSION,
        group_id="group-001",
        window_length=1,
        payload={"audit": {"pass": True}},
    )
    with pytest.raises(ValueError, match="pass"):
        row.to_dict()


def test_all_public_rows_are_strict_json_without_non_finite_numbers() -> None:
    group = _group()
    rows = [
        group.to_dict(),
        *group.iter_candidate_attempts(),
        GateLabelRow(
            schema_version=GATE_DATA_SCHEMA_VERSION,
            group_id=group.group_id,
            window_length=1,
            payload={"margin": 0.5},
        ).to_dict(),
    ]

    for row in rows:
        json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wfcllm-gate-data/v0", "schema_version"),
        ("group_id", "", "group_id"),
        ("group_id", 3, "group_id"),
        ("window_length", 0, "window_length"),
        ("window_length", 4, "window_length"),
        ("window_length", True, "window_length"),
        ("payload", [], "payload"),
        ("payload", {3: False}, "payload"),
        ("payload", {"score": object()}, "JSON-compatible"),
        ("payload", {"score": math.nan}, "finite"),
        ("payload", {"score": math.inf}, "finite"),
        (
            "payload",
            {"candidates_by_length": {"1": ["candidate"]}},
            "candidates_by_length",
        ),
    ],
)
def test_label_row_fails_closed_on_invalid_contract_input(
    field: str, value: object, message: str
) -> None:
    values: dict[str, object] = {
        "schema_version": GATE_DATA_SCHEMA_VERSION,
        "group_id": "group-001",
        "window_length": 1,
        "payload": {"close_target": False},
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        GateLabelRow(**values)


def test_group_requires_one_complete_ordered_trajectory_per_window_length() -> None:
    group = _group()
    values = {
        **group.__dict__,
        "candidates_by_length": {"1": (_observation(0), _observation(2))},
    }
    with pytest.raises(ValueError, match="indices 0 through 3"):
        GateTrainingGroup(**values)


def test_group_requires_candidate_keys_to_match_declared_window_lengths() -> None:
    group = _group()
    values = {**group.__dict__, "candidate_window_lengths": (1, 2)}
    with pytest.raises(ValueError, match="candidates_by_length"):
        GateTrainingGroup(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_index", -1, "candidate_index"),
        ("unit_count", -1, "unit_count"),
        ("boundary_span", (9, 0), "boundary_span"),
        ("lsh_by_key_id", {"deployment-key": {}}, "deployment"),
    ],
)
def test_candidate_observation_rejects_invalid_contract_values(
    field: str, value: object, message: str
) -> None:
    observation = _observation()
    values = {**observation.__dict__, field: value}
    with pytest.raises(ValueError, match=message):
        CandidateObservation(**values)


@pytest.mark.parametrize(
    "key_id",
    [
        "raw-key-000",
        "deployment-key-000",
        "train-key-00",
        "train-key-0000",
        "train-key-٠٠٠",
        "TRAIN-key-000",
        "train-key-000-secret",
    ],
)
def test_candidate_observation_rejects_non_opaque_key_id_grammar(
    key_id: str,
) -> None:
    observation = _observation()
    values = {
        **observation.__dict__,
        "lsh_by_key_id": {key_id: {"hit": True, "margin": 0.2}},
    }
    with pytest.raises(ValueError, match="key ID"):
        CandidateObservation(**values)


@pytest.mark.parametrize("key_id", ["train-key-000", "holdout-key-999"])
def test_candidate_observation_accepts_opaque_key_id_grammar(key_id: str) -> None:
    observation = _observation()
    values = {
        **observation.__dict__,
        "lsh_by_key_id": {key_id: {"hit": True, "margin": 0.2}},
    }
    assert tuple(CandidateObservation(**values).lsh_by_key_id) == (key_id,)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_candidate_observation_rejects_non_finite_lsh_numbers(value: float) -> None:
    observation = _observation()
    values = {
        **observation.__dict__,
        "lsh_by_key_id": {"train-key-000": {"margin": value}},
    }
    with pytest.raises(ValueError, match="finite"):
        CandidateObservation(**values)


def test_gate_configs_use_the_frozen_first_version_contract() -> None:
    data = GateDataConfig()
    train = GateTrainConfig()
    validate = GateValidateConfig()

    assert (
        data.training_key_count,
        data.holdout_key_count,
        data.rewrite_count,
        data.rewrite_budgets,
    ) == (32, 8, 3, (1, 3))
    assert train.max_tokens == 256
    assert train.base_model_path == Path("data/models/codet5-base")
    assert (
        validate.decision_agreement_min,
        validate.float_quantized_accepted_set_agreement_min,
        validate.formal_accepted_span_consensus_min,
        validate.suitable_false_positive_rate_max,
        validate.batch_sizes,
    ) == (0.999, 0.999, 1.0, 0.05, (1,))


@pytest.mark.parametrize(
    ("config_type", "field", "value"),
    [
        (GateDataConfig, "training_key_count", 31),
        (GateDataConfig, "training_key_count", 32.0),
        (GateDataConfig, "holdout_key_count", 7),
        (GateDataConfig, "rewrite_count", 6),
        (GateDataConfig, "rewrite_budgets", (1, 2)),
        (GateTrainConfig, "max_tokens", 513),
        (GateTrainConfig, "max_tokens", 512.0),
        (GateValidateConfig, "decision_agreement_min", 0.998),
        (GateValidateConfig, "decision_agreement_min", Decimal("0.999")),
        (
            GateValidateConfig,
            "float_quantized_accepted_set_agreement_min",
            0.998,
        ),
        (GateValidateConfig, "formal_accepted_span_consensus_min", 0.999),
        (GateValidateConfig, "suitable_false_positive_rate_max", 0.051),
        (GateValidateConfig, "batch_sizes", (1, 2, 4)),
    ],
)
def test_gate_configs_reject_contract_drift(
    config_type: type[object], field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        config_type(**{field: value})


@pytest.mark.parametrize(
    "path", [Path("https:/example.invalid/model"), Path("s3:/private/model")]
)
def test_gate_train_config_requires_a_local_model_path(path: Path) -> None:
    with pytest.raises(ValueError, match="local"):
        GateTrainConfig(base_model_path=path)
