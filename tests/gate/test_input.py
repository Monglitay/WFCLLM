from __future__ import annotations

from dataclasses import replace

import pytest

from wfcllm.gate import (
    GATE_INPUT_CONTRACT_VERSION,
    GateInput as ExportedGateInput,
    serialize_gate_input as exported_serialize_gate_input,
)
from wfcllm.gate.input import GateInput, serialize_gate_input
from wfcllm.windowing.normalization import WINDOW_NORMALIZATION_VERSION


def _valid_gate_input() -> GateInput:
    return GateInput(
        normalization_version="wfcllm-window-normalization/v1",
        parent_descriptor="v1|function|parent=block|ordinal=0|role=body",
        depth=1,
        previous_units=("x = 1",),
        previous_unit_types=("assignment",),
        current_units=("return x",),
        current_unit_types=("return_statement",),
        current_unit_count=1,
        current_token_count=2,
    )


def test_gate_input_contains_no_key_or_target_region() -> None:
    payload = serialize_gate_input(_valid_gate_input())

    assert payload == (
        "[CONTRACT] wfcllm-gate-input/v1\n"
        "[NORMALIZATION] wfcllm-window-normalization/v1\n"
        "[PARENT] v1|function|parent=block|ordinal=0|role=body\n"
        "[DEPTH] 1\n"
        "[CURRENT_UNIT_COUNT] 1\n"
        "[CURRENT_TOKEN_COUNT] 2\n"
        "[PREVIOUS]\n"
        "[S type=assignment] x = 1\n"
        "[CURRENT]\n"
        "[S type=return_statement] return x"
    )
    assert "secret" not in payload.lower()
    assert "lsh" not in payload.lower()
    assert "retry" not in payload.lower()


def test_gate_serializer_normalizes_every_unit_deterministically() -> None:
    gate_input = replace(
        _valid_gate_input(),
        previous_units=('name = "你好  世界"  \r\n',),
        current_units=("return name\t\r\n",),
    )

    first = serialize_gate_input(gate_input)
    second = serialize_gate_input(gate_input)

    assert first == second
    assert '[S type=assignment] name = "你好  世界"' in first
    assert "[S type=return_statement] return name" in first
    assert "\r" not in first
    assert " \n" not in first


def test_gate_root_exports_public_input_contract() -> None:
    assert GATE_INPUT_CONTRACT_VERSION == "wfcllm-gate-input/v1"
    assert ExportedGateInput is GateInput
    assert exported_serialize_gate_input is serialize_gate_input


def test_gate_input_rejects_unsupported_normalization_version() -> None:
    with pytest.raises(
        ValueError,
        match="normalization_version must equal wfcllm-window-normalization/v1",
    ):
        replace(_valid_gate_input(), normalization_version="future/v2")


@pytest.mark.parametrize(
    "field",
    [
        "previous_units",
        "previous_unit_types",
        "current_units",
        "current_unit_types",
    ],
)
def test_gate_input_requires_tuple_unit_containers(field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a tuple"):
        replace(_valid_gate_input(), **{field: ["value"]})


@pytest.mark.parametrize(
    "field",
    [
        "previous_units",
        "previous_unit_types",
        "current_units",
        "current_unit_types",
    ],
)
def test_gate_input_requires_string_container_elements(field: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field} must contain only strings",
    ):
        replace(_valid_gate_input(), **{field: (1,)})


def test_valid_gate_input_uses_current_normalization_version() -> None:
    assert _valid_gate_input().normalization_version == WINDOW_NORMALIZATION_VERSION


@pytest.mark.parametrize(
    "parent_descriptor",
    [
        "module",
        "v1|function|parent=block|ordinal=0|role=body|budget=6",
        "v1|function|parent=block|ordinal=0|role=model=hidden",
        "v1|function|parent=block|ordinal=-1|role=body",
        "v1|function|parent=block|ordinal=0|role=body [MODEL_ID] hidden",
        "v1|function|parent=block|ordinal=0|role=body\n[BUDGET] 6",
    ],
)
def test_gate_input_rejects_noncanonical_parent_descriptor(
    parent_descriptor: str,
) -> None:
    with pytest.raises(ValueError, match="parent_descriptor"):
        replace(_valid_gate_input(), parent_descriptor=parent_descriptor)


@pytest.mark.parametrize(
    "parent_descriptor",
    [
        "v1|function|parent=block|ordinal=0|role=body",
        (
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=12|role=header"
        ),
        "python-statement-window/v1||parent=module|ordinal=0|role=body",
    ],
)
def test_gate_input_accepts_plan_and_production_parent_descriptors(
    parent_descriptor: str,
) -> None:
    assert (
        replace(_valid_gate_input(), parent_descriptor=parent_descriptor)
        .parent_descriptor
        == parent_descriptor
    )


@pytest.mark.parametrize(
    "unit_type",
    [
        "assignment budget=6",
        "assignment][MODEL_ID=hidden",
        "return-statement",
        "[SOURCE_FAMILY]",
        "",
    ],
)
def test_gate_input_rejects_noncanonical_unit_type(unit_type: str) -> None:
    with pytest.raises(ValueError, match="unit types"):
        replace(_valid_gate_input(), current_unit_types=(unit_type,))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"depth": -1}, "depth must be non-negative"),
        ({"current_unit_count": -1}, "current_unit_count must be non-negative"),
        ({"current_token_count": -1}, "current_token_count must be non-negative"),
        (
            {"previous_unit_types": ()},
            "previous units and types must have equal lengths",
        ),
        (
            {"current_unit_types": ()},
            "current units and types must have equal lengths",
        ),
        (
            {"current_unit_count": 0},
            "current_unit_count must equal current_units length",
        ),
        (
            {
                "previous_units": ("a", "b", "c", "d"),
                "previous_unit_types": ("a", "b", "c", "d"),
            },
            "previous_units must contain at most 3 units",
        ),
        (
            {
                "current_units": ("a", "b", "c", "d"),
                "current_unit_types": ("a", "b", "c", "d"),
                "current_unit_count": 4,
            },
            "current_units must contain at most 3 units",
        ),
    ],
)
def test_gate_input_rejects_structural_contradictions(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_valid_gate_input(), **changes)
