"""Versioned, keyless serialization for semantic gate inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from wfcllm.windowing.normalization import (
    WINDOW_NORMALIZATION_VERSION,
    normalize_unit_text,
)

GATE_INPUT_CONTRACT_VERSION = "wfcllm-gate-input/v1"
_MAX_WINDOW_UNITS = 3
_NODE_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
_NODE_PATH = r"(?:[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*)?"
_CONTRACT = r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
_PARENT_DESCRIPTOR_RE = re.compile(
    rf"{_CONTRACT}\|{_NODE_PATH}\|"
    r"parent=[A-Za-z_][A-Za-z0-9_]*\|"
    r"ordinal=(?:0|[1-9][0-9]*)\|"
    r"role=[A-Za-z_][A-Za-z0-9_]*\Z",
    re.ASCII,
)


@dataclass(frozen=True)
class GateInput:
    """The complete public context available to the semantic gate.

    ``current_token_count`` must be computed by the gate bundle tokenizer over
    the normalized current window. Generation-model token counts are not part
    of this contract.
    """

    normalization_version: str
    parent_descriptor: str
    depth: int
    previous_units: tuple[str, ...]
    previous_unit_types: tuple[str, ...]
    current_units: tuple[str, ...]
    current_unit_types: tuple[str, ...]
    current_unit_count: int
    current_token_count: int

    def __post_init__(self) -> None:
        if self.normalization_version != WINDOW_NORMALIZATION_VERSION:
            raise ValueError(
                "normalization_version must equal "
                f"{WINDOW_NORMALIZATION_VERSION}"
            )
        if not isinstance(self.parent_descriptor, str) or not _PARENT_DESCRIPTOR_RE.fullmatch(
            self.parent_descriptor
        ):
            raise ValueError(
                "parent_descriptor must use the canonical window descriptor grammar"
            )
        containers = (
            ("previous_units", self.previous_units),
            ("previous_unit_types", self.previous_unit_types),
            ("current_units", self.current_units),
            ("current_unit_types", self.current_unit_types),
        )
        for name, values in containers:
            if not isinstance(values, tuple):
                raise ValueError(f"{name} must be a tuple")
            if any(not isinstance(value, str) for value in values):
                raise ValueError(f"{name} must contain only strings")
        for name, values in (
            ("previous unit types", self.previous_unit_types),
            ("current unit types", self.current_unit_types),
        ):
            if any(_NODE_TYPE_RE.fullmatch(value) is None for value in values):
                raise ValueError(
                    f"{name} must use Tree-sitter node type grammar"
                )
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("depth must be non-negative")
        if type(self.current_unit_count) is not int or self.current_unit_count < 0:
            raise ValueError("current_unit_count must be non-negative")
        if type(self.current_token_count) is not int or self.current_token_count < 0:
            raise ValueError("current_token_count must be non-negative")
        if len(self.previous_units) != len(self.previous_unit_types):
            raise ValueError("previous units and types must have equal lengths")
        if len(self.current_units) != len(self.current_unit_types):
            raise ValueError("current units and types must have equal lengths")
        if self.current_unit_count != len(self.current_units):
            raise ValueError("current_unit_count must equal current_units length")
        if len(self.previous_units) > _MAX_WINDOW_UNITS:
            raise ValueError("previous_units must contain at most 3 units")
        if len(self.current_units) > _MAX_WINDOW_UNITS:
            raise ValueError("current_units must contain at most 3 units")


def serialize_gate_input(gate_input: GateInput) -> str:
    """Serialize only normalized, final-code-recoverable gate features."""

    lines = [
        f"[CONTRACT] {GATE_INPUT_CONTRACT_VERSION}",
        f"[NORMALIZATION] {gate_input.normalization_version}",
        f"[PARENT] {gate_input.parent_descriptor}",
        f"[DEPTH] {gate_input.depth}",
        f"[CURRENT_UNIT_COUNT] {gate_input.current_unit_count}",
        f"[CURRENT_TOKEN_COUNT] {gate_input.current_token_count}",
        "[PREVIOUS]",
    ]
    lines.extend(
        _serialize_unit(unit_type, unit)
        for unit_type, unit in zip(
            gate_input.previous_unit_types,
            gate_input.previous_units,
            strict=True,
        )
    )
    lines.append("[CURRENT]")
    lines.extend(
        _serialize_unit(unit_type, unit)
        for unit_type, unit in zip(
            gate_input.current_unit_types,
            gate_input.current_units,
            strict=True,
        )
    )
    return "\n".join(lines)


def _serialize_unit(unit_type: str, text: str) -> str:
    return f"[S type={unit_type}] {normalize_unit_text(text)}"
