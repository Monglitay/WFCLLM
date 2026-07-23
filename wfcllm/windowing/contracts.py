"""Stable data contracts shared by generation and detection windowing."""

from __future__ import annotations

from dataclasses import dataclass
import math

WINDOW_CONTRACT_VERSIONS = {
    "python": "python-statement-window/v1",
    "cpp": "cpp-statement-window/v1",
    "java": "java-statement-window/v1",
    "js": "js-statement-window/v1",
}
WINDOW_CONTRACT_VERSION = WINDOW_CONTRACT_VERSIONS["python"]


def window_contract_for_language(language: str) -> str:
    try:
        return WINDOW_CONTRACT_VERSIONS[language]
    except KeyError as exc:
        raise ValueError(f"unsupported window language: {language!r}") from exc


def is_supported_window_contract(value: str) -> bool:
    return value in WINDOW_CONTRACT_VERSIONS.values()


def language_for_window_contract(value: str) -> str:
    for language, contract in WINDOW_CONTRACT_VERSIONS.items():
        if contract == value:
            return language
    raise ValueError(f"unsupported window contract: {value!r}")


@dataclass(frozen=True)
class GateScores:
    """Quantized gate output and its float-reference stability checks."""

    close_probability: float
    suitable_probability: float
    stable: bool
    precision_delta: float
    decision_agreement: bool = True

    def __post_init__(self) -> None:
        for name in (
            "close_probability",
            "suitable_probability",
            "precision_delta",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if not isinstance(self.stable, bool):
            raise ValueError("stable must be a bool")
        if not isinstance(self.decision_agreement, bool):
            raise ValueError("decision_agreement must be a bool")


@dataclass(frozen=True)
class ParentDescriptor:
    """Final-code-recoverable structural identity for an open window."""

    contract_version: str
    ancestor_node_types: tuple[str, ...]
    direct_parent_type: str
    first_unit_ordinal: int
    compound_header_role: str

    def __post_init__(self) -> None:
        if not self.contract_version:
            raise ValueError("contract_version must not be empty")
        if not isinstance(self.ancestor_node_types, tuple):
            raise ValueError("ancestor_node_types must be a tuple")
        if any(not isinstance(item, str) for item in self.ancestor_node_types):
            raise ValueError("ancestor_node_types must contain only strings")
        if not self.direct_parent_type:
            raise ValueError("direct_parent_type must not be empty")
        if (
            self.ancestor_node_types
            and self.ancestor_node_types[-1] == self.direct_parent_type
        ):
            raise ValueError(
                "direct parent must not be repeated in ancestor path"
            )
        if self.first_unit_ordinal < 0:
            raise ValueError("first_unit_ordinal must be non-negative")
        if not self.compound_header_role:
            raise ValueError("compound_header_role must not be empty")

    @property
    def canonical(self) -> str:
        ancestors = "/".join(self.ancestor_node_types)
        return (
            f"{self.contract_version}|{ancestors}|parent={self.direct_parent_type}|"
            f"ordinal={self.first_unit_ordinal}|role={self.compound_header_role}"
        )


@dataclass(frozen=True)
class StatementUnit:
    """A parser-defined Python statement unit with its structural metadata."""

    unit_id: str
    node_type: str
    text: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    depth: int
    parent_path: tuple[str, ...]
    direct_parent_type: str
    direct_child_ordinal: int
    eligible: bool
    hard_boundary: bool
    compound_header: bool

    def __post_init__(self) -> None:
        if self.start_byte < 0 or self.end_byte <= self.start_byte:
            raise ValueError("statement unit must have a non-empty byte span")
        if self.start_line < 1:
            raise ValueError("start_line must be at least 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.direct_child_ordinal < 0:
            raise ValueError("direct_child_ordinal must be non-negative")
        if not self.parent_path:
            raise ValueError("parent_path must not be empty")
        if self.parent_path[-1] != self.direct_parent_type:
            raise ValueError("parent_path must end with direct_parent_type")
