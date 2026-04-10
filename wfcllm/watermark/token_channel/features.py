"""Structural features for token-channel scoring."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

FEATURE_VERSION = "token-channel-features/v1"
PYTHON_LANGUAGE = "python"


@dataclass(frozen=True)
class TokenChannelFeatures:
    """Stable structural features passed into the token-channel model."""

    node_type: str
    parent_node_type: str
    block_relative_offset: int
    in_code_body: bool
    structure_mask: bool
    language: str = PYTHON_LANGUAGE

    def __post_init__(self) -> None:
        _coerce_string(self.node_type, "node_type")
        _coerce_string(self.parent_node_type, "parent_node_type")
        _coerce_int(self.block_relative_offset, "block_relative_offset")
        _coerce_bool(self.in_code_body, "in_code_body")
        _coerce_bool(self.structure_mask, "structure_mask")
        _coerce_string(self.language, "language")
        if not self.node_type:
            raise ValueError("node_type must be a non-empty string")
        if not self.parent_node_type:
            raise ValueError("parent_node_type must be a non-empty string")
        if self.block_relative_offset < 0:
            raise ValueError("block_relative_offset must be >= 0")
        if self.language != PYTHON_LANGUAGE:
            raise ValueError("language must be 'python'")

    def to_dict(self) -> dict[str, object]:
        return {
            "node_type": self.node_type,
            "parent_node_type": self.parent_node_type,
            "block_relative_offset": self.block_relative_offset,
            "in_code_body": self.in_code_body,
            "structure_mask": self.structure_mask,
            "language": self.language,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TokenChannelFeatures:
        if not isinstance(payload, Mapping):
            raise ValueError("TokenChannelFeatures payload must be a mapping")
        required_keys = {
            "node_type",
            "parent_node_type",
            "block_relative_offset",
            "in_code_body",
            "structure_mask",
        }
        missing_keys = sorted(required_keys - payload.keys())
        if missing_keys:
            raise ValueError(
                "Missing required TokenChannelFeatures keys: " + ", ".join(missing_keys)
            )
        return cls(
            node_type=_coerce_string(payload["node_type"], "node_type"),
            parent_node_type=_coerce_string(payload["parent_node_type"], "parent_node_type"),
            block_relative_offset=_coerce_int(payload["block_relative_offset"], "block_relative_offset"),
            in_code_body=_coerce_bool(payload["in_code_body"], "in_code_body"),
            structure_mask=_coerce_bool(payload["structure_mask"], "structure_mask"),
            language=_coerce_string(payload.get("language", PYTHON_LANGUAGE), "language"),
        )


def _coerce_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _coerce_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _coerce_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value
