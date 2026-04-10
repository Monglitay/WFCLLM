"""Structural features for token-channel scoring."""

from __future__ import annotations

from dataclasses import dataclass
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
        return cls(
            node_type=str(payload["node_type"]),
            parent_node_type=str(payload["parent_node_type"]),
            block_relative_offset=int(payload["block_relative_offset"]),
            in_code_body=bool(payload["in_code_body"]),
            structure_mask=bool(payload["structure_mask"]),
            language=str(payload.get("language", PYTHON_LANGUAGE)),
        )
