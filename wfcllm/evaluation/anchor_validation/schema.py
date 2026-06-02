"""Dataclasses for anchor validation diagnostic artifacts."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, field
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class AnchorMethod(StrEnum):
    VANILLA = "vanilla"
    RANDOM = "random"
    SLOT = "slot"
    CONTEXT = "context"
    SKELETON = "skeleton"
    SLOT_CONTEXT = "slot_context"
    SLOT_CONTEXT_SKELETON = "slot_context_skeleton"
    PROMPT_AWARE = "prompt_aware"
    SEQMARK_ORACLE = "seqmark_oracle"


@dataclass(frozen=True)
class CandidateBlock:
    candidate_id: str
    block_text: str
    rank: int
    syntax_valid: bool = True
    parse_valid: bool = True
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality", _freeze_json_value(self.quality))


@dataclass(frozen=True)
class CandidateContext:
    context_id: str
    dataset: str
    task_id: str
    prompt: str
    function_signature: str
    ast_path: tuple[str, ...]
    node_type: str
    parent_node_type: str
    block_ordinal: int
    context_hash: str
    temperature: float | None
    candidates: tuple[CandidateBlock, ...]
    context_before: str = ""
    context_after: str = ""
    masked_parent_context: str = ""
    import_and_helper_signatures: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionMetricRow:
    context_id: str
    dataset: str
    task_id: str
    method: str
    projection_key_id: str | None
    key_id: str | None
    gamma: float | None
    candidate_count: int
    normalized_entropy: float
    collapse_ratio: float
    effective_region_count: float
    hamming_diversity: float
    node_type: str | None = None
    valid_hit_rate: float | None = None
    gamma_deviation: float | None = None


@dataclass(frozen=True)
class SelectionSimulationRow:
    context_id: str
    method: str
    key_id: str
    gamma: float
    retry_budget: int
    selected_candidate_id: str
    selected_rank: int
    hit_acquired: bool
    fallback: bool
    z_proxy: float
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality", _freeze_json_value(self.quality))


def dataclass_to_jsonable(value: Any) -> dict[str, Any]:
    payload = _to_jsonable(value)
    if "ast_path" in payload:
        payload["ast_path"] = list(payload["ast_path"])
    return payload


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, MappingProxyType):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    return value


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_json_value(item)
            for key, item in value.items()
        })
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value
