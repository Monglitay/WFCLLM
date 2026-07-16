"""No-quality-proxy, streaming-friendly gate-data records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator

from wfcllm.method.contracts import reject_quality_proxy_fields

GATE_DATA_SCHEMA_VERSION = "wfcllm-gate-data/v1"
PARSE_STATUSES = frozenset(
    {"ok", "parse_error", "scope_changed", "unit_count_out_of_range"}
)
_CANDIDATES_PER_TRAJECTORY = 7
_MAX_CANDIDATE_CODE_BYTES = 256 * 1024
_OPAQUE_KEY_ID_RE = re.compile(r"(?:train|holdout)-key-[0-9]{3}\Z", re.ASCII)


@dataclass(frozen=True)
class CandidateObservation:
    """Parser/range and semantic-LSH facts for one key-blind candidate."""

    candidate_index: int
    code: str
    parse_status: str
    unit_count: int
    same_parent_scope: bool
    boundary_span: tuple[int, int]
    stable_across_precision_modes: bool
    stable_across_batch_modes: bool
    lsh_by_key_id: Mapping[str, Mapping[str, bool | float]]
    generation_seed_id: str
    rewrite_config_id: str
    lsh_signature: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("candidate_index", self.candidate_index)
        _require_non_negative_int("unit_count", self.unit_count)
        _require_string("code", self.code, allow_empty=True)
        if len(self.code.encode("utf-8")) > _MAX_CANDIDATE_CODE_BYTES:
            raise ValueError("candidate code exceeds 256 KiB")
        if self.parse_status not in PARSE_STATUSES:
            raise ValueError(
                "parse_status must be one of " + ", ".join(sorted(PARSE_STATUSES))
            )
        _require_bool("same_parent_scope", self.same_parent_scope)
        _require_bool(
            "stable_across_precision_modes", self.stable_across_precision_modes
        )
        _require_bool("stable_across_batch_modes", self.stable_across_batch_modes)
        _validate_boundary_span(self.boundary_span)
        _require_string("generation_seed_id", self.generation_seed_id)
        _require_string("rewrite_config_id", self.rewrite_config_id)
        _validate_lsh_signature(self.lsh_signature)
        _validate_lsh_observations(self.lsh_by_key_id)
        structurally_usable = (
            self.parse_status == "ok"
            and self.same_parent_scope
            and self.unit_count in {1, 2, 3}
        )
        if structurally_usable and not self.code.strip():
            raise ValueError(
                "structurally usable candidate requires nonblank code"
            )
        has_evidence = bool(self.lsh_by_key_id)
        if has_evidence and self.lsh_signature is None:
            raise ValueError("non-empty LSH evidence requires lsh_signature")
        if not has_evidence:
            if structurally_usable:
                raise ValueError(
                    "structurally usable candidate requires LSH evidence"
                )
            if self.lsh_signature is not None:
                raise ValueError("empty LSH evidence requires signature=None")
            if (
                self.stable_across_precision_modes
                or self.stable_across_batch_modes
            ):
                raise ValueError("empty LSH evidence requires false stability")
        if structurally_usable and self.boundary_span[0] == self.boundary_span[1]:
            raise ValueError(
                "structurally usable candidate requires non-empty boundary_span"
            )
        frozen_lsh = MappingProxyType(
            {
                key_id: MappingProxyType(dict(observation))
                for key_id, observation in self.lsh_by_key_id.items()
            }
        )
        object.__setattr__(self, "lsh_by_key_id", frozen_lsh)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": GATE_DATA_SCHEMA_VERSION,
            "candidate_index": self.candidate_index,
            "code": self.code,
            "parse_status": self.parse_status,
            "unit_count": self.unit_count,
            "same_parent_scope": self.same_parent_scope,
            "boundary_span": list(self.boundary_span),
            "stable_across_precision_modes": self.stable_across_precision_modes,
            "stable_across_batch_modes": self.stable_across_batch_modes,
            "lsh_by_key_id": {
                key_id: dict(observation)
                for key_id, observation in self.lsh_by_key_id.items()
            },
            "generation_seed_id": self.generation_seed_id,
            "rewrite_config_id": self.rewrite_config_id,
            "lsh_signature": (
                list(self.lsh_signature)
                if self.lsh_signature is not None
                else None
            ),
        }
        reject_quality_proxy_fields(row)
        return row


@dataclass(frozen=True)
class GateTrainingGroup:
    """Metadata plus in-memory candidate trajectories for one window start.

    ``to_dict`` intentionally returns only the window-group record. Candidate
    attempts are emitted one at a time by ``iter_candidate_attempts`` so a
    formal writer can stream them to a separate JSONL artifact.
    """

    schema_version: str
    group_id: str
    source_id: str
    source_family: str
    repository_group: str
    function_group: str
    language: str
    parser_contract_version: str
    split: str
    window_start_unit_id: str
    parent_descriptor: str
    candidate_window_lengths: tuple[int, ...]
    previous_units: tuple[str, ...]
    candidates_by_length: Mapping[str, tuple[CandidateObservation, ...]]

    def __post_init__(self) -> None:
        if self.schema_version != GATE_DATA_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {GATE_DATA_SCHEMA_VERSION!r}"
            )
        for name in (
            "group_id",
            "source_id",
            "source_family",
            "repository_group",
            "function_group",
            "language",
            "parser_contract_version",
            "split",
            "window_start_unit_id",
            "parent_descriptor",
        ):
            _require_string(name, getattr(self, name))
        if not isinstance(self.candidate_window_lengths, tuple):
            raise ValueError("candidate_window_lengths must be a tuple")
        if not self.candidate_window_lengths:
            raise ValueError("candidate_window_lengths must not be empty")
        if (
            any(
                isinstance(length, bool)
                or not isinstance(length, int)
                or length not in {1, 2, 3}
                for length in self.candidate_window_lengths
            )
            or len(set(self.candidate_window_lengths))
            != len(self.candidate_window_lengths)
            or tuple(sorted(self.candidate_window_lengths))
            != self.candidate_window_lengths
        ):
            raise ValueError(
                "candidate_window_lengths must be ordered unique values from 1 to 3"
            )
        if not isinstance(self.previous_units, tuple) or any(
            not isinstance(unit, str) for unit in self.previous_units
        ):
            raise ValueError("previous_units must be a tuple of strings")
        if len(self.previous_units) > 3:
            raise ValueError("previous_units must contain at most 3 units")
        _validate_trajectories(
            self.candidate_window_lengths, self.candidates_by_length
        )
        object.__setattr__(
            self,
            "candidates_by_length",
            MappingProxyType(
                {
                    key: tuple(candidates)
                    for key, candidates in self.candidates_by_length.items()
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "repository_group": self.repository_group,
            "function_group": self.function_group,
            "language": self.language,
            "parser_contract_version": self.parser_contract_version,
            "split": self.split,
            "window_start_unit_id": self.window_start_unit_id,
            "parent_descriptor": self.parent_descriptor,
            "candidate_window_lengths": list(self.candidate_window_lengths),
            "previous_units": list(self.previous_units),
        }
        reject_quality_proxy_fields(row)
        return row

    def iter_candidate_attempts(self) -> Iterator[dict[str, Any]]:
        """Yield flattened candidate-attempt records in deterministic order."""

        for window_length in self.candidate_window_lengths:
            for observation in self.candidates_by_length[str(window_length)]:
                row = {
                    "schema_version": self.schema_version,
                    "group_id": self.group_id,
                    "window_length": window_length,
                    **observation.to_dict(),
                }
                reject_quality_proxy_fields(row)
                yield row


@dataclass(frozen=True)
class GateLabelRow:
    """One independently streamable label record for a group/window pair.

    Task-specific label computation remains in ``wfcllm.gate.labels``. Its
    public label dictionary can be placed in ``payload`` without coupling the
    artifact envelope to that algorithm.
    """

    schema_version: str
    group_id: str
    window_length: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != GATE_DATA_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {GATE_DATA_SCHEMA_VERSION!r}"
            )
        _require_string("group_id", self.group_id)
        if (
            isinstance(self.window_length, bool)
            or not isinstance(self.window_length, int)
            or self.window_length not in {1, 2, 3}
        ):
            raise ValueError("window_length must be one of 1, 2, or 3")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        frozen_payload = _freeze_json_value(self.payload, path="payload")
        if _contains_key(self.payload, "candidates_by_length"):
            raise ValueError("payload must not contain candidates_by_length")
        object.__setattr__(self, "payload", frozen_payload)

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "window_length": self.window_length,
            "payload": _copy_json_value(self.payload, path="payload"),
        }
        reject_quality_proxy_fields(row)
        return row


def _require_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_string(name: str, value: object, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{name} must be {qualifier}")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _validate_boundary_span(span: object) -> None:
    if (
        not isinstance(span, tuple)
        or len(span) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in span)
        or span[0] < 0
        or span[0] > span[1]
    ):
        raise ValueError(
            "boundary_span must be a non-negative ordered integer pair"
        )


def _validate_lsh_observations(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("lsh_by_key_id must be a mapping")
    for key_id, observation in value.items():
        _require_string("lsh key ID", key_id)
        if "deployment" in key_id.casefold():
            raise ValueError("deployment key IDs are forbidden in gate data")
        if _OPAQUE_KEY_ID_RE.fullmatch(key_id) is None:
            raise ValueError(
                "lsh key ID must match train-key-NNN or holdout-key-NNN"
            )
        if not isinstance(observation, Mapping):
            raise ValueError("each lsh_by_key_id value must be a mapping")
        if any(not isinstance(name, str) for name in observation):
            raise ValueError("LSH observation field names must be strings")
        for name, nested_value in observation.items():
            if not isinstance(nested_value, (bool, float)):
                raise ValueError(
                    f"LSH observation {name!r} must be a bool or float"
                )
            if isinstance(nested_value, float) and not math.isfinite(nested_value):
                raise ValueError(f"LSH observation {name!r} must be finite")


def _validate_lsh_signature(value: object) -> None:
    if value is None:
        return
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(bit) is not int or bit not in (0, 1) for bit in value)
    ):
        raise ValueError(
            "lsh_signature must be None or a non-empty tuple of integer bits"
        )


def _validate_trajectories(
    lengths: tuple[int, ...],
    candidates_by_length: object,
) -> None:
    if not isinstance(candidates_by_length, Mapping):
        raise ValueError("candidates_by_length must be a mapping")
    expected_keys = {str(length) for length in lengths}
    if set(candidates_by_length) != expected_keys:
        raise ValueError(
            "candidates_by_length keys must exactly match candidate_window_lengths"
        )
    expected_indices = tuple(range(_CANDIDATES_PER_TRAJECTORY))
    for key in (str(length) for length in lengths):
        candidates = candidates_by_length[key]
        if not isinstance(candidates, tuple) or any(
            not isinstance(candidate, CandidateObservation)
            for candidate in candidates
        ):
            raise ValueError(
                "each candidates_by_length value must be a tuple of observations"
            )
        indices = tuple(candidate.candidate_index for candidate in candidates)
        if indices != expected_indices:
            raise ValueError(
                "each candidate trajectory must contain ordered indices 0 through 6"
            )


def _freeze_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            output[key] = _freeze_json_value(item, path=f"{path}.{key}")
        return MappingProxyType(output)
    raise ValueError(f"{path} must contain only JSON-compatible values")


def _copy_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            output[key] = _copy_json_value(item, path=f"{path}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _copy_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} must contain only JSON-compatible values")


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, Mapping):
        return target in value or any(
            _contains_key(item, target) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_key(item, target) for item in value)
    return False
