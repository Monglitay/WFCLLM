"""Key-blind, deterministic candidate trajectories for semantic-gate data."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol

from wfcllm.gate.schema import (
    GATE_DATA_SCHEMA_VERSION,
    PARSE_STATUSES,
    CandidateObservation,
    GateTrainingGroup,
)
from wfcllm.gate.sources import (
    canonical_gate_source_identity,
    validate_gate_source_family,
    validate_gate_source_identity,
)
from wfcllm.method.contracts import (
    FORBIDDEN_QUALITY_GATE_FIELDS,
    reject_quality_proxy_fields,
)
from wfcllm.windowing.contracts import (
    WINDOW_CONTRACT_VERSION,
    ParentDescriptor,
    StatementUnit,
)

_REWRITE_INDICES = tuple(range(1, 7))
_CANONICAL_TRAINING_KEY_IDS = tuple(
    f"train-key-{index:03d}" for index in range(32)
)
_CANONICAL_HOLDOUT_KEY_IDS = tuple(
    f"holdout-key-{index:03d}" for index in range(8)
)
_KEY_ID_RE = re.compile(r"(?:train|holdout)-key-[0-9]{3}\Z", re.ASCII)
_LEAK_FIELD_PARTS = (
    "secret",
    "trainingkey",
    "deploymentkey",
    "lsh",
    "targetregion",
    "firsthit",
)
_EXACT_SECRET_FIELD_NAMES = frozenset(
    {
        "key",
        "keys",
        "keyid",
        "apikey",
        "secretkey",
        "keymaterial",
        "privatekey",
        "accesskey",
        "signingkey",
        "encryptionkey",
        "rawkey",
    }
)
_NORMALIZED_QUALITY_FIELDS = frozenset(
    "".join(character for character in unicodedata.normalize("NFKC", name).casefold()
            if character.isalnum())
    for name in FORBIDDEN_QUALITY_GATE_FIELDS
)


class WindowRewriter(Protocol):
    """A key-blind whole-window rewrite boundary."""

    def rewrite(
        self,
        request: RewriteRequest,
        *,
        candidate_index: int,
    ) -> RewriteCandidate: ...


class MultiKeyLshProbe(Protocol):
    """Probe public key IDs without exposing key material to the rewriter."""

    def probe(
        self,
        *,
        window_text: str,
        parent_descriptor: str,
        key_ids: tuple[str, ...],
    ) -> dict[str, LshProbeResult]: ...


@dataclass(frozen=True)
class StructuralBoundary:
    """Public parser facts constraining one whole-window rewrite."""

    start_byte: int
    end_byte: int
    depth: int
    direct_parent_type: str
    unit_ids: tuple[str, ...]
    compound_singleton: bool
    hard_boundary_after: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_byte, bool)
            or not isinstance(self.start_byte, int)
            or self.start_byte < 0
            or isinstance(self.end_byte, bool)
            or not isinstance(self.end_byte, int)
            or self.end_byte <= self.start_byte
        ):
            raise ValueError("structural boundary byte span must be non-empty")
        if (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or self.depth < 0
        ):
            raise ValueError("structural boundary depth must be non-negative")
        _require_nonempty_string("direct_parent_type", self.direct_parent_type)
        if (
            not isinstance(self.unit_ids, tuple)
            or not self.unit_ids
            or len(self.unit_ids) > 3
            or any(
                not isinstance(unit_id, str) or not unit_id
                for unit_id in self.unit_ids
            )
        ):
            raise ValueError("unit_ids must contain one to three non-empty strings")
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("unit_ids must be unique")
        _require_bool("compound_singleton", self.compound_singleton)
        _require_bool("hard_boundary_after", self.hard_boundary_after)
        if self.compound_singleton and len(self.unit_ids) != 1:
            raise ValueError("compound_singleton requires exactly one unit")

    def to_dict(self) -> dict[str, object]:
        return {
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "depth": self.depth,
            "direct_parent_type": self.direct_parent_type,
            "unit_ids": list(self.unit_ids),
            "compound_singleton": self.compound_singleton,
            "hard_boundary_after": self.hard_boundary_after,
        }


@dataclass(frozen=True)
class RewriteRequest:
    """The complete allowlist of information visible to the rewriter."""

    prompt: str
    completed_prefix: str
    original_window: str
    canonical_parent: str
    window_start_unit_id: str
    window_length: int
    structural_boundary: StructuralBoundary

    def __post_init__(self) -> None:
        _require_string("prompt", self.prompt)
        _require_string("completed_prefix", self.completed_prefix)
        _require_nonempty_string("original_window", self.original_window)
        _require_nonempty_string("canonical_parent", self.canonical_parent)
        _require_nonempty_string("window_start_unit_id", self.window_start_unit_id)
        if (
            isinstance(self.window_length, bool)
            or not isinstance(self.window_length, int)
            or self.window_length not in {1, 2, 3}
        ):
            raise ValueError("window_length must be 1, 2, or 3")
        if not isinstance(self.structural_boundary, StructuralBoundary):
            raise ValueError("structural_boundary must be a StructuralBoundary")
        if len(self.structural_boundary.unit_ids) != self.window_length:
            raise ValueError("window_length must match structural boundary units")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "prompt": self.prompt,
            "completed_prefix": self.completed_prefix,
            "original_window": self.original_window,
            "canonical_parent": self.canonical_parent,
            "window_start_unit_id": self.window_start_unit_id,
            "window_length": self.window_length,
            "structural_boundary": self.structural_boundary.to_dict(),
        }
        validate_key_blind_payload(payload)
        return payload


@dataclass(frozen=True)
class RewriteCandidate:
    """A rewriter result with parser facts computed outside the LSH probe."""

    code: str
    parse_status: str
    unit_count: int
    same_parent_scope: bool
    boundary_span: tuple[int, int]
    generation_seed_id: str
    rewrite_config_id: str

    def __post_init__(self) -> None:
        _require_string("code", self.code)
        if self.parse_status not in PARSE_STATUSES:
            raise ValueError("parse_status is not part of the gate-data contract")
        if self.parse_status == "ok" and not self.code:
            raise ValueError("code must not be empty when parse_status is ok")
        if (
            isinstance(self.unit_count, bool)
            or not isinstance(self.unit_count, int)
            or self.unit_count < 0
        ):
            raise ValueError("unit_count must be a non-negative integer")
        _require_bool("same_parent_scope", self.same_parent_scope)
        _validate_span(self.boundary_span)
        structurally_usable = (
            self.parse_status == "ok"
            and self.same_parent_scope
            and self.unit_count in {1, 2, 3}
        )
        if structurally_usable and not self.code.strip():
            raise ValueError(
                "structurally usable rewrite requires nonblank code"
            )
        if structurally_usable and self.boundary_span[0] == self.boundary_span[1]:
            raise ValueError(
                "structurally usable rewrite requires non-empty boundary_span"
            )
        _require_nonempty_string("generation_seed_id", self.generation_seed_id)
        _require_nonempty_string("rewrite_config_id", self.rewrite_config_id)


@dataclass(frozen=True)
class LshProbeResult:
    """Per-key evidence across approved precision and batch modes."""

    signature: tuple[int, ...]
    margin: float
    hit: bool
    stable: bool
    stable_across_precision_modes: bool
    stable_across_batch_modes: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.signature, tuple)
            or not self.signature
            or any(type(bit) is not int or bit not in (0, 1) for bit in self.signature)
        ):
            raise ValueError("signature must be a non-empty tuple of integer bits")
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or not math.isfinite(self.margin)
            or self.margin < 0
        ):
            raise ValueError("margin must be a finite non-negative number")
        _require_bool("hit", self.hit)
        _require_bool("stable", self.stable)
        _require_bool(
            "stable_across_precision_modes", self.stable_across_precision_modes
        )
        _require_bool("stable_across_batch_modes", self.stable_across_batch_modes)

    def is_reliable_hit(self, *, configured_margin: float) -> bool:
        if (
            isinstance(configured_margin, bool)
            or not isinstance(configured_margin, (int, float))
            or not math.isfinite(configured_margin)
            or configured_margin < 0
        ):
            raise ValueError(
                "configured_margin must be finite and non-negative"
            )
        return (
            self.hit
            and self.stable
            and self.stable_across_precision_modes
            and self.stable_across_batch_modes
            and self.margin >= configured_margin
        )

@dataclass(frozen=True)
class GateBuildContext:
    """Source grouping metadata kept separate from serialized gate inputs."""

    prompt: str = ""
    source_id: str = "source-unspecified"
    source_family: str = "main_generation"
    repository_id: str | None = None
    task_id: str | None = None
    function_id: str | None = "function-unspecified"
    language: str = "python"
    parser_contract_version: str = WINDOW_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_string("prompt", self.prompt)
        for name in (
            "source_id",
            "source_family",
            "language",
            "parser_contract_version",
        ):
            _require_nonempty_string(name, getattr(self, name))
        validate_gate_source_family(self.source_family)
        validate_gate_source_identity("source_id", self.source_id)
        for name in ("repository_id", "task_id", "function_id"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty_string(name, value)
                validate_gate_source_identity(name, value)
        if not any((self.repository_id, self.task_id, self.function_id)):
            raise ValueError("one repository, task, or function ID is required")


@dataclass(frozen=True)
class GateDataVariant:
    """A context/budget view over one shared immutable trajectory."""

    group: GateDataGroup
    context_length: int
    rewrite_budget: int
    previous_units: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.group, GateDataGroup):
            raise ValueError("group must be a GateDataGroup")
        if type(self.context_length) is not int or self.context_length not in {1, 2, 3}:
            raise ValueError("context_length must be 1, 2, or 3")
        if type(self.rewrite_budget) is not int or self.rewrite_budget not in {1, 3, 6}:
            raise ValueError("rewrite_budget must be 1, 3, or 6")
        if (
            not isinstance(self.previous_units, tuple)
            or len(self.previous_units) > self.context_length
            or any(not isinstance(unit, str) for unit in self.previous_units)
        ):
            raise ValueError(
                "previous_units must be a bounded tuple of strings"
            )

    @property
    def item_id(self) -> str:
        return (
            f"{self.group.group_id}:context={self.context_length}:"
            f"budget={self.rewrite_budget}"
        )

    @property
    def split_group_id(self) -> str:
        return self.group.split_group_id

    @property
    def repository_id(self) -> str | None:
        return self.group.repository_id

    @property
    def task_id(self) -> str | None:
        return self.group.task_id

    @property
    def function_id(self) -> str | None:
        return self.group.function_id

    @property
    def candidates_by_length(
        self,
    ) -> Mapping[str, tuple[CandidateObservation, ...]]:
        return self.group.candidates_by_length


@dataclass(frozen=True)
class GateDataGroup:
    """One window start and all W1/W2/W3 candidate trajectories."""

    training_group: GateTrainingGroup
    repository_id: str | None
    task_id: str | None
    function_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.training_group, GateTrainingGroup):
            raise ValueError("training_group must be a GateTrainingGroup")
        for name in ("repository_id", "task_id", "function_id"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty_string(name, value)
        if not any((self.repository_id, self.task_id, self.function_id)):
            raise ValueError("group requires a repository, task, or function ID")
        expected_repository_group = (
            self.repository_id or self.task_id or self.function_id
        )
        assert expected_repository_group is not None
        expected_repository_group = canonical_gate_source_identity(
            expected_repository_group
        )
        expected_function_group = canonical_gate_source_identity(
            self.function_id
            if self.function_id is not None
            else expected_repository_group
        )
        if self.training_group.repository_group != expected_repository_group:
            raise ValueError(
                "training_group.repository_group contradicts wrapper identity"
            )
        if self.training_group.function_group != expected_function_group:
            raise ValueError(
                "training_group.function_group contradicts wrapper identity"
            )

    @property
    def group_id(self) -> str:
        return self.training_group.group_id

    @property
    def item_id(self) -> str:
        return self.group_id

    @property
    def window_start_unit_id(self) -> str:
        return self.training_group.window_start_unit_id

    @property
    def candidate_window_lengths(self) -> tuple[int, ...]:
        return self.training_group.candidate_window_lengths

    @property
    def candidates_by_length(
        self,
    ) -> Mapping[str, tuple[CandidateObservation, ...]]:
        return self.training_group.candidates_by_length

    @property
    def split_group_id(self) -> str:
        if self.repository_id is not None:
            return (
                "repository:"
                + canonical_gate_source_identity(self.repository_id)
            )
        if self.task_id is not None:
            return "task:" + canonical_gate_source_identity(self.task_id)
        if self.function_id is not None:
            return (
                "function:" + canonical_gate_source_identity(self.function_id)
            )
        raise ValueError("group has no split identity")

    def expand_contexts_and_budgets(self) -> tuple[GateDataVariant, ...]:
        """Create views without copying or regenerating candidate trajectories."""

        return tuple(
            GateDataVariant(
                group=self,
                context_length=context_length,
                rewrite_budget=budget,
                previous_units=self.training_group.previous_units[-context_length:],
            )
            for context_length in (1, 2, 3)
            for budget in (1, 3, 6)
        )

    def to_training_group(self, *, split: str) -> GateTrainingGroup:
        _require_nonempty_string("split", split)
        return replace(self.training_group, split=split)


class GateDataBuilder:
    """Build each legal start exactly once, in W1/W2/W3 order."""

    def __init__(
        self,
        *,
        rewriter: WindowRewriter,
        lsh_probe: MultiKeyLshProbe,
        key_ids: tuple[str, ...] | None = None,
        key_mode: str = "training",
        test_only_allow_key_subset: bool = False,
    ) -> None:
        if not callable(getattr(rewriter, "rewrite", None)):
            raise ValueError("rewriter must define rewrite")
        if not callable(getattr(lsh_probe, "probe", None)):
            raise ValueError("lsh_probe must define probe")
        if key_mode not in {"training", "holdout"}:
            raise ValueError("key_mode must be training or holdout")
        if not isinstance(test_only_allow_key_subset, bool):
            raise ValueError("test_only_allow_key_subset must be a bool")
        if key_ids is None:
            key_ids = (
                _CANONICAL_TRAINING_KEY_IDS
                if key_mode == "training"
                else _CANONICAL_HOLDOUT_KEY_IDS
            )
        _validate_key_ids(
            key_ids,
            key_mode=key_mode,
            allow_test_subset=test_only_allow_key_subset,
        )
        self._rewriter = rewriter
        self._lsh_probe = lsh_probe
        self._key_ids = tuple(key_ids)

    def build(
        self,
        units: Sequence[StatementUnit],
        *,
        context: GateBuildContext | None = None,
        source_text: str | None = None,
    ) -> tuple[GateDataGroup, ...]:
        """Build candidate trajectories.

        Formal collection must provide ``source_text`` so byte slices preserve
        comments, blank lines, newline style, and indentation exactly. ``None``
        retains the unit-join fallback for small tests and legacy fixtures only.
        The full source is never included in a rewriter request or gate input.
        """

        if (
            isinstance(units, (str, bytes, bytearray))
            or not isinstance(units, Sequence)
            or not units
            or any(not isinstance(unit, StatementUnit) for unit in units)
        ):
            raise ValueError(
                "units must be a non-empty sequence of StatementUnit instances"
            )
        unit_tuple = tuple(units)
        unit_ids = tuple(unit.unit_id for unit in unit_tuple)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("units contain duplicate unit_id values")
        if any(
            left.end_byte > right.start_byte
            for left, right in zip(unit_tuple, unit_tuple[1:])
        ):
            raise ValueError("units must not overlap and must be in source order")
        source_bytes = _validated_source_bytes(source_text, units=unit_tuple)
        source_content_sha256 = _source_content_sha256(
            source_bytes, units=unit_tuple
        )
        build_context = context if context is not None else GateBuildContext()
        if not isinstance(build_context, GateBuildContext):
            raise ValueError("context must be a GateBuildContext")

        groups: list[GateDataGroup] = []
        for start_index, start in enumerate(unit_tuple):
            if not start.eligible or start.hard_boundary:
                continue
            windows = _enumerate_windows(unit_tuple, start_index)
            trajectories: dict[str, tuple[CandidateObservation, ...]] = {}
            for window in windows:
                length_key = str(len(window))
                request = _make_request(
                    unit_tuple,
                    start_index=start_index,
                    window=window,
                    prompt=build_context.prompt,
                    source_bytes=source_bytes,
                )
                observations: list[CandidateObservation] = []
                original = RewriteCandidate(
                    code=request.original_window,
                    parse_status="ok",
                    unit_count=len(window),
                    same_parent_scope=True,
                    boundary_span=(
                        request.structural_boundary.start_byte,
                        request.structural_boundary.end_byte,
                    ),
                    generation_seed_id="candidate-zero/original",
                    rewrite_config_id="candidate-zero/original",
                )
                observation = self._observe(
                    original,
                    candidate_index=0,
                    parent_descriptor=request.canonical_parent,
                )
                observations.append(observation)
                for candidate_index in _REWRITE_INDICES:
                    candidate = self._rewriter.rewrite(
                        request, candidate_index=candidate_index
                    )
                    if not isinstance(candidate, RewriteCandidate):
                        raise ValueError("rewriter must return RewriteCandidate")
                    observation = self._observe(
                        candidate,
                        candidate_index=candidate_index,
                        parent_descriptor=request.canonical_parent,
                    )
                    observations.append(observation)
                trajectories[length_key] = tuple(observations)

            descriptor = _parent_descriptor(windows[0][0]).canonical
            group_id = _group_id(
                build_context,
                start,
                descriptor,
                windows=windows,
                source_content_sha256=source_content_sha256,
            )
            repository_group = (
                build_context.repository_id
                or build_context.task_id
                or build_context.function_id
            )
            assert repository_group is not None
            repository_group = canonical_gate_source_identity(repository_group)
            function_group = canonical_gate_source_identity(
                build_context.function_id or repository_group
            )
            training_group = GateTrainingGroup(
                schema_version=GATE_DATA_SCHEMA_VERSION,
                group_id=group_id,
                source_id=build_context.source_id,
                source_family=build_context.source_family,
                repository_group=repository_group,
                function_group=function_group,
                language=build_context.language,
                parser_contract_version=build_context.parser_contract_version,
                split="unassigned",
                window_start_unit_id=start.unit_id,
                parent_descriptor=descriptor,
                candidate_window_lengths=tuple(len(window) for window in windows),
                previous_units=tuple(
                    unit.text for unit in unit_tuple[:start_index][-3:]
                ),
                candidates_by_length=trajectories,
            )
            groups.append(
                GateDataGroup(
                    training_group=training_group,
                    repository_id=build_context.repository_id,
                    task_id=build_context.task_id,
                    function_id=build_context.function_id,
                )
            )
        return tuple(groups)

    def _observe(
        self,
        candidate: RewriteCandidate,
        *,
        candidate_index: int,
        parent_descriptor: str,
    ) -> CandidateObservation:
        structurally_valid = (
            candidate.parse_status == "ok"
            and 1 <= candidate.unit_count <= 3
            and candidate.same_parent_scope
        )
        if structurally_valid:
            raw_results = self._lsh_probe.probe(
                window_text=candidate.code,
                parent_descriptor=parent_descriptor,
                key_ids=self._key_ids,
            )
            results = _validate_probe_results(raw_results, key_ids=self._key_ids)
        else:
            results = MappingProxyType({})

        precision_stable = bool(results) and all(
            result.stable_across_precision_modes for result in results.values()
        )
        batch_stable = bool(results) and all(
            result.stable_across_batch_modes for result in results.values()
        )
        lsh_observations = {
            key_id: {
                "hit": result.hit,
                "margin": float(result.margin),
                "stable": result.stable,
            }
            for key_id, result in results.items()
        }
        lsh_signature: tuple[int, ...] | None = None
        if results:
            signatures = {result.signature for result in results.values()}
            if len(signatures) != 1:
                raise ValueError(
                    "all keys must report one key-independent canonical signature"
                )
            lsh_signature = next(iter(signatures))
        observation = CandidateObservation(
            candidate_index=candidate_index,
            code=candidate.code,
            parse_status=candidate.parse_status,
            unit_count=candidate.unit_count,
            same_parent_scope=candidate.same_parent_scope,
            boundary_span=candidate.boundary_span,
            stable_across_precision_modes=precision_stable,
            stable_across_batch_modes=batch_stable,
            lsh_by_key_id=lsh_observations,
            generation_seed_id=candidate.generation_seed_id,
            rewrite_config_id=candidate.rewrite_config_id,
            lsh_signature=lsh_signature,
        )
        return observation


def validate_key_blind_payload(value: object) -> None:
    """Recursively reject secret/LSH fields and existing quality proxies."""

    reject_quality_proxy_fields(_plain_containers(value))
    forbidden_path = _find_leak_field(value, path="")
    if forbidden_path is not None:
        raise ValueError(f"forbidden rewriter field: {forbidden_path}")


def _enumerate_windows(
    units: tuple[StatementUnit, ...], start_index: int
) -> tuple[tuple[StatementUnit, ...], ...]:
    start = units[start_index]
    output: list[tuple[StatementUnit, ...]] = []
    for length in (1, 2, 3):
        end_index = start_index + length
        if end_index > len(units):
            break
        window = units[start_index:end_index]
        if length > 1 and start.compound_header:
            break
        if any(
            unit.hard_boundary
            or not unit.eligible
            or unit.compound_header
            or unit.parent_path != start.parent_path
            or unit.direct_parent_type != start.direct_parent_type
            or unit.depth != start.depth
            for unit in window[1:]
        ):
            break
        output.append(window)
    return tuple(output)


def _make_request(
    units: tuple[StatementUnit, ...],
    *,
    start_index: int,
    window: tuple[StatementUnit, ...],
    prompt: str,
    source_bytes: bytes | None,
) -> RewriteRequest:
    first = window[0]
    end_index = start_index + len(window)
    next_unit = units[end_index] if end_index < len(units) else None
    hard_boundary_after = (
        next_unit is None
        or next_unit.hard_boundary
        or not next_unit.eligible
        or next_unit.compound_header
        or next_unit.parent_path != first.parent_path
        or next_unit.direct_parent_type != first.direct_parent_type
        or next_unit.depth != first.depth
        or len(window) == 3
        or first.compound_header
    )
    boundary = StructuralBoundary(
        start_byte=first.start_byte,
        end_byte=window[-1].end_byte,
        depth=first.depth,
        direct_parent_type=first.direct_parent_type,
        unit_ids=tuple(unit.unit_id for unit in window),
        compound_singleton=first.compound_header,
        hard_boundary_after=hard_boundary_after,
    )
    if source_bytes is None:
        completed_prefix = "\n".join(unit.text for unit in units[:start_index])
        original_window = "\n".join(unit.text for unit in window)
    else:
        completed_prefix = _decode_source_slice(
            source_bytes, 0, first.start_byte, field_name="completed_prefix"
        )
        original_window = _decode_source_slice(
            source_bytes,
            first.start_byte,
            window[-1].end_byte,
            field_name="original_window",
        )
    return RewriteRequest(
        prompt=prompt,
        completed_prefix=completed_prefix,
        original_window=original_window,
        canonical_parent=_parent_descriptor(first).canonical,
        window_start_unit_id=first.unit_id,
        window_length=len(window),
        structural_boundary=boundary,
    )


def _parent_descriptor(unit: StatementUnit) -> ParentDescriptor:
    return ParentDescriptor(
        contract_version=WINDOW_CONTRACT_VERSION,
        ancestor_node_types=unit.parent_path[:-1],
        direct_parent_type=unit.direct_parent_type,
        first_unit_ordinal=unit.direct_child_ordinal,
        compound_header_role="header" if unit.compound_header else "body",
    )


def _validate_probe_results(
    value: object, *, key_ids: tuple[str, ...]
) -> Mapping[str, LshProbeResult]:
    if not isinstance(value, Mapping):
        raise ValueError("lsh probe must return a mapping")
    if set(value) != set(key_ids):
        raise ValueError("lsh probe result keys must exactly match requested key_ids")
    output: dict[str, LshProbeResult] = {}
    for key_id in key_ids:
        result = value[key_id]
        if not isinstance(result, LshProbeResult):
            raise ValueError("lsh probe values must be LshProbeResult")
        output[key_id] = result
    return MappingProxyType(output)


def _validated_source_bytes(
    source_text: object,
    *,
    units: tuple[StatementUnit, ...],
) -> bytes | None:
    if source_text is None:
        return None
    if not isinstance(source_text, str):
        raise ValueError("source_text must be a string or None")
    try:
        source_bytes = source_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("source_text must be valid UTF-8 text") from exc
    for unit in units:
        if unit.end_byte > len(source_bytes):
            raise ValueError("unit byte span exceeds source_text length")
        sliced = _decode_source_slice(
            source_bytes,
            unit.start_byte,
            unit.end_byte,
            field_name=f"unit {unit.unit_id}",
        )
        if sliced != unit.text:
            raise ValueError(
                f"unit {unit.unit_id} byte span does not match source_text"
            )
    return source_bytes


def _decode_source_slice(
    source_bytes: bytes,
    start: int,
    end: int,
    *,
    field_name: str,
) -> str:
    if start < 0 or end < start or end > len(source_bytes):
        raise ValueError(f"{field_name} has an invalid source_text byte span")
    try:
        return source_bytes[start:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{field_name} byte span is not aligned to UTF-8 boundaries"
        ) from exc


def _validate_key_ids(
    key_ids: object,
    *,
    key_mode: str,
    allow_test_subset: bool,
) -> None:
    if not isinstance(key_ids, tuple) or not key_ids:
        raise ValueError("key_ids must be a non-empty tuple")
    if len(set(key_ids)) != len(key_ids):
        raise ValueError("key_ids contain duplicate values")
    for key_id in key_ids:
        if not isinstance(key_id, str):
            raise ValueError("key_ids must contain strings")
        normalized = _normalized_field_name(key_id)
        if "deployment" in normalized:
            raise ValueError("deployment key IDs are forbidden")
        if _KEY_ID_RE.fullmatch(key_id) is None:
            raise ValueError("key IDs must match train-key-NNN or holdout-key-NNN")
    prefixes = {
        "training" if key_id.startswith("train-key-") else "holdout"
        for key_id in key_ids
    }
    if len(prefixes) != 1:
        raise ValueError("mixed training and holdout key IDs are forbidden")
    actual_mode = next(iter(prefixes))
    if actual_mode != key_mode:
        raise ValueError("key IDs must match the explicit key_mode")
    canonical = (
        _CANONICAL_TRAINING_KEY_IDS
        if key_mode == "training"
        else _CANONICAL_HOLDOUT_KEY_IDS
    )
    if allow_test_subset:
        if any(key_id not in canonical for key_id in key_ids):
            raise ValueError("test key subset must use canonical key IDs")
    elif tuple(key_ids) != canonical:
        required = 32 if key_mode == "training" else 8
        raise ValueError(
            f"{key_mode} mode requires exactly {required} canonical key IDs"
        )


def _group_id(
    context: GateBuildContext,
    unit: StatementUnit,
    descriptor: str,
    *,
    windows: tuple[tuple[StatementUnit, ...], ...],
    source_content_sha256: str,
) -> str:
    payload = {
        "contract": "wfcllm-gate-group-identity/v2",
        "context": {
            "prompt": context.prompt,
            "source_id": context.source_id,
            "source_family": context.source_family,
            "repository_id": context.repository_id,
            "task_id": context.task_id,
            "function_id": context.function_id,
            "language": context.language,
            "parser_contract_version": context.parser_contract_version,
        },
        "start_unit": _unit_identity_row(unit),
        "parent_descriptor": descriptor,
        "windows": [
            [_unit_identity_row(window_unit) for window_unit in window]
            for window in windows
        ],
        "source_content_sha256": source_content_sha256,
    }
    message = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"gate-group/v2:sha256:{hashlib.sha256(message).hexdigest()}"


def _unit_identity_row(unit: StatementUnit) -> dict[str, object]:
    return {
        "unit_id": unit.unit_id,
        "node_type": unit.node_type,
        "text": unit.text,
        "start_byte": unit.start_byte,
        "end_byte": unit.end_byte,
        "start_line": unit.start_line,
        "end_line": unit.end_line,
        "depth": unit.depth,
        "parent_path": list(unit.parent_path),
        "direct_parent_type": unit.direct_parent_type,
        "direct_child_ordinal": unit.direct_child_ordinal,
        "eligible": unit.eligible,
        "hard_boundary": unit.hard_boundary,
        "compound_header": unit.compound_header,
    }


def _source_content_sha256(
    source_bytes: bytes | None,
    *,
    units: tuple[StatementUnit, ...],
) -> str:
    if source_bytes is not None:
        return hashlib.sha256(source_bytes).hexdigest()
    fallback = json.dumps(
        [_unit_identity_row(unit) for unit in units],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(fallback).hexdigest()


def _find_leak_field(value: object, *, path: str) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            normalized = _normalized_field_name(key_text)
            if (
                normalized in _EXACT_SECRET_FIELD_NAMES
                or normalized in _NORMALIZED_QUALITY_FIELDS
                or any(part in normalized for part in _LEAK_FIELD_PARTS)
            ):
                return child_path
            found = _find_leak_field(nested, path=child_path)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found = _find_leak_field(item, path=child_path)
            if found is not None:
                return found
    return None


def _plain_containers(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_containers(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain_containers(item) for item in value]
    return value


def _normalized_field_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _validate_span(span: object) -> None:
    if (
        not isinstance(span, tuple)
        or len(span) != 2
        or any(type(item) is not int for item in span)
        or span[0] < 0
        or span[0] > span[1]
    ):
        raise ValueError("boundary_span must be an ordered non-negative pair")


def _require_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
