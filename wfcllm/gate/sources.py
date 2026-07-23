"""Allowlisted gate-data sources and deterministic group-aware splitting."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from wfcllm.method.contracts import reject_quality_proxy_fields

GATE_SOURCE_MANIFEST_VERSION = "wfcllm-gate-source-manifest/v1"
GATE_SPLIT_CONTRACT_VERSION = "wfcllm-gate-split/v1"

APPROVED_GATE_SOURCE_FAMILIES = frozenset(
    {
        "main_generation",
        "mbpp_train",
        "mbpp_validation",
        "oss_python",
        "oss_cpp",
        "oss_java",
        "oss_js",
        "parser_boundary",
    }
)
_HOLDOUT_MARKERS = (
    "humaneval",
    "deploymentdetection",
    "finalnegativecalibration",
)


class SplittableGroup(Protocol):
    item_id: str
    split_group_id: str
    repository_id: str | None
    task_id: str | None
    function_id: str | None


@dataclass(frozen=True)
class GateSourceRecord:
    """One local source item; model provenance remains metadata only."""

    source_family: str
    source_id: str
    code: str
    repository_id: str | None = None
    task_id: str | None = None
    function_id: str | None = None
    source_model_id: str | None = None
    license_id: str | None = None
    contract_or_hard_set: bool = False

    def __post_init__(self) -> None:
        _require_identity_string("source_family", self.source_family)
        _require_identity_string("source_id", self.source_id)
        _require_nonempty_string("code", self.code)
        try:
            self.code.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("code must be valid UTF-8 text") from exc
        validate_gate_source_family(self.source_family)
        for name in (
            "repository_id",
            "task_id",
            "function_id",
            "source_model_id",
            "license_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_identity_string(name, value)
        _validate_source_identity_fields(self)
        if not isinstance(self.contract_or_hard_set, bool):
            raise ValueError("contract_or_hard_set must be a bool")
        if not any((self.repository_id, self.task_id, self.function_id)):
            raise ValueError("source requires a repository, task, or function ID")
        if self.source_family in {"oss_python", "oss_cpp", "oss_java", "oss_js"}:
            if self.repository_id is None:
                raise ValueError("OSS source families require repository_id")
            if self.license_id is None or not self.license_id.strip():
                raise ValueError("OSS source families require an explicit license_id")
        if self.source_family == "parser_boundary":
            if not self.contract_or_hard_set:
                raise ValueError(
                    "parser_boundary requires contract_or_hard_set=true"
                )
        elif self.contract_or_hard_set:
            raise ValueError(
                "contract_or_hard_set is reserved for parser_boundary sources"
            )

    @property
    def item_id(self) -> str:
        return self.source_id

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
        raise ValueError("source has no split identity")

    def gate_input_metadata(self) -> dict[str, object]:
        """Return only grouping facts allowed beside a serialized gate input."""

        payload: dict[str, object] = {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "repository_id": self.repository_id,
            "task_id": self.task_id,
            "function_id": self.function_id,
            "contract_or_hard_set": self.contract_or_hard_set,
        }
        reject_quality_proxy_fields(payload)
        return payload

    def manifest_metadata(self) -> dict[str, object]:
        """Return provenance metadata without copying source code."""

        payload = {
            **self.gate_input_metadata(),
            "source_model_id": self.source_model_id,
            "license_id": self.license_id,
            "code_sha256": hashlib.sha256(
                self.code.encode("utf-8")
            ).hexdigest(),
        }
        reject_quality_proxy_fields(payload)
        return payload


class GateSourceLoader:
    """In-memory/local-source registry with a strict family allowlist."""

    def __init__(self, records: Sequence[GateSourceRecord]) -> None:
        if (
            isinstance(records, (str, bytes, bytearray))
            or not isinstance(records, Sequence)
            or any(not isinstance(record, GateSourceRecord) for record in records)
        ):
            raise ValueError("records must be a sequence of GateSourceRecord")
        snapshot = tuple(records)
        for record in snapshot:
            _validate_source_identity_fields(record)
        _reject_group_identity_aliases(snapshot)
        source_ids = tuple(record.source_id for record in snapshot)
        if len({_normalized_name(value) for value in source_ids}) != len(source_ids):
            raise ValueError("records contain duplicate source_id values")
        self._records = snapshot

    def load(self, *, source_family: str) -> tuple[GateSourceRecord, ...]:
        _require_identity_string("source_family", source_family)
        _reject_holdout_name(source_family)
        if source_family not in APPROVED_GATE_SOURCE_FAMILIES:
            raise ValueError(
                "source_family must be an approved local gate-data source"
            )
        return tuple(
            record
            for record in self._records
            if record.source_family == source_family
        )


@dataclass(frozen=True)
class GateSourceManifest:
    """A complete formal-source manifest with model-diversity enforcement."""

    records: tuple[GateSourceRecord, ...]

    def __init__(self, records: Sequence[GateSourceRecord]) -> None:
        if (
            isinstance(records, (str, bytes, bytearray))
            or not isinstance(records, Sequence)
            or any(not isinstance(record, GateSourceRecord) for record in records)
        ):
            raise ValueError("records must be a sequence of GateSourceRecord")
        snapshot = tuple(records)
        for record in snapshot:
            _validate_source_identity_fields(record)
            try:
                record.code.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("source record code must be valid UTF-8") from exc
        _reject_group_identity_aliases(snapshot)
        source_ids = tuple(record.source_id for record in snapshot)
        if len({_normalized_name(value) for value in source_ids}) != len(source_ids):
            raise ValueError("manifest contains duplicate source_id values")
        model_ids = {
            _normalized_name(record.source_model_id)
            for record in snapshot
            if record.source_family == "main_generation"
            and record.source_model_id is not None
        }
        if len(model_ids) < 3:
            raise ValueError(
                "a complete source manifest requires at least three "
                "main_generation source models"
            )
        formal = tuple(
            record for record in snapshot if not record.contract_or_hard_set
        )
        group_counts = _split_group_counts_by_family(formal)
        oss_repository_groups = {
            canonical_gate_source_identity(record.repository_id)
            for record in formal
            if record.source_family in {"oss_python", "oss_cpp", "oss_java", "oss_js"}
            and record.repository_id is not None
        }
        if not oss_repository_groups:
            raise ValueError(
                "a complete source manifest requires licensed OSS repositories"
            )
        largest_other_family = max(
            (
                count
                for family, count in group_counts.items()
                if family not in {"oss_python", "oss_cpp", "oss_java", "oss_js"}
            ),
            default=0,
        )
        if len(oss_repository_groups) < largest_other_family:
            raise ValueError(
                "OSS repository groups must be a primary formal source"
            )
        object.__setattr__(self, "records", snapshot)

    @property
    def source_model_ids(self) -> tuple[str, ...]:
        by_canonical_id: dict[str, str] = {}
        for record in sorted(self.formal_records, key=lambda item: item.source_id):
            if (
                record.source_family == "main_generation"
                and record.source_model_id is not None
            ):
                by_canonical_id.setdefault(
                    _normalized_name(record.source_model_id),
                    record.source_model_id,
                )
        return tuple(
            by_canonical_id[key] for key in sorted(by_canonical_id)
        )

    @property
    def formal_records(self) -> tuple[GateSourceRecord, ...]:
        """Records eligible for formal label ratios and threshold fitting."""

        return tuple(
            record for record in self.records if not record.contract_or_hard_set
        )

    @property
    def hard_set_records(self) -> tuple[GateSourceRecord, ...]:
        return tuple(record for record in self.records if record.contract_or_hard_set)

    def to_dict(self) -> dict[str, Any]:
        canonical_rows = sorted(
            (record.manifest_metadata() for record in self.records),
            key=lambda row: (
                _normalized_name(str(row["source_id"])),
                _canonical_json_bytes(row),
            ),
        )
        canonical_bytes = _canonical_manifest_bytes(canonical_rows)
        group_counts = _split_group_counts_by_family(self.formal_records)
        oss_repository_group_count = len(
            {
                canonical_gate_source_identity(record.repository_id)
                for record in self.formal_records
                if record.source_family in {"oss_python", "oss_cpp", "oss_java", "oss_js"}
                and record.repository_id is not None
            }
        )
        payload: dict[str, Any] = {
            "schema_version": GATE_SOURCE_MANIFEST_VERSION,
            "manifest_id": (
                "wfcllm-gate-sources/v1:sha256:"
                + hashlib.sha256(canonical_bytes).hexdigest()
            ),
            "source_count": len(self.records),
            "formal_source_count": len(self.formal_records),
            "hard_set_source_count": len(self.hard_set_records),
            "source_model_ids": list(self.source_model_ids),
            "split_group_count_by_family": group_counts,
            "oss_repository_group_count": oss_repository_group_count,
            "sources": canonical_rows,
        }
        reject_quality_proxy_fields(payload)
        return payload


@dataclass(frozen=True)
class SplitAssignments:
    """Read-only item assignments plus group-level audit views."""

    _by_item_id: Mapping[str, str]
    _group_by_item_id: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self._by_item_id, Mapping) or not isinstance(
            self._group_by_item_id, Mapping
        ):
            raise ValueError("split assignments must be mappings")
        if set(self._by_item_id) != set(self._group_by_item_id):
            raise ValueError("assignment mappings must contain the same item IDs")
        for item_id, split_name in self._by_item_id.items():
            _require_identity_string("item_id", item_id)
            if split_name not in {"train", "validation", "test"}:
                raise ValueError("split must be train, validation, or test")
            _require_identity_string(
                "split_group_id", self._group_by_item_id[item_id]
            )
        object.__setattr__(
            self, "_by_item_id", MappingProxyType(dict(self._by_item_id))
        )
        object.__setattr__(
            self,
            "_group_by_item_id",
            MappingProxyType(dict(self._group_by_item_id)),
        )

    def __getitem__(self, item_id: str) -> str:
        return self._by_item_id[item_id]

    def by_repository_group(self) -> dict[str, frozenset[str]]:
        """Return all split names seen for each indivisible split group."""

        output: dict[str, set[str]] = {}
        for item_id, split_name in self._by_item_id.items():
            group_id = self._group_by_item_id[item_id]
            output.setdefault(group_id, set()).add(split_name)
        return {
            group_id: frozenset(split_names)
            for group_id, split_names in output.items()
        }

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": GATE_SPLIT_CONTRACT_VERSION,
            "assignments": dict(self._by_item_id),
            "split_groups": dict(self._group_by_item_id),
        }
        reject_quality_proxy_fields(payload)
        return payload


class GateDataSplitter:
    """Assign whole repository/task/function groups using SHA-256."""

    def __init__(self, *, seed: str) -> None:
        _require_nonempty_string("seed", seed)
        self._seed = seed

    def assign(self, item: SplittableGroup) -> str:
        _, split_group_id, _ = _splittable_identity(item)
        return _assign_split_group_id(self._seed, split_group_id)

    def assign_all(
        self, items: Sequence[SplittableGroup]
    ) -> SplitAssignments:
        if (
            isinstance(items, (str, bytes, bytearray))
            or not isinstance(items, Sequence)
        ):
            raise ValueError("items must be a sequence of splittable groups")
        by_item_id: dict[str, str] = {}
        group_by_item_id: dict[str, str] = {}
        for item in items:
            item_id, split_group_id, _ = _splittable_identity(item)
            if item_id in by_item_id:
                raise ValueError(f"duplicate item_id: {item_id}")
            by_item_id[item_id] = _assign_split_group_id(
                self._seed, split_group_id
            )
            group_by_item_id[item_id] = split_group_id
        return SplitAssignments(by_item_id, group_by_item_id)


def _splittable_identity(item: object) -> tuple[str, str, str | None]:
    try:
        item_id = getattr(item, "item_id")
        split_group_id = getattr(item, "split_group_id")
        repository_id = getattr(item, "repository_id")
        task_id = getattr(item, "task_id")
        function_id = getattr(item, "function_id")
    except (AttributeError, ValueError) as exc:
        raise ValueError("item must expose a valid split identity") from exc
    _require_identity_string("item_id", item_id)
    _require_identity_string("split_group_id", split_group_id)
    if repository_id is not None:
        _require_identity_string("repository_id", repository_id)
    if task_id is not None:
        _require_identity_string("task_id", task_id)
    if function_id is not None:
        _require_identity_string("function_id", function_id)
    if not any((repository_id, task_id, function_id)):
        raise ValueError("item must provide repository, task, or function ID")
    canonical_split_group_id = (
        "repository:" + canonical_gate_source_identity(repository_id)
        if repository_id is not None
        else "task:" + canonical_gate_source_identity(task_id)
        if task_id is not None
        else "function:" + canonical_gate_source_identity(function_id)
    )
    if split_group_id != canonical_split_group_id:
        raise ValueError("reported split_group_id contradicts canonical identity")
    return item_id, canonical_split_group_id, repository_id


def _reject_holdout_name(value: str) -> None:
    normalized = canonical_gate_source_identity(value)
    if any(marker in normalized for marker in _HOLDOUT_MARKERS):
        raise ValueError(
            "HumanEval, deployment detection, and final calibration are holdout-only"
        )


def _normalized_name(value: str) -> str:
    return canonical_gate_source_identity(value)


def canonical_gate_source_identity(value: str) -> str:
    """Return the shared NFKC/casefold/alnum identity representation."""

    if not isinstance(value, str) or not value:
        raise ValueError("source identity must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    canonical = "".join(
        character for character in normalized if character.isalnum()
    )
    if not canonical:
        raise ValueError("source identity must contain letters or digits")
    return canonical


def validate_gate_source_identity(name: str, value: object) -> None:
    """Apply the shared identity and holdout policy to one public field."""

    _require_identity_string(name, value)
    assert isinstance(value, str)
    _reject_holdout_name(value)


def validate_gate_source_family(value: object) -> None:
    """Require one of the approved formal gate-data source families."""

    validate_gate_source_identity("source_family", value)
    if value not in APPROVED_GATE_SOURCE_FAMILIES:
        raise ValueError("source_family must be an approved gate-data source")


def _canonical_manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"wfcllm-gate-source-manifest/private-free/v1\0" + _canonical_json_bytes(
        rows
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _split_group_counts_by_family(
    records: Sequence[GateSourceRecord],
) -> dict[str, int]:
    groups: dict[str, set[str]] = {}
    for record in records:
        groups.setdefault(record.source_family, set()).add(record.split_group_id)
    return {
        family: len(groups[family])
        for family in sorted(groups)
    }


def _assign_split_group_id(seed: str, split_group_id: str) -> str:
    """Pure SHA-256 split assignment over one already-snapshotted group ID."""

    digest = hashlib.sha256((seed + split_group_id).encode("utf-8")).digest()
    fraction = int.from_bytes(digest, "big") / (1 << 256)
    if fraction < 0.8:
        return "train"
    if fraction < 0.9:
        return "validation"
    return "test"


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_identity_string(name: str, value: object) -> None:
    _require_nonempty_string(name, value)
    assert isinstance(value, str)
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")


def _validate_source_identity_fields(record: GateSourceRecord) -> None:
    for name in (
        "source_family",
        "source_id",
        "repository_id",
        "task_id",
        "function_id",
        "source_model_id",
    ):
        value = getattr(record, name)
        if value is not None:
            validate_gate_source_identity(name, value)


def _reject_group_identity_aliases(
    records: Sequence[GateSourceRecord],
) -> None:
    for field_name in ("repository_id", "task_id", "function_id"):
        originals_by_canonical: dict[str, str] = {}
        for record in records:
            value = getattr(record, field_name)
            if value is None:
                continue
            canonical = canonical_gate_source_identity(value)
            previous = originals_by_canonical.setdefault(canonical, value)
            if previous != value:
                raise ValueError(
                    f"{field_name} alias collision: {previous!r} and {value!r}"
                )
