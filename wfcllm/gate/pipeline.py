"""Atomic low-level gate data, training, and validation pipelines."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Protocol

from wfcllm.gate.feasibility import (
    FEASIBILITY_CONTRACT_VERSION,
    FEASIBILITY_THRESHOLD_ITEMS,
    FeasibilityGroup,
    GateDataFeasibilitySummary,
    evaluate_gate_data_feasibility,
)
from wfcllm.gate.bundle import GateBundle, sha256_directory
from wfcllm.gate.validation import GateValidationSummary
from wfcllm.gate.data import LshProbeResult
from wfcllm.gate.labels import GateLabels, build_gate_labels
from wfcllm.gate.schema import CandidateObservation

GATE_DATA_MANIFEST_VERSION = "wfcllm-gate-data-manifest/v1"
GATE_TRAIN_MANIFEST_VERSION = "wfcllm-gate-train-candidate/v1"
GATE_VALIDATE_MANIFEST_VERSION = "wfcllm-gate-validate-publication/v1"
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_PUBLIC_JSON_BYTES = 1024 * 1024
_MAX_GATE_GROUPS = 2_000
_MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
_MAX_CANDIDATE_CODE_BYTES = 512 * 1024
_MAX_METADATA_JSON_BYTES = 64 * 1024 * 1024
_MAX_GATE_ARTIFACT_BYTES = 32 * 1024 * 1024 * 1024
_EVIDENCE_LRU_SIZE = 512
_MAX_CANDIDATE_FILES = 128
_MAX_CANDIDATE_FILE_BYTES = 512 * 1024 * 1024
_MAX_CANDIDATE_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_TOKENIZER_FILES = 64
_MAX_CANDIDATE_DEPTH = 8
_MAX_CANDIDATE_RELATIVE_PATH_BYTES = 1024
_MAX_CANDIDATE_SEGMENT_BYTES = 255
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LABEL_CACHE: OrderedDict[str, GateLabels] = OrderedDict()


@dataclass(frozen=True)
class GateDataPipelineConfig:
    output_root: Path
    scale: str
    config_hash: str
    parser_contract: str
    rewriter_config_hash: str
    semantic_encoder_hash: str
    lsh_config_hash: str
    feasibility_contract: str
    feasibility_thresholds: tuple[tuple[str, int | float], ...]
    pilot_feasibility_path: Path | None = None
    max_groups: int | None = None
    fast_experimental: bool = False

    def __post_init__(self) -> None:
        _path("output_root", self.output_root)
        if self.scale not in {"pilot", "full"}:
            raise ValueError("scale must be pilot or full")
        for name in ("config_hash", "rewriter_config_hash", "semantic_encoder_hash", "lsh_config_hash"):
            _digest(name, getattr(self, name))
        if not isinstance(self.parser_contract, str) or not self.parser_contract:
            raise ValueError("parser_contract must be a non-empty string")
        if self.pilot_feasibility_path is not None:
            _path("pilot_feasibility_path", self.pilot_feasibility_path)
        if self.max_groups is not None and (
            type(self.max_groups) is not int or self.max_groups <= 0
        ):
            raise ValueError("max_groups must be a positive integer or None")
        if type(self.fast_experimental) is not bool:
            raise ValueError("fast_experimental must be a bool")
        if self.feasibility_contract != FEASIBILITY_CONTRACT_VERSION:
            raise ValueError("feasibility_contract must remain gate-data-feasibility/v1")
        if self.feasibility_thresholds != FEASIBILITY_THRESHOLD_ITEMS:
            raise ValueError("feasibility thresholds are frozen by gate-data-feasibility/v1")


@dataclass(frozen=True)
class GateTrainPipelineConfig:
    output_root: Path
    data_dir: Path
    config_hash: str
    pilot_feasibility_path: Path | None = None
    fast_experimental: bool = False

    def __post_init__(self) -> None:
        for name in ("output_root", "data_dir"):
            _path(name, getattr(self, name))
        if self.pilot_feasibility_path is not None:
            _path("pilot_feasibility_path", self.pilot_feasibility_path)
        if type(self.fast_experimental) is not bool:
            raise ValueError("fast_experimental must be a bool")
        _digest("config_hash", self.config_hash)


@dataclass(frozen=True)
class GateValidatePipelineConfig:
    output_root: Path
    candidate_bundle: Path
    data_dir: Path
    config_hash: str

    def __post_init__(self) -> None:
        for name in ("output_root", "candidate_bundle", "data_dir"):
            _path(name, getattr(self, name))
        _digest("config_hash", self.config_hash)


@dataclass(frozen=True)
class KeyBankSnapshot:
    key_ids: tuple[str, ...]
    bank_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.key_ids, tuple) or not self.key_ids or any(not isinstance(value, str) or not value for value in self.key_ids):
            raise ValueError("key_ids must be a non-empty tuple of strings")
        if len(set(self.key_ids)) != len(self.key_ids):
            raise ValueError("key IDs must be unique")
        if re.fullmatch(r"(?:training|holdout)-key-bank/v1:sha256:[0-9a-f]{64}", self.bank_id) is None:
            raise ValueError("bank_id must be an irreversible key bank identifier")


@dataclass(frozen=True)
class GatePipelineGroup:
    group_id: str
    split_group_id: str
    split: str
    suitable_target: bool
    close_target: bool
    window_lengths: tuple[int, ...]
    statement_family: str
    r1_success_rate: float
    r3_success_rate: float
    holdout_success_rate: float
    repository_id: str
    task_id: str
    generation_model_id: str
    structural_invalid_rate: float
    numeric_instability_rate: float
    first_hit_candidate_position: int | None
    candidate_indices_by_window_length: Mapping[int, tuple[int, ...]]
    observed_training_key_ids: tuple[str, ...]
    observed_holdout_key_ids: tuple[str, ...]
    candidate_observations_by_length: Mapping[str, tuple[CandidateObservation, ...]]
    probe_results_by_length: Mapping[
        str, tuple[Mapping[str, LshProbeResult], ...]
    ]
    row: Mapping[str, Any]

    def __post_init__(self) -> None:
        feasibility = self.to_feasibility()
        if not isinstance(self.split_group_id, str) or not self.split_group_id:
            raise ValueError("split_group_id must be a non-empty string")
        if type(self.close_target) is not bool:
            raise ValueError("close_target must be a bool")
        if self.suitable_target and not self.close_target:
            raise ValueError("suitable_target requires close_target")
        if not isinstance(self.row, Mapping):
            raise ValueError("row must be a mapping")
        if self.row.get("group_id") != feasibility.group_id or self.row.get("split") != self.split:
            raise ValueError("row group_id/split must match the independent group")
        _validate_public_window_row(dict(self.row), diagnostic_expected=None)
        if not isinstance(self.candidate_indices_by_window_length, Mapping):
            raise ValueError("candidate_indices_by_window_length must be a mapping")
        if any(not isinstance(value, tuple) for value in (self.observed_training_key_ids, self.observed_holdout_key_ids)):
            raise ValueError("observed key IDs must be tuples")
        object.__setattr__(
            self,
            "candidate_indices_by_window_length",
            MappingProxyType(dict(self.candidate_indices_by_window_length)),
        )
        observations = self.candidate_observations_by_length
        probes = self.probe_results_by_length
        if not isinstance(observations, Mapping) or not isinstance(probes, Mapping):
            raise ValueError("candidate observations and probe results must be mappings")
        object.__setattr__(
            self,
            "candidate_observations_by_length",
            MappingProxyType({key: tuple(rows) for key, rows in observations.items()}),
        )
        object.__setattr__(
            self,
            "probe_results_by_length",
            MappingProxyType(
                {
                    key: tuple(
                        row if isinstance(row, MappingProxyType) else MappingProxyType(dict(row))
                        for row in rows
                    )
                    for key, rows in probes.items()
                }
            ),
        )

    def to_feasibility(self) -> FeasibilityGroup:
        return FeasibilityGroup(
            self.group_id,
            self.suitable_target,
            self.window_lengths,
            self.statement_family,
            self.r1_success_rate,
            self.r3_success_rate,
            self.holdout_success_rate,
            self.split,
            self.repository_id,
            self.task_id,
            self.generation_model_id,
            self.structural_invalid_rate,
            self.numeric_instability_rate,
            self.first_hit_candidate_position,
        )


@dataclass(frozen=True)
class GateGroupIdentity:
    """Immutable source identity shared by every gate-data stage."""

    group_id: str
    split_group_id: str
    repository_id: str
    task_id: str
    generation_model_id: str
    statement_family: str
    window_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "group_id", "split_group_id", "repository_id", "task_id",
            "generation_model_id", "statement_family",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.window_lengths != (1, 2, 3):
            raise ValueError("identity window_lengths must remain W1/W2/W3")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(
                {
                    "group_id": self.group_id,
                    "split_group_id": self.split_group_id,
                    "repository_id": self.repository_id,
                    "task_id": self.task_id,
                    "generation_model_id": self.generation_model_id,
                    "statement_family": self.statement_family,
                    "window_lengths": list(self.window_lengths),
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class ParsedWindowGroup:
    """Parser-stage record; it contains no rewrite, probe, label, or split state."""

    identity: GateGroupIdentity
    parser_contract: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, GateGroupIdentity):
            raise ValueError("parsed group requires GateGroupIdentity")
        if not isinstance(self.parser_contract, str) or not self.parser_contract:
            raise ValueError("parser_contract must be a non-empty string")


@dataclass(frozen=True)
class CandidateTrajectoryGroup:
    """Generation-stage record containing only ordered candidate identities."""

    parsed: ParsedWindowGroup
    candidate_indices_by_window_length: Mapping[int, tuple[int, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.parsed, ParsedWindowGroup):
            raise ValueError("candidate trajectory requires ParsedWindowGroup")
        values = dict(self.candidate_indices_by_window_length)
        if set(values) != {1, 2, 3} or any(
            values[length] != tuple(range(4)) for length in (1, 2, 3)
        ):
            raise ValueError("candidate trajectory must contain candidate 0 through 3 for W1/W2/W3")
        object.__setattr__(self, "candidate_indices_by_window_length", MappingProxyType(values))

    @property
    def identity(self) -> GateGroupIdentity:
        return self.parsed.identity


@dataclass(frozen=True)
class ProbedGroup:
    """Probe-stage record with complete immutable 32+8-key evidence."""

    trajectory: CandidateTrajectoryGroup
    candidate_observations_by_length: Mapping[str, tuple[CandidateObservation, ...]]
    probe_results_by_length: Mapping[str, tuple[Mapping[str, LshProbeResult], ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, CandidateTrajectoryGroup):
            raise ValueError("probed group requires CandidateTrajectoryGroup")
        observations = {key: tuple(value) for key, value in self.candidate_observations_by_length.items()}
        probes = {key: tuple(value) for key, value in self.probe_results_by_length.items()}
        if set(observations) != {"1", "2", "3"} or set(probes) != {"1", "2", "3"}:
            raise ValueError("probed group must contain W1/W2/W3 evidence")
        object.__setattr__(self, "candidate_observations_by_length", MappingProxyType(observations))
        object.__setattr__(self, "probe_results_by_length", MappingProxyType(probes))

    @property
    def identity(self) -> GateGroupIdentity:
        return self.trajectory.identity


@dataclass(frozen=True)
class LabeledGroup:
    """Label-stage record derived only from its preceding probe evidence."""

    probed: ProbedGroup
    labels_by_window_length: Mapping[int, GateLabels]
    holdout_success_rate: float
    first_hit_candidate_position: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.probed, ProbedGroup):
            raise ValueError("labeled group requires ProbedGroup")
        labels = dict(self.labels_by_window_length)
        if set(labels) != {1, 2, 3} or any(not isinstance(value, GateLabels) for value in labels.values()):
            raise ValueError("labeled group requires Task6 labels for W1/W2/W3")
        object.__setattr__(self, "labels_by_window_length", MappingProxyType(labels))

    @property
    def identity(self) -> GateGroupIdentity:
        return self.probed.identity


@dataclass(frozen=True)
class SplitGroup:
    """Final stage record; split metadata cannot flow backward into labels."""

    labeled: LabeledGroup
    split: str
    row: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.labeled, LabeledGroup):
            raise ValueError("split group requires LabeledGroup")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        row = dict(self.row)
        if row.get("group_id") != self.identity.group_id or row.get("split") != self.split:
            raise ValueError("split row identity contradicts immutable base identity")
        _canonical_bytes(row)
        object.__setattr__(self, "row", MappingProxyType(row))

    @property
    def identity(self) -> GateGroupIdentity:
        return self.labeled.identity


@dataclass(frozen=True)
class _CompactGroupIndex:
    """Evidence-free state retained while the streamed collection is finalized."""

    group_id: str
    identity_sha256: str
    split_group_id: str
    split: str
    close_target: bool
    suitable_target: bool


@dataclass
class _BoundedJsonlWriter:
    path: Path
    handle: Any
    bytes_written: int = 0

    @classmethod
    def open(cls, path: Path) -> _BoundedJsonlWriter:
        return cls(path=path, handle=path.open("wb"))

    def write(self, value: object) -> None:
        line = _canonical_bytes(value) + b"\n"
        if len(line) > _MAX_JSONL_LINE_BYTES:
            raise ValueError(f"JSONL row exceeds {_MAX_JSONL_LINE_BYTES} bytes: {self.path.name}")
        if self.bytes_written + len(line) > _MAX_GATE_ARTIFACT_BYTES:
            raise ValueError(f"gate artifact exceeds {_MAX_GATE_ARTIFACT_BYTES} bytes: {self.path.name}")
        self.handle.write(line)
        self.bytes_written += len(line)

    def close(self) -> None:
        self.handle.close()


@dataclass(frozen=True)
class GateDataResult:
    output_dir: Path
    manifest_path: Path
    data_path: Path
    feasibility_path: Path
    manifest: Mapping[str, Any]
    feasibility: GateDataFeasibilitySummary
    group_count: int


@dataclass(frozen=True)
class GateTrainResult:
    output_dir: Path
    candidate_bundle_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class ValidationOutcome:
    validated: bool
    summary: GateValidationSummary | Mapping[str, Any]
    bundle: object | None

    def __post_init__(self) -> None:
        if type(self.validated) is not bool or not isinstance(self.summary, (GateValidationSummary, Mapping)):
            raise ValueError("validation outcome schema mismatch")
        if self.validated and self.bundle is None:
            raise ValueError("validated outcome requires a bundle")


@dataclass(frozen=True)
class GateValidateResult:
    validated: bool
    output_dir: Path
    bundle: object | None
    manifest_path: Path | None
    failed_summary_path: Path | None
    summary: Mapping[str, Any]


class GatePipelineDependencies(Protocol):
    def load_source_manifest(self, config: GateDataPipelineConfig) -> Mapping[str, Any]: ...
    def load_key_bank(self, *, role: str, expected_count: int, config: GateDataPipelineConfig) -> KeyBankSnapshot: ...
    def parse_statement_units(self, source_manifest: Mapping[str, Any], config: GateDataPipelineConfig) -> object: ...
    def generate_candidate_trajectories(self, parsed_units: object, config: GateDataPipelineConfig) -> Iterable[GatePipelineGroup]: ...
    def run_multi_key_lsh_probe(self, groups: Iterable[GatePipelineGroup], *, training_key_ids: tuple[str, ...], holdout_key_ids: tuple[str, ...], config: GateDataPipelineConfig) -> Iterable[GatePipelineGroup]: ...
    def split_groups(self, groups: Iterable[GatePipelineGroup], config: GateDataPipelineConfig) -> Iterable[GatePipelineGroup]: ...
    def audit_gate_data(self, staging_dir: Path, manifest: Mapping[str, Any]) -> None: ...
    def release_private_keys(self) -> None: ...


def run_gate_data(config: GateDataPipelineConfig, dependencies: GatePipelineDependencies) -> GateDataResult:
    """Run gate-data and always release privately held key material."""

    try:
        return _run_gate_data_impl(config, dependencies)
    finally:
        release = getattr(dependencies, "release_private_keys", None)
        if callable(release):
            release()


def _run_gate_data_impl(config: GateDataPipelineConfig, dependencies: GatePipelineDependencies) -> GateDataResult:
    if not isinstance(config, GateDataPipelineConfig):
        raise ValueError("config must be GateDataPipelineConfig")
    diagnostic = bool(getattr(dependencies, "diagnostic_test_backend", False))
    source_manifest = dependencies.load_source_manifest(config)
    if not isinstance(source_manifest, Mapping):
        raise ValueError("source manifest must be a mapping")
    source_snapshot = json.loads(_canonical_bytes(dict(source_manifest)))
    if _contains_humaneval(source_snapshot):
        raise ValueError("HumanEval is forbidden from gate data sources")
    private_field = _find_private_manifest_field(source_snapshot)
    if private_field is not None:
        raise ValueError(f"source manifest contains private field: {private_field}")
    source_bytes = _canonical_bytes(source_snapshot)
    if len(source_bytes) > _MAX_METADATA_JSON_BYTES:
        raise ValueError("source manifest exceeds the metadata size limit")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    training_bank = dependencies.load_key_bank(role="training", expected_count=32, config=config)
    holdout_bank = dependencies.load_key_bank(role="holdout", expected_count=8, config=config)
    _validate_bank(training_bank, role="training", expected=32)
    _validate_bank(holdout_bank, role="holdout", expected=8)
    if training_bank.bank_id == holdout_bank.bank_id or set(training_bank.key_ids) & set(holdout_bank.key_ids):
        raise ValueError("training and holdout key banks must be disjoint")
    parsed = dependencies.parse_statement_units(source_snapshot, config)
    pilot_payload = None
    if config.pilot_feasibility_path is not None:
        pilot_payload = _load_passed_pilot(config.pilot_feasibility_path, config.config_hash)
    output = config.output_root / "gate-data"
    staging = _new_staging(config.output_root, "gate-data")
    try:
        data_path = staging / "window_groups.jsonl"
        attempts_path = staging / "candidate_attempts.jsonl"
        labels_path = staging / "labels.jsonl"
        split_manifest_path = staging / "split_manifest.json"
        training_bank_manifest_path = staging / "training_key_bank_manifest.json"
        index_path = staging / "group_index.jsonl"
        writer_paths = (
            (data_path, index_path)
            if config.fast_experimental
            else (data_path, attempts_path, labels_path, index_path)
        )
        writers = [_BoundedJsonlWriter.open(path) for path in writer_paths]
        data_writer = writers[0]
        index_writer = writers[-1]
        attempts_writer = None if config.fast_experimental else writers[1]
        labels_writer = None if config.fast_experimental else writers[2]
        compact_groups: list[_CompactGroupIndex] = []
        feasibility_groups: list[FeasibilityGroup] = []
        seen_group_ids: set[str] = set()
        split_assignments: dict[str, str] = {}
        evidence_cache: OrderedDict[str, None] = OrderedDict()
        close_suitable_counts = Counter(
            {
                "close_false_suitable_false": 0,
                "close_true_suitable_false": 0,
                "close_true_suitable_true": 0,
            }
        )
        statement_family_counts: Counter[str] = Counter()
        rewrite_parse_status_counts: Counter[str] = Counter()
        rewrite_structurally_valid_count = 0
        rewrite_structurally_invalid_count = 0
        rewrite_semantic_signature_stable_count = 0
        rewrite_semantic_signature_unstable_count = 0
        repository_ids: set[str] = set()
        task_ids: set[str] = set()
        try:
            generated_values = dependencies.generate_candidate_trajectories(parsed, config)
            for generated in _iter_groups(generated_values, "generated trajectory"):
                if (
                    config.max_groups is not None
                    and len(compact_groups) >= config.max_groups
                ):
                    break
                if len(compact_groups) >= _MAX_GATE_GROUPS:
                    raise ValueError(f"gate data exceeds the {_MAX_GATE_GROUPS}-group collection limit")
                if generated.group_id in seen_group_ids:
                    raise ValueError("duplicate independent gate groups are forbidden")
                seen_group_ids.add(generated.group_id)
                _validate_trajectory_contract((generated,))
                trajectory = _candidate_stage(generated, config.parser_contract)
                built = _single_group(
                    dependencies.run_multi_key_lsh_probe(
                        (generated,),
                        training_key_ids=training_bank.key_ids,
                        holdout_key_ids=holdout_bank.key_ids,
                        config=config,
                    ),
                    "multi-key LSH probed",
                )
                _validate_same_group_identities((generated,), (built,), "multi-key LSH probe")
                _validate_probe_contract((built,), training_bank, holdout_bank)
                probed = _probed_stage(trajectory, built)
                labeled = _recompute_and_attest_labels(probed, built)
                split_group = _single_group(
                    dependencies.split_groups((built,), config),
                    "split",
                )
                _validate_same_group_identities((built,), (split_group,), "split")
                split_stage = _split_stage(labeled, split_group)
                previous_split = split_assignments.setdefault(
                    split_group.split_group_id, split_group.split,
                )
                if previous_split != split_group.split:
                    raise ValueError("repository/task/function split leakage detected")

                selected = labeled.labels_by_window_length[3]
                close_suitable_counts[
                    (
                        "close_true_suitable_true"
                        if selected.suitable_target
                        else "close_true_suitable_false"
                        if selected.close_target
                        else "close_false_suitable_false"
                    )
                ] += 1
                statement_family_counts[split_group.statement_family] += 1
                repository_ids.add(split_group.repository_id)
                task_ids.add(split_group.task_id)
                for observations in built.candidate_observations_by_length.values():
                    for observation in observations:
                        if observation.candidate_index == 0:
                            continue
                        rewrite_parse_status_counts[observation.parse_status] += 1
                        structurally_valid = (
                            observation.parse_status == "ok"
                            and observation.same_parent_scope
                            and observation.unit_count in {1, 2, 3}
                        )
                        if structurally_valid:
                            rewrite_structurally_valid_count += 1
                        else:
                            rewrite_structurally_invalid_count += 1
                        signature_stable = (
                            observation.lsh_signature is not None
                            and observation.stable_across_precision_modes
                            and observation.stable_across_batch_modes
                        )
                        if signature_stable:
                            rewrite_semantic_signature_stable_count += 1
                        else:
                            rewrite_semantic_signature_unstable_count += 1
                compact = _CompactGroupIndex(
                    group_id=split_group.group_id,
                    identity_sha256=split_stage.identity.digest,
                    split_group_id=split_group.split_group_id,
                    split=split_group.split,
                    close_target=selected.close_target,
                    suitable_target=selected.suitable_target,
                )
                compact_groups.append(compact)
                feasibility_groups.append(_derived_feasibility(labeled, split_group))
                public_row = dict(split_group.row)
                _validate_public_window_row(
                    public_row, diagnostic_expected=diagnostic,
                )
                data_writer.write(public_row)
                if attempts_writer is not None and labels_writer is not None:
                    _write_group_attempts(
                        attempts_writer, split_group, probed, evidence_cache,
                    )
                    labels_writer.write(_label_row(split_group.group_id, labeled))
                index_writer.write(_compact_index_row(compact))
        finally:
            for writer in writers:
                writer.close()
        if not compact_groups:
            raise ValueError("generated trajectory groups must contain GatePipelineGroup values")

        feasibility = evaluate_gate_data_feasibility(tuple(feasibility_groups), scale=config.scale)
        if (
            config.scale == "full"
            and not feasibility.passed
            and not config.fast_experimental
        ):
            raise ValueError("full gate data does not satisfy independent group admission minima")
        split_counts = {
            name: sum(group.split == name for group in compact_groups)
            for name in ("train", "validation", "test")
        }
        split_labels = {
            name: {
                "positive": sum(
                    group.split == name and group.suitable_target
                    for group in compact_groups
                ),
                "negative": sum(
                    group.split == name and not group.suitable_target
                    for group in compact_groups
                ),
            }
            for name in ("train", "validation", "test")
        }
        subset_ids = (
            _deterministic_subsets(compact_groups, config.config_hash)
            if config.scale == "full" else {}
        )
        positive_count = sum(group.suitable_target for group in compact_groups)
        selection_summary_loader = getattr(
            dependencies, "gate_data_selection_summary", None
        )
        selection_summary = (
            selection_summary_loader()
            if callable(selection_summary_loader)
            else None
        )
        if selection_summary is not None:
            if not isinstance(selection_summary, Mapping):
                raise ValueError("gate-data selection summary must be a mapping")
            selection_summary = json.loads(
                _canonical_bytes(dict(selection_summary))
            )
        collection_statistics = {
            "close_suitable_counts": dict(sorted(close_suitable_counts.items())),
            "statement_family_counts": dict(sorted(statement_family_counts.items())),
            "rewrite_parse_status_counts": dict(
                sorted(rewrite_parse_status_counts.items())
            ),
            "rewrite_structurally_valid_count": rewrite_structurally_valid_count,
            "rewrite_structurally_invalid_count": rewrite_structurally_invalid_count,
            "rewrite_semantic_signature_stable_count": (
                rewrite_semantic_signature_stable_count
            ),
            "rewrite_semantic_signature_unstable_count": (
                rewrite_semantic_signature_unstable_count
            ),
            "unique_repository_id_count": len(repository_ids),
            "unique_task_id_count": len(task_ids),
        }
        manifest: dict[str, Any] = {
            "schema_version": GATE_DATA_MANIFEST_VERSION,
            "scale": config.scale,
            "config_hash": config.config_hash,
            "human_eval_included": False,
            "source_manifest_sha256": source_hash,
            "source_manifest": source_snapshot,
            "parser_contract": config.parser_contract,
            "rewriter_config_hash": config.rewriter_config_hash,
            "semantic_encoder_hash": config.semantic_encoder_hash,
            "lsh_config_hash": config.lsh_config_hash,
            "training_key_bank_id": training_bank.bank_id,
            "holdout_key_bank_id": holdout_bank.bank_id,
            "training_key_count": 32,
            "holdout_key_count": 8,
            "rewrite_count": 3,
            "rewrite_budgets": [1, 3],
            "window_lengths": [1, 2, 3],
            "group_count": len(compact_groups),
            "split_counts": split_counts,
            "split_label_counts": split_labels,
            "suitable_positive_group_count": positive_count,
            "suitable_negative_group_count": len(compact_groups) - positive_count,
            "deterministic_group_subset_ids": subset_ids,
            "feasibility_contract": FEASIBILITY_CONTRACT_VERSION,
            "feasibility_thresholds": dict(FEASIBILITY_THRESHOLD_ITEMS),
            "pilot_feasibility_sha256": None if pilot_payload is None else hashlib.sha256(_canonical_bytes(pilot_payload) + b"\n").hexdigest(),
            "diagnostic_test_backend": diagnostic,
            "experimental_only": config.fast_experimental,
            "diagnostic_only": config.fast_experimental,
            "not_official_method": config.fast_experimental,
            "formal_eligible": not diagnostic and not config.fast_experimental,
            "collection_statistics": collection_statistics,
        }
        if selection_summary is not None:
            manifest["selection_summary"] = selection_summary
        if not config.fast_experimental:
            _write_json(split_manifest_path, {
                "schema_version": "wfcllm-gate-split/v1",
                "assignments": {group.group_id: group.split for group in compact_groups},
                "split_groups": {group.group_id: group.split_group_id for group in compact_groups},
            })
            _write_json(training_bank_manifest_path, {
                "schema_version": "wfcllm-training-key-bank-manifest/v1",
                "bank_id": training_bank.bank_id,
                "key_count": len(training_bank.key_ids),
                "key_ids": list(training_bank.key_ids),
            })
        manifest["grouped_jsonl_sha256"] = _sha_file(data_path)
        manifest["group_index_sha256"] = _sha_file(index_path)
        artifact_names = (
            ("window_groups.jsonl", "group_index.jsonl")
            if config.fast_experimental
            else (
                "window_groups.jsonl",
                "candidate_attempts.jsonl",
                "labels.jsonl",
                "split_manifest.json",
                "training_key_bank_manifest.json",
                "group_index.jsonl",
            )
        )
        manifest["artifacts"] = {
            name: _sha_file(staging / name) for name in artifact_names
        }
        feasibility_payload = {**feasibility.to_dict(), "config_hash": config.config_hash, "diagnostic_test_backend": diagnostic}
        _reject_sensitive_public_fields(manifest)
        _reject_sensitive_public_fields(feasibility_payload)
        _write_json(staging / "feasibility_summary.json", feasibility_payload)
        manifest["artifacts"]["feasibility_summary.json"] = _sha_file(
            staging / "feasibility_summary.json"
        )
        _write_json(staging / "manifest.json", manifest)
        if not config.fast_experimental:
            _audit_gate_data_artifacts(staging, manifest, config.config_hash)
            dependencies.audit_gate_data(staging, manifest)
        _publish_new(staging, output)
    except BaseException:
        if os.environ.get("WFCLLM_PRESERVE_FAILED_GATE_DATA") == "1":
            failed = config.output_root / f".gate-data-failed-{os.getpid()}"
            if failed.exists():
                raise ValueError("failed gate-data preservation path already exists")
            staging.replace(failed)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return GateDataResult(output, output / "manifest.json", output / "window_groups.jsonl", output / "feasibility_summary.json", manifest, feasibility, len(compact_groups))


def run_gate_train(config: GateTrainPipelineConfig, dependencies: object) -> GateTrainResult:
    if not isinstance(config, GateTrainPipelineConfig):
        raise ValueError("config must be GateTrainPipelineConfig")
    manifest_path = config.data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("full data manifest is missing")
    manifest = _read_json(manifest_path, "data manifest")
    if manifest.get("schema_version") != GATE_DATA_MANIFEST_VERSION or manifest.get("scale") != "full":
        raise ValueError("data manifest must describe full gate data")
    if manifest.get("feasibility_contract") != FEASIBILITY_CONTRACT_VERSION or manifest.get("feasibility_thresholds") != dict(FEASIBILITY_THRESHOLD_ITEMS):
        raise ValueError("data manifest feasibility contract mismatch")
    if manifest.get("config_hash") != config.config_hash:
        raise ValueError("data manifest config hash mismatch")
    diagnostic = bool(getattr(dependencies, "diagnostic_test_backend", False))
    if config.fast_experimental:
        expected_markers = {
            "experimental_only": True,
            "diagnostic_only": True,
            "not_official_method": True,
            "formal_eligible": False,
        }
        if any(manifest.get(name) is not value for name, value in expected_markers.items()):
            raise ValueError("fast training requires an explicitly experimental data manifest")
    elif (
        manifest.get("diagnostic_test_backend") is True
        or manifest.get("formal_eligible") is not True
    ) and not diagnostic:
        raise ValueError("diagnostic data manifest cannot train a formal candidate")
    if config.pilot_feasibility_path is not None:
        _load_passed_pilot(config.pilot_feasibility_path, config.config_hash)
        if manifest.get("pilot_feasibility_sha256") != _sha_file(config.pilot_feasibility_path):
            raise ValueError("data manifest pilot feasibility hash mismatch")
    elif manifest.get("pilot_feasibility_sha256") is not None:
        raise ValueError("data manifest requires its recorded pilot feasibility input")
    if not config.fast_experimental:
        _enforce_train_minima(manifest)
    if not config.fast_experimental:
        _audit_gate_data_artifacts(config.data_dir, manifest, config.config_hash)
    data_jsonl = config.data_dir / "window_groups.jsonl"
    if not data_jsonl.is_file():
        raise ValueError("grouped gate data JSONL is missing")
    if manifest.get("grouped_jsonl_sha256") != _sha_file(data_jsonl):
        raise ValueError("grouped gate data hash mismatch")
    _audit_training_group_index(config.data_dir, manifest, config.config_hash)
    subsets = manifest.get("deterministic_group_subset_ids")
    if not isinstance(subsets, Mapping) or set(subsets) != {"full"}:
        raise ValueError("data manifest full training subset is missing")
    plan = {"subset_ids": subsets, "status": "planned_not_executed", "unseen_group_metrics": ["close", "suitable"]}
    output = config.output_root / "gate-train"
    staging = _new_staging(config.output_root, "gate-train")
    original_manifest_sha256 = _sha_file(manifest_path)
    try:
        training_data_dir = config.data_dir
        training_manifest = manifest
        snapshot: Path | None = None
        original_data_digest: str | None = None
        if not config.fast_experimental:
            original_data_digest = _tree_hash(config.data_dir)
            snapshot = staging / "_data_snapshot"
            _copy_tree_snapshot(config.data_dir, snapshot)
            training_data_dir = snapshot
            training_manifest = _read_json(
                snapshot / "manifest.json", "snapshot data manifest"
            )
            _audit_gate_data_artifacts(
                snapshot, training_manifest, config.config_hash
            )
        candidate = staging / "candidate_bundle"
        train_result = dependencies.train_candidate(
            config=config,
            data_manifest=training_manifest,
            data_jsonl=training_data_dir / "window_groups.jsonl",
            output_dir=candidate,
            learning_curve_plan=plan,
        )
        if snapshot is not None:
            _audit_gate_data_artifacts(snapshot, training_manifest, config.config_hash)
            if (
                _tree_hash(config.data_dir) != original_data_digest
                or _sha_file(manifest_path) != original_manifest_sha256
            ):
                raise ValueError("trainer mutated original gate-data inputs")
        if not candidate.is_dir() or not any(candidate.iterdir()):
            raise ValueError("trainer did not produce a candidate bundle")
        if not diagnostic and not config.fast_experimental:
            _validate_formal_candidate_tree(candidate)
        elif _has_symlink(candidate):
            raise ValueError("diagnostic candidate bundle cannot contain symlinks")
        training_result = _json_mapping(train_result, "training result")
        _reject_sensitive_public_fields(training_result)
        if len(_canonical_bytes(training_result)) > _MAX_PUBLIC_JSON_BYTES:
            raise ValueError("training result size limit exceeded")
        candidate_manifest = {
            "schema_version": GATE_TRAIN_MANIFEST_VERSION,
            "config_hash": config.config_hash,
            "data_manifest_sha256": original_manifest_sha256,
            "candidate_bundle_sha256": _tree_hash(candidate),
            "training_result": training_result,
            "learning_curve_plan": plan,
            "learning_curve_runs_executed": False,
            "diagnostic_test_backend": diagnostic,
            "experimental_only": config.fast_experimental,
            "diagnostic_only": config.fast_experimental,
            "not_official_method": config.fast_experimental,
            "formal_eligible": not diagnostic and not config.fast_experimental,
        }
        development_summary = {
            "schema_version": "wfcllm-gate-development-summary/v1",
            "config_hash": config.config_hash,
            "data_manifest_sha256": original_manifest_sha256,
            "candidate_bundle_sha256": candidate_manifest["candidate_bundle_sha256"],
            "learning_curve_plan": plan,
            "learning_curve_runs_executed": False,
            "unseen_group_metrics_status": "not_run_pending_formal_experiment_approval",
            "diagnostic_test_backend": diagnostic,
            "formal_eligible": not diagnostic,
        }
        _reject_sensitive_public_fields(candidate_manifest)
        _reject_sensitive_public_fields(development_summary)
        if snapshot is not None:
            shutil.rmtree(snapshot)
        _write_json(staging / "candidate_bundle_manifest.json", candidate_manifest)
        if not config.fast_experimental:
            _write_json(staging / "development_summary.json", development_summary)
        _publish_new(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return GateTrainResult(output, output / "candidate_bundle", output / "candidate_bundle_manifest.json", candidate_manifest)


def run_gate_validate(config: GateValidatePipelineConfig, dependencies: object) -> GateValidateResult:
    if not isinstance(config, GateValidatePipelineConfig):
        raise ValueError("config must be GateValidatePipelineConfig")
    if not config.candidate_bundle.is_dir() or _has_symlink(config.candidate_bundle):
        raise ValueError("candidate bundle is missing or unsafe")
    data_manifest_path = config.data_dir / "manifest.json"
    if not data_manifest_path.is_file():
        raise ValueError("data manifest is missing")
    data_manifest = _read_json(data_manifest_path, "data manifest")
    diagnostic = bool(getattr(dependencies, "diagnostic_test_backend", False))
    threshold_fit_group_ids: tuple[str, ...] = ()
    agreement_group_ids: tuple[str, ...] = ()
    candidate_manifest_path: Path | None = None
    if not diagnostic:
        if (
            data_manifest.get("schema_version") != GATE_DATA_MANIFEST_VERSION
            or data_manifest.get("scale") != "full"
            or data_manifest.get("config_hash") != config.config_hash
            or data_manifest.get("feasibility_contract") != FEASIBILITY_CONTRACT_VERSION
            or data_manifest.get("feasibility_thresholds") != dict(FEASIBILITY_THRESHOLD_ITEMS)
            or data_manifest.get("formal_eligible") is not True
            or data_manifest.get("diagnostic_test_backend") is not False
        ):
            raise ValueError("formal validation requires a matching full data manifest")
        index_rows = _audit_training_group_index(config.data_dir, data_manifest, config.config_hash)
        _audit_gate_data_artifacts(config.data_dir, data_manifest, config.config_hash)
        threshold_fit_group_ids = tuple(row["group_id"] for row in index_rows if row["split"] == "validation")
        agreement_group_ids = tuple(row["group_id"] for row in index_rows if row["split"] == "test")
        if not threshold_fit_group_ids or not agreement_group_ids:
            raise ValueError("formal validation requires non-empty validation and test holdout groups")
        candidate_manifest_path = config.candidate_bundle.parent / "candidate_bundle_manifest.json"
        candidate_manifest = _read_json(candidate_manifest_path, "candidate bundle manifest")
        if (
            candidate_manifest.get("schema_version") != GATE_TRAIN_MANIFEST_VERSION
            or candidate_manifest.get("config_hash") != config.config_hash
            or candidate_manifest.get("formal_eligible") is not True
            or candidate_manifest.get("diagnostic_test_backend") is not False
            or candidate_manifest.get("data_manifest_sha256") != _sha_file(data_manifest_path)
            or candidate_manifest.get("candidate_bundle_sha256") != _tree_hash(config.candidate_bundle)
        ):
            raise ValueError("candidate bundle manifest does not match formal validation inputs")
        _validate_formal_candidate_tree(config.candidate_bundle)
    root = config.output_root / "gate-validate"
    if _has_symlink(config.output_root) or root.is_symlink():
        raise ValueError("gate-validate output path cannot traverse symlinks")
    config.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gate-validate-", dir=config.output_root))
    try:
        callback_candidate = config.candidate_bundle
        original_candidate_digest = _tree_hash(config.candidate_bundle)
        original_data_digest = _tree_hash(config.data_dir)
        original_data_manifest_sha256 = _sha_file(data_manifest_path)
        original_candidate_manifest_sha256 = (
            None if candidate_manifest_path is None else _sha_file(candidate_manifest_path)
        )
        candidate_snapshot: Path | None = None
        if not diagnostic:
            candidate_snapshot = staging / "_candidate_snapshot"
            _copy_tree_snapshot(config.candidate_bundle, candidate_snapshot)
            _validate_formal_candidate_tree(candidate_snapshot)
            callback_candidate = candidate_snapshot
        outcome = dependencies.validate_candidate(
            config=config,
            candidate_bundle=callback_candidate,
            data_manifest=data_manifest,
            threshold_fit_group_ids=threshold_fit_group_ids,
            agreement_group_ids=agreement_group_ids,
            output_dir=staging,
        )
        if not diagnostic:
            assert candidate_snapshot is not None
            _validate_formal_candidate_tree(candidate_snapshot)
            if (
                _tree_hash(candidate_snapshot) != original_candidate_digest
                or _tree_hash(config.candidate_bundle) != original_candidate_digest
                or _tree_hash(config.data_dir) != original_data_digest
                or _sha_file(data_manifest_path) != original_data_manifest_sha256
                or candidate_manifest_path is None
                or _sha_file(candidate_manifest_path) != original_candidate_manifest_sha256
            ):
                raise ValueError("validator mutated formal validation inputs")
        if not isinstance(outcome, ValidationOutcome):
            raise ValueError("validator must return ValidationOutcome")
        if diagnostic:
            if isinstance(outcome.summary, GateValidationSummary):
                raw_summary = outcome.summary.to_dict()
            else:
                raw_summary = dict(outcome.summary)
            summary = {**raw_summary, "diagnostic_test_backend": True}
            summary["validated"] = False
            summary["formal_eligible"] = False
            _reject_sensitive_public_fields(summary)
            root.mkdir(parents=True, exist_ok=True)
            failed = root / "failed_validation_summary.json"
            _write_json_atomic(failed, summary)
            shutil.rmtree(staging, ignore_errors=True)
            return GateValidateResult(False, root, None, None, failed, summary)
        if not isinstance(outcome.summary, GateValidationSummary):
            raise ValueError("formal validator must return GateValidationSummary")
        if outcome.validated != outcome.summary.validated:
            raise ValueError("validator outcome flag contradicts strict validation summary")
        summary = outcome.summary.to_dict()
        if not outcome.summary.validated:
            failed_payload = {**summary, "diagnostic_test_backend": False, "formal_eligible": False}
            root.mkdir(parents=True, exist_ok=True)
            failed = root / "failed_validation_summary.json"
            _write_json_atomic(failed, failed_payload)
            shutil.rmtree(staging, ignore_errors=True)
            return GateValidateResult(False, root, None, None, failed, failed_payload)
        bundle_path = staging / "bundle"
        if not bundle_path.is_dir() or not any(bundle_path.iterdir()) or _has_symlink(bundle_path):
            raise ValueError("validated publisher did not produce a safe bundle directory")
        loaded_bundle = GateBundle.load(bundle_path)
        if dict(loaded_bundle.validation_summary) != summary:
            raise ValueError("staging bundle validation summary differs from strict validator result")
        expected_fit_digest = _validation_group_digest(set(threshold_fit_group_ids))
        expected_agreement_digest = _validation_group_digest(set(agreement_group_ids))
        if (
            summary["threshold_fit_group_digest"] != expected_fit_digest
            or summary["agreement_group_digest"] != expected_agreement_digest
        ):
            raise ValueError("validation summary group digests do not match pipeline holdout groups")
        if (
            loaded_bundle.manifest.training_data_manifest_sha256 != original_data_manifest_sha256
            or loaded_bundle.manifest.training_key_bank_id != data_manifest.get("training_key_bank_id")
            or loaded_bundle.manifest.holdout_key_bank_id != data_manifest.get("holdout_key_bank_id")
        ):
            raise ValueError("staging bundle provenance does not match gate data manifest")
        if (
            loaded_bundle.manifest.float_model_sha256 != _sha_file(callback_candidate / "gate_float.pt")
            or loaded_bundle.manifest.tokenizer_sha256 != sha256_directory(callback_candidate / "tokenizer")
        ):
            raise ValueError("validated bundle model/tokenizer do not match candidate bundle")
        publication_summary = {**summary, "diagnostic_test_backend": False, "formal_eligible": True}
        publication = {
            "schema_version": GATE_VALIDATE_MANIFEST_VERSION,
            "validated": True,
            "config_hash": config.config_hash,
            "candidate_bundle_sha256": original_candidate_digest,
            "data_manifest_sha256": original_data_manifest_sha256,
            "bundle_sha256": _tree_hash(bundle_path),
            "validation_summary": publication_summary,
            "diagnostic_test_backend": False,
        }
        _reject_sensitive_public_fields(publication)
        if candidate_snapshot is not None:
            shutil.rmtree(candidate_snapshot)
        _write_json(staging / "gate_bundle_manifest.json", publication)
        final_manifest = root / "gate_bundle_manifest.json"
        if root.is_symlink():
            raise ValueError("validated bundle publication path is unsafe")
        if root.exists():
            names = {path.name for path in root.iterdir()}
            if names != {"failed_validation_summary.json"}:
                raise ValueError("validated bundle publication already exists")
            backup = config.output_root / f".gate-validate-failed-{os.getpid()}"
            if backup.exists() or backup.is_symlink():
                raise ValueError("validation publication backup already exists")
            os.replace(root, backup)
            try:
                os.replace(staging, root)
            except BaseException:
                if not root.exists():
                    os.replace(backup, root)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, root)
        published_bundle = GateBundle.load(root / "bundle")
        return GateValidateResult(True, root, published_bundle, final_manifest, None, publication_summary)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _iter_groups(values: object, stage: str) -> Iterator[GatePipelineGroup]:
    if isinstance(values, (str, bytes, bytearray, Mapping)) or not isinstance(values, Iterable):
        raise ValueError(f"{stage} groups must be an iterable")
    for group in values:
        if not isinstance(group, GatePipelineGroup):
            raise ValueError(f"{stage} groups must contain GatePipelineGroup values")
        yield group


def _single_group(values: object, stage: str) -> GatePipelineGroup:
    iterator = iter(_iter_groups(values, stage))
    try:
        group = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"{stage} must return exactly one GatePipelineGroup") from exc
    try:
        next(iterator)
    except StopIteration:
        return group
    raise ValueError(f"{stage} must return exactly one GatePipelineGroup")


def _groups(values: object, stage: str) -> tuple[GatePipelineGroup, ...]:
    """Legacy bounded snapshot helper retained for narrow internal callers."""

    result = tuple(_iter_groups(values, stage))
    if not result:
        raise ValueError(f"{stage} groups must contain GatePipelineGroup values")
    return result


def _validate_groups(groups: tuple[GatePipelineGroup, ...]) -> None:
    ids = [group.group_id for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate independent gate groups are forbidden")
    assignments: dict[str, str] = {}
    for group in groups:
        previous = assignments.setdefault(group.split_group_id, group.split)
        if previous != group.split:
            raise ValueError("repository/task/function split leakage detected")


def _validate_same_group_identities(
    before: tuple[GatePipelineGroup, ...],
    after: tuple[GatePipelineGroup, ...],
    stage: str,
) -> None:
    if tuple(group.group_id for group in before) != tuple(group.group_id for group in after):
        raise ValueError(f"{stage} changed independent group identities or order")


def _validate_trajectory_contract(groups: tuple[GatePipelineGroup, ...]) -> None:
    for group in groups:
        if group.window_lengths != (1, 2, 3):
            raise ValueError("generated trajectories must contain W1/W2/W3 in order")
        if set(group.candidate_indices_by_window_length) != {1, 2, 3} or any(
            group.candidate_indices_by_window_length[length] != tuple(range(4))
            for length in (1, 2, 3)
        ):
            raise ValueError("generated trajectories must contain candidate 0 through 3 for every window")


def _group_identity(group: GatePipelineGroup) -> GateGroupIdentity:
    return GateGroupIdentity(
        group_id=group.group_id,
        split_group_id=group.split_group_id,
        repository_id=group.repository_id,
        task_id=group.task_id,
        generation_model_id=group.generation_model_id,
        statement_family=group.statement_family,
        window_lengths=group.window_lengths,
    )


def _candidate_stage(group: GatePipelineGroup, parser_contract: str) -> CandidateTrajectoryGroup:
    return CandidateTrajectoryGroup(
        parsed=ParsedWindowGroup(_group_identity(group), parser_contract),
        candidate_indices_by_window_length=group.candidate_indices_by_window_length,
    )


def _probed_stage(
    trajectory: CandidateTrajectoryGroup,
    group: GatePipelineGroup,
) -> ProbedGroup:
    if trajectory.identity.digest != _group_identity(group).digest:
        raise ValueError("probe stage changed immutable base identity")
    return ProbedGroup(
        trajectory=trajectory,
        candidate_observations_by_length=group.candidate_observations_by_length,
        probe_results_by_length=group.probe_results_by_length,
    )


def _split_stage(labeled: LabeledGroup, group: GatePipelineGroup) -> SplitGroup:
    if labeled.identity.digest != _group_identity(group).digest:
        raise ValueError("split stage changed immutable base identity")
    selected = labeled.labels_by_window_length[3]
    if (
        selected.close_target != group.close_target
        or selected.suitable_target != group.suitable_target
    ):
        raise ValueError("split stage changed causal labels")
    return SplitGroup(labeled=labeled, split=group.split, row=group.row)


def _derived_feasibility(
    labeled: LabeledGroup,
    group: GatePipelineGroup,
) -> FeasibilityGroup:
    selected = labeled.labels_by_window_length[3]
    first_hits = [
        value
        for value in selected.budgets[3].first_hit_by_key_id.values()
        if value is not None
    ]
    return FeasibilityGroup(
        group_id=group.group_id,
        suitable_target=selected.suitable_target,
        window_lengths=group.window_lengths,
        statement_family=group.statement_family,
        r1_success_rate=selected.budgets[1].success_rate,
        r3_success_rate=selected.budgets[3].success_rate,
        holdout_success_rate=labeled.holdout_success_rate,
        split=group.split,
        repository_id=group.repository_id,
        task_id=group.task_id,
        generation_model_id=group.generation_model_id,
        structural_invalid_rate=1.0 - selected.budgets[3].structural_valid_rate,
        numeric_instability_rate=selected.budgets[3].unstable_rate,
        first_hit_candidate_position=min(first_hits) if first_hits else None,
    )


def _validate_probe_contract(
    groups: tuple[GatePipelineGroup, ...],
    training_bank: KeyBankSnapshot,
    holdout_bank: KeyBankSnapshot,
) -> None:
    for group in groups:
        if group.observed_training_key_ids != training_bank.key_ids or group.observed_holdout_key_ids != holdout_bank.key_ids:
            raise ValueError("multi-key LSH probe evidence does not cover the exact 32/8 key banks")
        if set(group.candidate_observations_by_length) != {"1", "2", "3"} or set(group.probe_results_by_length) != {"1", "2", "3"}:
            raise ValueError("probe evidence must cover W1/W2/W3")
        expected_keys = set(training_bank.key_ids) | set(holdout_bank.key_ids)
        for length in (1, 2, 3):
            observations = group.candidate_observations_by_length[str(length)]
            probes = group.probe_results_by_length[str(length)]
            if len(observations) != 4 or len(probes) != 4:
                raise ValueError("probe evidence must cover candidate 0 through 3")
            for candidate_index, (observation, results) in enumerate(zip(observations, probes, strict=True)):
                if not isinstance(observation, CandidateObservation) or observation.candidate_index != candidate_index:
                    raise ValueError("candidate observation identity/order mismatch")
                exact_structural_candidate = (
                    observation.parse_status == "ok"
                    and observation.same_parent_scope
                    and observation.unit_count == length
                )
                if observation.parse_status == "ok" and not exact_structural_candidate:
                    raise ValueError(
                        "ok candidate parser facts contradict requested window length/scope"
                    )
                if not isinstance(results, Mapping) or any(
                    not isinstance(result, LshProbeResult)
                    for result in results.values()
                ):
                    raise ValueError("probe evidence contains invalid LshProbeResult values")
                if not exact_structural_candidate:
                    if results or observation.lsh_by_key_id or observation.lsh_signature is not None:
                        raise ValueError(
                            "parser-invalid candidate must not contain semantic probe evidence"
                        )
                    if (
                        observation.stable_across_precision_modes
                        or observation.stable_across_batch_modes
                    ):
                        raise ValueError(
                            "parser-invalid candidate must not claim semantic stability"
                        )
                    continue
                if set(results) != expected_keys:
                    raise ValueError(
                        "structurally valid probe evidence must cover exact 32/8 key banks"
                    )
                training_results = {key: results[key] for key in training_bank.key_ids}
                if set(observation.lsh_by_key_id) != set(training_bank.key_ids):
                    raise ValueError("CandidateObservation training evidence key coverage mismatch")
                expected_precision = all(result.stable_across_precision_modes for result in training_results.values())
                expected_batch = all(result.stable_across_batch_modes for result in training_results.values())
                signatures = {result.signature for result in training_results.values()}
                if len(signatures) != 1 or observation.lsh_signature != next(iter(signatures)):
                    raise ValueError("candidate signature contradicts raw probe evidence")
                if observation.stable_across_precision_modes != expected_precision or observation.stable_across_batch_modes != expected_batch:
                    raise ValueError("candidate stability facts contradict raw probe evidence")
                for key, result in training_results.items():
                    raw = observation.lsh_by_key_id[key]
                    if raw != {"hit": result.hit, "stable": result.stable, "margin": result.margin}:
                        raise ValueError("CandidateObservation evidence contradicts LshProbeResult")


def _labels_for_group(group: GatePipelineGroup, length: int) -> GateLabels:
    candidates = group.candidate_observations_by_length[str(length)]
    cache_key = hashlib.sha256(
        _canonical_bytes([candidate.to_dict() for candidate in candidates])
    ).hexdigest()
    cached = _LABEL_CACHE.get(cache_key)
    if cached is not None:
        _LABEL_CACHE.move_to_end(cache_key)
        return cached
    labels = build_gate_labels(candidates, training_key_count=32)
    _LABEL_CACHE[cache_key] = labels
    if len(_LABEL_CACHE) > _EVIDENCE_LRU_SIZE:
        _LABEL_CACHE.popitem(last=False)
    return labels


def _recompute_and_attest_labels(probed: ProbedGroup, group: GatePipelineGroup) -> LabeledGroup:
    if probed.identity.digest != _group_identity(group).digest:
        raise ValueError("label stage changed immutable base identity")
    labels = {length: _labels_for_group(group, length) for length in (1, 2, 3)}
    selected = labels[3]
    r1, r3 = (selected.budgets[budget].success_rate for budget in (1, 3))
    first_hits = [value for value in selected.budgets[3].first_hit_by_key_id.values() if value is not None]
    holdout_ids = group.observed_holdout_key_ids
    holdout_results = group.probe_results_by_length["3"]
    holdout_hit_count = sum(
        any(
            key_id in holdout_results[candidate_index]
            and holdout_results[candidate_index][key_id].is_reliable_hit(
                configured_margin=0.0
            )
            for candidate_index in range(4)
        )
        for key_id in holdout_ids
    )
    holdout_rate = holdout_hit_count / len(holdout_ids)
    structural_invalid_rate = 1.0 - selected.budgets[3].structural_valid_rate
    numeric_instability_rate = selected.budgets[3].unstable_rate
    if (
        group.close_target != selected.close_target
        or group.suitable_target != selected.suitable_target
        or (group.r1_success_rate, group.r3_success_rate) != (r1, r3)
        or group.holdout_success_rate != holdout_rate
        or group.structural_invalid_rate != structural_invalid_rate
        or group.numeric_instability_rate != numeric_instability_rate
        or group.first_hit_candidate_position != (min(first_hits) if first_hits else None)
    ):
        raise ValueError("injected labels/metrics contradict causal labels recomputed from probe evidence")
    return LabeledGroup(
        probed=probed,
        labels_by_window_length=labels,
        holdout_success_rate=holdout_rate,
        first_hit_candidate_position=min(first_hits) if first_hits else None,
    )


def _probe_evidence_payload(
    observation: CandidateObservation,
    results: Mapping[str, LshProbeResult],
) -> dict[str, Any]:
    if len(observation.code.encode("utf-8")) > _MAX_CANDIDATE_CODE_BYTES:
        raise ValueError(
            f"candidate code exceeds {_MAX_CANDIDATE_CODE_BYTES} bytes"
        )
    return {
        "candidate_observation": observation.to_dict(),
        "lsh_probe_results": {
            key: {
            "signature": list(result.signature), "margin": result.margin,
            "hit": result.hit, "stable": result.stable,
            "stable_across_precision_modes": result.stable_across_precision_modes,
            "stable_across_batch_modes": result.stable_across_batch_modes,
            }
            for key, result in sorted(results.items())
        },
    }


def _write_group_attempts(
    writer: _BoundedJsonlWriter,
    group: GatePipelineGroup,
    probed: ProbedGroup,
    evidence_cache: OrderedDict[str, None],
) -> None:
    """Write one complete trajectory while retaining only content digests."""

    for length in (1, 2, 3):
        observations = probed.candidate_observations_by_length[str(length)]
        results_by_candidate = probed.probe_results_by_length[str(length)]
        for candidate_index, (observation, results) in enumerate(
            zip(observations, results_by_candidate, strict=True)
        ):
            evidence = _probe_evidence_payload(observation, results)
            evidence_sha256 = hashlib.sha256(_canonical_bytes(evidence)).hexdigest()
            if evidence_sha256 in evidence_cache:
                evidence_cache.move_to_end(evidence_sha256)
            else:
                writer.write({
                    "schema_version": "wfcllm-gate-candidate-attempts/v2",
                    "record_type": "probe_evidence",
                    "evidence_sha256": evidence_sha256,
                    **evidence,
                })
                evidence_cache[evidence_sha256] = None
                if len(evidence_cache) > _EVIDENCE_LRU_SIZE:
                    evidence_cache.popitem(last=False)
            writer.write({
                "schema_version": "wfcllm-gate-candidate-attempts/v2",
                "record_type": "candidate_attempt",
                "group_id": group.group_id,
                "identity_sha256": probed.identity.digest,
                "window_length": length,
                "candidate_index": candidate_index,
                "evidence_sha256": evidence_sha256,
            })


def _label_row(group_id: str, stage: LabeledGroup) -> dict[str, Any]:
    selected = stage.labels_by_window_length[3]
    return {
        "schema_version": "wfcllm-gate-label/v1",
        "group_id": group_id,
        "close_target": selected.close_target,
        "suitable_target": selected.suitable_target,
        "r1_success_rate": selected.budgets[1].success_rate,
        "r3_success_rate": selected.budgets[3].success_rate,
        "holdout_success_rate": stage.holdout_success_rate,
        "budget_outcomes": {
            str(length): {
                str(budget): outcome.to_dict()
                for budget, outcome in stage.labels_by_window_length[length].budgets.items()
            }
            for length in (1, 2, 3)
        },
    }


def _compact_index_row(group: _CompactGroupIndex) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "identity_sha256": group.identity_sha256,
        "split_group_id": group.split_group_id,
        "split": group.split,
        "close_target": group.close_target,
        "suitable_target": group.suitable_target,
    }


def _validate_bank(bank: object, *, role: str, expected: int) -> None:
    if not isinstance(bank, KeyBankSnapshot) or len(bank.key_ids) != expected or not bank.bank_id.startswith(f"{role}-key-bank/v1:sha256:"):
        raise ValueError(f"{role} key bank must contain exactly {expected} keys")


def _load_passed_pilot(path: Path | None, config_hash: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise ValueError("passed pilot feasibility summary is missing")
    payload = _read_json(path, "pilot feasibility")
    if payload.get("contract_version") != FEASIBILITY_CONTRACT_VERSION or payload.get("scale") != "pilot" or payload.get("passed") is not True:
        raise ValueError("pilot feasibility contract has not passed")
    if payload.get("thresholds") != dict(FEASIBILITY_THRESHOLD_ITEMS):
        raise ValueError("pilot feasibility thresholds do not match the v1 contract")
    if payload.get("config_hash") != config_hash:
        raise ValueError("pilot feasibility config hash mismatch")
    return payload


def _enforce_train_minima(manifest: Mapping[str, Any]) -> None:
    thresholds = dict(FEASIBILITY_THRESHOLD_ITEMS)
    checks = {
        "independent groups": (
            manifest.get("group_count"),
            int(thresholds["full_independent_group_min"]),
        ),
        "suitable positive groups": (
            manifest.get("suitable_positive_group_count"),
            int(thresholds["full_suitable_positive_min"]),
        ),
        "suitable negative groups": (
            manifest.get("suitable_negative_group_count"),
            int(thresholds["full_suitable_negative_min"]),
        ),
    }
    split = manifest.get("split_label_counts")
    if not isinstance(split, Mapping):
        raise ValueError("data manifest split label counts are missing")
    for name in ("validation", "test"):
        value = split.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"data manifest {name} counts are missing")
        checks[f"{name} positive groups"] = (
            value.get("positive"),
            int(thresholds["validation_test_suitable_positive_min"]),
        )
        checks[f"{name} negative groups"] = (
            value.get("negative"),
            int(thresholds["validation_test_suitable_negative_min"]),
        )
    failed = [f"{name}={observed!r} < {minimum}" for name, (observed, minimum) in checks.items() if type(observed) is not int or observed < minimum]
    if failed:
        raise ValueError("gate-train independent group minima failed: " + "; ".join(failed))


def _audit_training_group_index(data_dir: Path, manifest: Mapping[str, Any], seed: str) -> tuple[dict[str, Any], ...]:
    path = data_dir / "group_index.jsonl"
    if not path.is_file() or _has_symlink(path) or manifest.get("group_index_sha256") != _sha_file(path):
        raise ValueError("gate data group index hash mismatch")
    rows: list[dict[str, Any]] = []
    try:
        for value in _iter_jsonl(path, "group index"):
            if set(value) != {"group_id", "identity_sha256", "split_group_id", "split", "close_target", "suitable_target"}:
                raise ValueError("group index row schema mismatch")
            if not all(isinstance(value[name], str) and value[name] for name in ("group_id", "split_group_id")):
                raise ValueError("group index identity is invalid")
            if not isinstance(value["identity_sha256"], str) or _DIGEST.fullmatch(value["identity_sha256"]) is None:
                raise ValueError("group index identity digest is invalid")
            if value["split"] not in {"train", "validation", "test"}:
                raise ValueError("group index split is invalid")
            if type(value["close_target"]) is not bool or type(value["suitable_target"]) is not bool or (value["suitable_target"] and not value["close_target"]):
                raise ValueError("group index labels are invalid")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("gate data group index is invalid") from exc
    ids = [row["group_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(rows) != manifest.get("group_count"):
        raise ValueError("gate data group index independent count mismatch")
    split_assignments: dict[str, str] = {}
    for row in rows:
        previous = split_assignments.setdefault(row["split_group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("gate data group index split leakage")
    split_counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    split_labels = {
        name: {
            "positive": sum(row["split"] == name and row["suitable_target"] for row in rows),
            "negative": sum(row["split"] == name and not row["suitable_target"] for row in rows),
        }
        for name in ("train", "validation", "test")
    }
    if manifest.get("split_counts") != split_counts or manifest.get("split_label_counts") != split_labels:
        raise ValueError("gate data group index counts contradict manifest")
    if manifest.get("suitable_positive_group_count") != sum(row["suitable_target"] for row in rows):
        raise ValueError("gate data group index positive count contradicts manifest")
    ordered = sorted(ids, key=lambda group_id: (hashlib.sha256((seed + "\0" + group_id).encode()).hexdigest(), group_id))
    expected = (
        {"full": ordered}
        if manifest.get("scale") == "full"
        else {}
    )
    if manifest.get("deterministic_group_subset_ids") != expected:
        raise ValueError("gate data deterministic subset IDs contradict group index")
    return tuple(rows)


def _audit_gate_data_artifacts(data_dir: Path, manifest: Mapping[str, Any], seed: str) -> None:
    artifact_names = {
        "window_groups.jsonl", "candidate_attempts.jsonl", "labels.jsonl",
        "split_manifest.json", "training_key_bank_manifest.json",
        "group_index.jsonl", "feasibility_summary.json",
    }
    if (
        _has_symlink(data_dir)
        or not data_dir.is_dir()
        or {path.name for path in data_dir.iterdir()} != artifact_names | {"manifest.json"}
    ):
        raise ValueError("gate data artifact root allowlist mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_names:
        raise ValueError("gate data manifest artifact allowlist mismatch")
    for name in artifact_names:
        path = data_dir / name
        if _has_symlink(path) or not path.is_file() or artifacts[name] != _sha_file(path):
            raise ValueError(f"gate data artifact missing, unsafe, or hash-mismatched: {name}")
    index_rows = _audit_training_group_index(data_dir, manifest, seed)
    ids = [row["group_id"] for row in index_rows]
    window_rows = _iter_jsonl(data_dir / "window_groups.jsonl", "window groups")
    diagnostic_expected = manifest.get("diagnostic_test_backend") is True
    for expected_index in index_rows:
        try:
            row = next(window_rows)
        except StopIteration as exc:
            raise ValueError("window group artifact is incomplete") from exc
        _validate_public_window_row(row, diagnostic_expected=diagnostic_expected)
        if row.get("group_id") != expected_index["group_id"] or row.get("split") != expected_index["split"]:
            raise ValueError("window group identity/split contradicts group index")
    try:
        next(window_rows)
    except StopIteration:
        pass
    else:
        raise ValueError("window group artifact contains extra rows")
    indexed = {row["group_id"]: row for row in index_rows}
    training_ids = [f"train-key-{index:03d}" for index in range(32)]
    holdout_ids = [f"holdout-key-{index:03d}" for index in range(8)]
    expected_label_digests = _audit_candidate_attempts(
        data_dir / "candidate_attempts.jsonl",
        index_rows=index_rows,
        training_ids=training_ids,
        holdout_ids=holdout_ids,
    )
    _audit_labels(
        data_dir / "labels.jsonl",
        index_rows=index_rows,
        expected_digests=expected_label_digests,
    )
    split = _read_json(data_dir / "split_manifest.json", "split manifest")
    if (
        split.get("assignments") != {row["group_id"]: row["split"] for row in index_rows}
        or split.get("split_groups") != {row["group_id"]: row["split_group_id"] for row in index_rows}
    ):
        raise ValueError("split manifest contradicts group index")
    bank = _read_json(data_dir / "training_key_bank_manifest.json", "training key bank manifest")
    if bank.get("bank_id") != manifest.get("training_key_bank_id") or bank.get("key_count") != 32 or bank.get("key_ids") != training_ids:
        raise ValueError("training key bank provenance mismatch")
    feasibility = _read_json(data_dir / "feasibility_summary.json", "feasibility summary")
    if (
        feasibility.get("contract_version") != FEASIBILITY_CONTRACT_VERSION
        or feasibility.get("thresholds") != dict(FEASIBILITY_THRESHOLD_ITEMS)
        or feasibility.get("independent_group_count") != manifest.get("group_count")
        or feasibility.get("config_hash") != manifest.get("config_hash")
    ):
        raise ValueError("feasibility summary contradicts data manifest")


def _audit_candidate_attempts(
    path: Path,
    *,
    index_rows: tuple[dict[str, Any], ...],
    training_ids: list[str],
    holdout_ids: list[str],
) -> dict[str, str]:
    """Stream the potentially very large attempt artifact without materializing it."""

    expected = (
        (row["group_id"], row["identity_sha256"], length, candidate_index)
        for row in index_rows
        for length in (1, 2, 3)
        for candidate_index in range(4)
    )
    expected_keys = set(training_ids) | set(holdout_ids)
    evidence_cache: OrderedDict[str, tuple[CandidateObservation, dict[str, LshProbeResult]]] = OrderedDict()
    label_cache: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()
    expected_label_digests: dict[str, str] = {}
    current_evidence_hashes: list[str] = []
    current_observations: dict[int, list[CandidateObservation]] = {1: [], 2: [], 3: []}
    current_results: dict[int, list[dict[str, LshProbeResult]]] = {1: [], 2: [], 3: []}
    pending_evidence: str | None = None
    candidate_count = 0
    try:
        for row in _iter_jsonl(path, "candidate attempts"):
            record_type = row.get("record_type")
            if record_type == "probe_evidence":
                if pending_evidence is not None:
                    raise ValueError("probe evidence record is not followed by its candidate attempt")
                if set(row) != {
                    "schema_version", "record_type", "evidence_sha256",
                    "candidate_observation", "lsh_probe_results",
                } or row.get("schema_version") != "wfcllm-gate-candidate-attempts/v2":
                    raise ValueError("probe evidence row schema mismatch")
                digest = row["evidence_sha256"]
                if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
                    raise ValueError("probe evidence digest is invalid")
                payload = {
                    "candidate_observation": row["candidate_observation"],
                    "lsh_probe_results": row["lsh_probe_results"],
                }
                if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != digest:
                    raise ValueError("probe evidence content digest mismatch")
                observation, results = _deserialize_probe_evidence(payload)
                if set(results) not in (set(), expected_keys):
                    raise ValueError(
                        "probe evidence must be empty or cover exact 32/8 key banks"
                    )
                _validate_observation_against_results(observation, results, tuple(training_ids))
                evidence_cache[digest] = (observation, results)
                evidence_cache.move_to_end(digest)
                if len(evidence_cache) > 512:
                    evidence_cache.popitem(last=False)
                pending_evidence = digest
                continue
            if record_type != "candidate_attempt" or set(row) != {
                "schema_version", "record_type", "group_id", "identity_sha256",
                "window_length", "candidate_index", "evidence_sha256",
            } or row.get("schema_version") != "wfcllm-gate-candidate-attempts/v2":
                raise ValueError("candidate attempt row schema mismatch")
            try:
                expected_value = next(expected)
            except StopIteration as exc:
                raise ValueError("candidate attempt artifact contains extra trajectories") from exc
            observed = (
                row.get("group_id"), row.get("identity_sha256"),
                row.get("window_length"), row.get("candidate_index"),
            )
            if observed != expected_value:
                raise ValueError("candidate attempt group/window/candidate order mismatch")
            digest = row.get("evidence_sha256")
            if pending_evidence is not None and digest != pending_evidence:
                raise ValueError("probe evidence record is not bound to the following candidate attempt")
            evidence_value = evidence_cache.get(digest)
            if evidence_value is None:
                raise ValueError("candidate attempt references absent or expired probe evidence")
            evidence_cache.move_to_end(digest)
            observation, results = evidence_value
            if observation.candidate_index != row["candidate_index"] or (
                bool(results) and observation.unit_count != row["window_length"]
            ):
                raise ValueError("candidate attempt identity contradicts serialized observation")
            length = row["window_length"]
            current_evidence_hashes.append(digest)
            current_observations[length].append(observation)
            current_results[length].append(results)
            pending_evidence = None
            candidate_count += 1
            if length == 3 and row["candidate_index"] == 3:
                group_id = row["group_id"]
                cache_key = tuple(current_evidence_hashes)
                label_payload = label_cache.get(cache_key)
                if label_payload is None:
                    label_payload = _derived_label_payload(
                        current_observations,
                        current_results,
                        holdout_ids=tuple(holdout_ids),
                    )
                    label_cache[cache_key] = label_payload
                    if len(label_cache) > 512:
                        label_cache.popitem(last=False)
                else:
                    label_cache.move_to_end(cache_key)
                expected_row = {
                    "schema_version": "wfcllm-gate-label/v1",
                    "group_id": group_id,
                    **label_payload,
                }
                indexed = index_rows[len(expected_label_digests)]
                if (
                    indexed["group_id"] != group_id
                    or indexed["close_target"] != expected_row["close_target"]
                    or indexed["suitable_target"] != expected_row["suitable_target"]
                ):
                    raise ValueError("causal labels recomputed from probe evidence contradict group index")
                expected_label_digests[group_id] = hashlib.sha256(_canonical_bytes(expected_row)).hexdigest()
                current_evidence_hashes = []
                current_observations = {1: [], 2: [], 3: []}
                current_results = {1: [], 2: [], 3: []}
        try:
            next(expected)
        except StopIteration:
            pass
        else:
            raise ValueError("candidate attempt artifact is incomplete")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"candidate attempts JSONL is invalid: {exc}") from exc
    if pending_evidence is not None or candidate_count != len(index_rows) * 12:
        raise ValueError("candidate trajectory/probe evidence coverage mismatch")
    return expected_label_digests


def _derived_label_payload(
    observations: Mapping[int, list[CandidateObservation]],
    results: Mapping[int, list[dict[str, LshProbeResult]]],
    *,
    holdout_ids: tuple[str, ...],
) -> dict[str, Any]:
    labels = {
        length: build_gate_labels(tuple(observations[length]), training_key_count=32)
        for length in (1, 2, 3)
    }
    selected = labels[3]
    holdout_success_rate = sum(
        any(
            (probe := results[3][candidate_index].get(key_id)) is not None
            and probe.is_reliable_hit(configured_margin=0.0)
            for candidate_index in range(4)
        )
        for key_id in holdout_ids
    ) / len(holdout_ids)
    return {
        "close_target": selected.close_target,
        "suitable_target": selected.suitable_target,
        "r1_success_rate": selected.budgets[1].success_rate,
        "r3_success_rate": selected.budgets[3].success_rate,
        "holdout_success_rate": holdout_success_rate,
        "budget_outcomes": {
            str(length): {
                str(budget): outcome.to_dict()
                for budget, outcome in labels[length].budgets.items()
            }
            for length in (1, 2, 3)
        },
    }


def _audit_labels(
    path: Path,
    *,
    index_rows: tuple[dict[str, Any], ...],
    expected_digests: Mapping[str, str],
) -> None:
    expected_ids = iter(row["group_id"] for row in index_rows)
    count = 0
    try:
        for row in _iter_jsonl(path, "labels"):
            try:
                group_id = next(expected_ids)
            except StopIteration as exc:
                raise ValueError("label artifact contains extra rows") from exc
            if row.get("group_id") != group_id:
                raise ValueError("label group IDs/order contradict group index")
            if hashlib.sha256(_canonical_bytes(row)).hexdigest() != expected_digests.get(group_id):
                raise ValueError("labels contradict causal Task6 outcomes recomputed from probe evidence")
            count += 1
        try:
            next(expected_ids)
        except StopIteration:
            pass
        else:
            raise ValueError("label artifact is incomplete")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("labels JSONL is invalid") from exc
    if count != len(index_rows):
        raise ValueError("label artifact group count mismatch")


def _deserialize_probe_evidence(
    payload: Mapping[str, Any],
) -> tuple[CandidateObservation, dict[str, LshProbeResult]]:
    raw_observation = payload.get("candidate_observation")
    raw_results = payload.get("lsh_probe_results")
    if not isinstance(raw_observation, Mapping) or not isinstance(raw_results, Mapping):
        raise ValueError("serialized probe evidence must contain mappings")
    observation_fields = dict(raw_observation)
    if observation_fields.pop("schema_version", None) != "wfcllm-gate-data/v1":
        raise ValueError("candidate observation schema version mismatch")
    observation_fields["boundary_span"] = tuple(observation_fields.get("boundary_span", ()))
    signature = observation_fields.get("lsh_signature")
    observation_fields["lsh_signature"] = None if signature is None else tuple(signature)
    code = observation_fields.get("code")
    if not isinstance(code, str) or len(code.encode("utf-8")) > _MAX_CANDIDATE_CODE_BYTES:
        raise ValueError("serialized candidate code exceeds the size limit")
    observation = CandidateObservation(**observation_fields)
    results: dict[str, LshProbeResult] = {}
    for key_id, value in raw_results.items():
        if not isinstance(key_id, str) or not isinstance(value, Mapping) or set(value) != {
            "signature", "margin", "hit", "stable",
            "stable_across_precision_modes", "stable_across_batch_modes",
        }:
            raise ValueError("serialized LSH probe result schema mismatch")
        results[key_id] = LshProbeResult(
            signature=tuple(value["signature"]),
            margin=value["margin"],
            hit=value["hit"],
            stable=value["stable"],
            stable_across_precision_modes=value["stable_across_precision_modes"],
            stable_across_batch_modes=value["stable_across_batch_modes"],
        )
    return observation, results


def _validate_observation_against_results(
    observation: CandidateObservation,
    results: Mapping[str, LshProbeResult],
    training_ids: tuple[str, ...],
) -> None:
    if not results:
        if (
            observation.lsh_by_key_id
            or observation.lsh_signature is not None
            or observation.stable_across_precision_modes
            or observation.stable_across_batch_modes
        ):
            raise ValueError(
                "evidence-free candidate contradicts serialized semantic observation"
            )
        if (
            observation.parse_status == "ok"
            and observation.same_parent_scope
            and observation.unit_count in {1, 2, 3}
        ):
            raise ValueError("structurally usable candidate is missing semantic evidence")
        return
    training_results = {key: results[key] for key in training_ids}
    if set(observation.lsh_by_key_id) != set(training_ids):
        raise ValueError("CandidateObservation training evidence key coverage mismatch")
    signatures = {result.signature for result in training_results.values()}
    if len(signatures) != 1 or observation.lsh_signature != next(iter(signatures)):
        raise ValueError("candidate signature contradicts raw probe evidence")
    if observation.stable_across_precision_modes != all(
        result.stable_across_precision_modes for result in training_results.values()
    ) or observation.stable_across_batch_modes != all(
        result.stable_across_batch_modes for result in training_results.values()
    ):
        raise ValueError("candidate stability facts contradict raw probe evidence")
    for key, result in training_results.items():
        if observation.lsh_by_key_id[key] != {
            "hit": result.hit, "stable": result.stable, "margin": result.margin,
        }:
            raise ValueError("CandidateObservation evidence contradicts LshProbeResult")


def _iter_jsonl(path: Path, name: str) -> Iterator[dict[str, Any]]:
    if _has_symlink(path) or not path.is_file():
        raise ValueError(f"{name} is missing or unsafe")
    if path.stat().st_size > _MAX_GATE_ARTIFACT_BYTES:
        raise ValueError(f"{name} exceeds the gate artifact size limit")
    with path.open("rb") as handle:
        while line := handle.readline(_MAX_JSONL_LINE_BYTES + 1):
            if len(line) > _MAX_JSONL_LINE_BYTES or not line.endswith(b"\n"):
                raise ValueError(f"{name} contains an oversized or unterminated row")
            try:
                text = line.decode("utf-8")
                row = json.loads(
                    text,
                    object_pairs_hook=_unique_pairs,
                    parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                )
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{name} contains invalid JSON") from exc
            if not isinstance(row, dict) or text != _canonical_text(row):
                raise ValueError(f"{name} contains a non-canonical row")
            yield row


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        rows.extend(_iter_jsonl(path, name))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} JSONL is invalid") from exc
    return rows


def _deterministic_subsets(groups: Sequence[Any], seed: str) -> dict[str, list[str]]:
    ordered = sorted(groups, key=lambda group: (hashlib.sha256((seed + "\0" + group.group_id).encode()).hexdigest(), group.group_id))
    return {"full": [group.group_id for group in ordered]}


def _validation_group_digest(groups: set[str]) -> str:
    payload = json.dumps(sorted(groups), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_humaneval(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_humaneval(key) or _contains_humaneval(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_humaneval(item) for item in value)
    if isinstance(value, str):
        normalized = "".join(character for character in value.casefold() if character.isalnum())
        return "humaneval" in normalized
    return False


def _find_private_manifest_field(value: object, *, path: str = "source_manifest") -> str | None:
    forbidden = {"code", "rawcode", "key", "keys", "keymaterial", "material", "secret", "secretkey", "privatekey"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(character for character in str(key).casefold() if character.isalnum())
            if normalized in forbidden:
                return f"{path}.{key}"
            found = _find_private_manifest_field(item, path=f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_private_manifest_field(item, path=f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _reject_sensitive_public_fields(value: object, *, path: str = "artifact") -> None:
    forbidden = {
        "key",
        "keys",
        "keymaterial",
        "material",
        "secret",
        "secretkey",
        "privatekey",
        "apikey",
        "accesstoken",
        "rawcode",
        "rawkey",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(character for character in str(key).casefold() if character.isalnum())
            if normalized in forbidden:
                raise ValueError(f"public artifact contains sensitive field: {path}.{key}")
            _reject_sensitive_public_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_public_fields(item, path=f"{path}[{index}]")


def _validate_public_window_row(
    row: Mapping[str, Any], *, diagnostic_expected: bool | None,
) -> None:
    base = {"schema_version", "group_id", "split"}
    diagnostic = base | {"diagnostic_test_backend"}
    keys = set(row)
    if diagnostic_expected is True:
        valid = keys == diagnostic and row.get("diagnostic_test_backend") is True
    elif diagnostic_expected is False:
        valid = keys == base
    else:
        valid = keys == base or (
            keys == diagnostic and row.get("diagnostic_test_backend") is True
        )
    if (
        not valid
        or row.get("schema_version") != "wfcllm-gate-data/v1"
        or not isinstance(row.get("group_id"), str)
        or not row.get("group_id")
        or row.get("split") not in {"train", "validation", "test"}
    ):
        raise ValueError("window group public row schema mismatch")
    _reject_sensitive_public_fields(row, path="window_group")
    if len(_canonical_bytes(dict(row))) + 1 > _MAX_JSONL_LINE_BYTES:
        raise ValueError("window group public row size limit exceeded")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("artifact must be canonical JSON data") from exc


def _canonical_text(value: object) -> str:
    return _canonical_bytes(value).decode("utf-8") + "\n"


def _write_json(path: Path, value: object) -> None:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > _MAX_METADATA_JSON_BYTES:
        raise ValueError(f"metadata artifact exceeds {_MAX_METADATA_JSON_BYTES} bytes: {path.name}")
    with path.open("wb") as handle:
        handle.write(payload)


def _write_json_atomic(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temp.exists():
        raise ValueError("atomic temporary artifact already exists")
    try:
        _write_json(temp, value)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if _has_symlink(path) or not path.is_file():
        raise ValueError(f"{name} is missing or unsafe")
    try:
        if path.stat().st_size > _MAX_METADATA_JSON_BYTES:
            raise ValueError("metadata artifact size limit exceeded")
        with path.open("rb") as handle:
            raw = handle.read(_MAX_METADATA_JSON_BYTES + 1).decode("utf-8")
        value = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_text(value):
        raise ValueError(f"{name} must be a canonical JSON object")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _new_staging(root: Path, name: str) -> Path:
    if _has_symlink(root):
        raise ValueError("output root cannot traverse symlinks")
    root.mkdir(parents=True, exist_ok=True)
    if (root / name).exists() or (root / name).is_symlink():
        raise ValueError(f"{name} output already exists")
    return Path(tempfile.mkdtemp(prefix=f".{name}-", dir=root))


def _publish_new(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError("artifact output already exists")
    os.replace(staging, output)


def _has_symlink(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _path(name: str, value: object) -> None:
    if not isinstance(value, Path):
        raise ValueError(f"{name} must be a pathlib.Path")
    if _has_symlink(value):
        raise ValueError(f"{name} cannot traverse symlinks")


def _digest(name: str, value: object) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    if not root.is_dir() or _has_symlink(root):
        raise ValueError("artifact tree is missing or unsafe")
    digest = hashlib.sha256(b"wfcllm-artifact-tree/v1\0")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            if path.is_dir() and not path.is_symlink():
                continue
            raise ValueError("artifact tree contains unsupported entries")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big") + relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    return digest.hexdigest()


def _copy_tree_snapshot(source: Path, target: Path) -> None:
    if not source.is_dir() or _has_symlink(source) or target.exists():
        raise ValueError("snapshot source or target is unsafe")
    target.mkdir(parents=True)
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_symlink():
            raise ValueError("snapshot source contains symlink")
        if path.is_dir():
            destination.mkdir()
            continue
        if not path.is_file():
            raise ValueError("snapshot source contains unsupported entry")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as input_handle, destination.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=_HASH_CHUNK_BYTES)
        if _sha_file(path) != _sha_file(destination):
            raise ValueError("snapshot copy changed while being read")
        destination.chmod(0o444)


def _validate_formal_candidate_tree(root: Path) -> None:
    if not root.is_dir() or _has_symlink(root) or {path.name for path in root.iterdir()} != {"gate_float.pt", "tokenizer"}:
        raise ValueError("formal candidate bundle allowlist mismatch")
    model = root / "gate_float.pt"
    tokenizer = root / "tokenizer"
    if not model.is_file() or not tokenizer.is_dir() or _has_symlink(model) or _has_symlink(tokenizer):
        raise ValueError("formal candidate files are missing or unsafe")
    tokenizer_entries = sorted(tokenizer.rglob("*"))
    entries = [model, tokenizer, *tokenizer_entries]
    regular = [path for path in entries if path.is_file()]
    if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in entries):
        raise ValueError("formal candidate contains unsupported entries")
    if len(entries) > _MAX_CANDIDATE_FILES or sum(path.is_file() for path in tokenizer_entries) > _MAX_TOKENIZER_FILES:
        raise ValueError("formal candidate entry/file-count limit exceeded")
    sizes = [path.stat().st_size for path in regular]
    if any(size > _MAX_CANDIDATE_FILE_BYTES for size in sizes) or sum(sizes) > _MAX_CANDIDATE_TOTAL_BYTES:
        raise ValueError("formal candidate size limit exceeded")
    forbidden = ("secret", "checkpoint", "optimizer", "private", "key")
    for path in tokenizer_entries:
        relative = path.relative_to(tokenizer)
        parts = relative.parts
        if len(parts) > _MAX_CANDIDATE_DEPTH:
            raise ValueError("formal candidate path depth limit exceeded")
        try:
            relative_utf8 = relative.as_posix().encode("utf-8")
            relative_fs = os.fsencode(relative.as_posix())
            segment_sizes = [max(len(part.encode("utf-8")), len(os.fsencode(part))) for part in parts]
        except (UnicodeEncodeError, OSError) as exc:
            raise ValueError("formal candidate path encoding is unsafe") from exc
        if max(len(relative_utf8), len(relative_fs)) > _MAX_CANDIDATE_RELATIVE_PATH_BYTES or any(
            size > _MAX_CANDIDATE_SEGMENT_BYTES for size in segment_sizes
        ):
            raise ValueError("formal candidate path length limit exceeded")
        if any(any(token in part.casefold() for token in forbidden) for part in parts):
            raise ValueError("formal tokenizer path is forbidden")


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return json.loads(_canonical_bytes(dict(value)))
