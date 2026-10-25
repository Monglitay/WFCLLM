"""Single-machine, local-only production runtime for gated experiments."""

from __future__ import annotations

import ast
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import shutil
import textwrap
import unicodedata
from typing import Any

from wfcllm.gate.data import (
    GateBuildContext,
    GateDataBuilder,
    LshProbeResult,
    RewriteRequest,
    _make_request,
    complete_w1_w2_w3_start_unit_ids,
)
from wfcllm.generation.window_rewriter import (
    CausalWindowRewriter,
    KeyBlindAstEquivalentWindowRewriter,
    RewriteGeneration,
    StatementUnitExtractor,
    _parse_generation,
)
from wfcllm.gate.labels import LabelThresholds, build_gate_labels
from wfcllm.gate.pipeline import GatePipelineGroup
from wfcllm.gate.schema import CandidateObservation, GateTrainingGroup
from wfcllm.gate.sources import GateSourceRecord, canonical_gate_source_identity
from wfcllm.semantic.keying import WatermarkKeying
from wfcllm.semantic.window_lsh import canonical_semantic_window_text
from wfcllm.gate.input import GATE_INPUT_CONTRACT_VERSION
from wfcllm.windowing import (
    WINDOW_CONTRACT_VERSION,
    GateScores,
    GateThresholds,
    PythonStatementUnitExtractor,
    get_statement_unit_extractor,
    language_for_window_contract,
)
from wfcllm.windowing.contracts import is_supported_window_contract
from wfcllm.windowing.normalization import WINDOW_NORMALIZATION_VERSION, normalize_unit_text

LOCAL_HF_ADAPTER_NAME = "local-hf-v1"
_CATALOG_FIELDS = {
    "source_family",
    "source_id",
    "code",
    "repository_id",
    "task_id",
    "function_id",
    "source_model_id",
    "license_id",
    "contract_or_hard_set",
    "prompt",
}
_MAX_CATALOG_LINE_BYTES = 2 * 1024 * 1024
_PUBLIC_SEMANTIC_RUNTIME_VERSION = "wfcllm-public-semantic-runtime/v2"
_PUBLIC_SEMANTIC_INITIALIZATION_SEED = int.from_bytes(
    hashlib.sha256(_PUBLIC_SEMANTIC_RUNTIME_VERSION.encode("utf-8")).digest()[:8],
    "big",
)


@dataclass(frozen=True)
class GateSourceCatalogRecord:
    source_family: str
    source_id: str
    code: str
    repository_id: str | None
    task_id: str | None
    function_id: str | None
    source_model_id: str | None
    license_id: str | None
    contract_or_hard_set: bool
    prompt: str = ""

    def __post_init__(self) -> None:
        GateSourceRecord(
            source_family=self.source_family,
            source_id=self.source_id,
            code=self.code,
            repository_id=self.repository_id,
            task_id=self.task_id,
            function_id=self.function_id,
            source_model_id=self.source_model_id,
            license_id=self.license_id,
            contract_or_hard_set=self.contract_or_hard_set,
        )
        if not isinstance(self.prompt, str):
            raise ValueError("source catalog prompt must be a string")
        identities = (self.source_id, self.repository_id, self.task_id, self.function_id)
        if any("humaneval" in _identity(value) for value in identities if value is not None):
            raise ValueError("HumanEval is forbidden from gate source catalogs")


@dataclass(frozen=True)
class LocalHFGateRuntimeOptions:
    source_catalog: Path
    generation_model_path: Path
    rewrite_model_path: Path | None
    semantic_encoder_model_path: Path
    semantic_encoder_checkpoint_path: Path
    gate_base_model_path: Path
    model_device: str = "cuda"
    gate_device: str = "cuda"
    cache_dir: Path = Path("data/gate-cache")
    semantic_embed_dim: int = 128
    lsh_dimension: int = 4
    lsh_gamma: float = 0.25
    semantic_margin: float = 0.0
    close_low_threshold: float = 0.45
    close_high_threshold: float = 0.55
    suitable_accept_threshold: float = 0.5
    max_units: int = 3
    semantic_evidence_rule: str = "semantic_lsh"
    semantic_preservation_threshold: float = 0.9
    rewrite_max_new_tokens: int = 32
    rewrite_generation_attempts: int = 3
    rewrite_temperature: float = 0.8
    rewrite_top_p: float = 0.95
    gate_batch_size: int = 9
    gate_epochs: int = 4
    gate_max_tokens: int = 256
    gate_learning_rate: float = 2e-5
    gate_early_stopping_patience: int = 1
    window_contract_version: str = WINDOW_CONTRACT_VERSION
    rewrite_strategy: str = "model_semantic_window"

    def __post_init__(self) -> None:
        for name in (
            "source_catalog",
            "generation_model_path",
            "semantic_encoder_model_path",
            "semantic_encoder_checkpoint_path",
            "gate_base_model_path",
            "cache_dir",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a pathlib.Path")
        if self.rewrite_model_path is not None and not isinstance(
            self.rewrite_model_path, Path
        ):
            raise ValueError("rewrite_model_path must be a pathlib.Path or None")
        if self.rewrite_strategy not in {
            "model_semantic_window", "python_ast_equivalent", "cpp_structural_equivalent", "js_structural_equivalent", "java_structural_equivalent",
        }:
            raise ValueError("unsupported local rewrite strategy")
        language = language_for_window_contract(self.window_contract_version)
        if self.rewrite_strategy == "cpp_structural_equivalent" and language != "cpp":
            raise ValueError("C++ structural strategy requires the C++ window contract")
        if self.rewrite_strategy == "js_structural_equivalent" and language != "js":
            raise ValueError("JavaScript structural strategy requires the JS window contract")
        if self.rewrite_strategy == "java_structural_equivalent" and language != "java":
            raise ValueError("Java structural strategy requires the Java window contract")
        if self.rewrite_strategy == "python_ast_equivalent" and language != "python":
            raise ValueError("Python AST strategy requires the Python window contract")
        if (
            language_for_window_contract(self.window_contract_version) != "python"
            and self.rewrite_strategy == "model_semantic_window"
            and self.rewrite_model_path is None
        ):
            raise ValueError("rewrite_model_path is required for model rewriting")
        for name in ("model_device", "gate_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.semantic_evidence_rule != "semantic_lsh":
            raise ValueError("semantic_evidence_rule must be semantic_lsh")
        if not is_supported_window_contract(self.window_contract_version):
            raise ValueError("unsupported window contract version")
        if (
            isinstance(self.lsh_gamma, bool)
            or not isinstance(self.lsh_gamma, (int, float))
            or not 0.0 < self.lsh_gamma < 1.0
        ):
            raise ValueError("lsh_gamma must be in (0, 1)")
        if (
            isinstance(self.semantic_margin, bool)
            or not isinstance(self.semantic_margin, (int, float))
            or not math.isfinite(float(self.semantic_margin))
            or self.semantic_margin < 0.0
        ):
            raise ValueError("semantic_margin must be finite and non-negative")
        for name in (
            "close_low_threshold",
            "close_high_threshold",
            "suitable_accept_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if self.close_low_threshold >= self.close_high_threshold:
            raise ValueError(
                "close_low_threshold must be less than close_high_threshold"
            )
        if (
            isinstance(self.semantic_preservation_threshold, bool)
            or not isinstance(self.semantic_preservation_threshold, (int, float))
            or not 0.0 <= self.semantic_preservation_threshold <= 1.0
        ):
            raise ValueError(
                "semantic_preservation_threshold must be in [0, 1]"
            )
        for name in (
            "semantic_embed_dim",
            "lsh_dimension",
            "rewrite_max_new_tokens",
            "rewrite_generation_attempts",
            "gate_batch_size",
            "gate_epochs",
            "gate_max_tokens",
            "gate_early_stopping_patience",
            "max_units",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gate_batch_size < 3:
            raise ValueError("gate_batch_size must be at least 3")
        if self.max_units > 3:
            raise ValueError("max_units must be between 1 and 3")
        if self.rewrite_generation_attempts != 3:
            raise ValueError("rewrite_generation_attempts must equal 3")
        if (
            isinstance(self.rewrite_temperature, bool)
            or not isinstance(self.rewrite_temperature, (int, float))
            or self.rewrite_temperature <= 0
        ):
            raise ValueError("rewrite_temperature must be positive")
        if (
            isinstance(self.rewrite_top_p, bool)
            or not isinstance(self.rewrite_top_p, (int, float))
            or not 0 < self.rewrite_top_p <= 1
        ):
            raise ValueError("rewrite_top_p must be in (0, 1]")
        if not isinstance(self.gate_learning_rate, (int, float)) or self.gate_learning_rate <= 0:
            raise ValueError("gate_learning_rate must be positive")
        required = (
            self.source_catalog,
            self.generation_model_path,
            self.semantic_encoder_model_path,
            self.semantic_encoder_checkpoint_path,
            self.gate_base_model_path,
        )
        if self.rewrite_model_path is not None:
            required = (*required, self.rewrite_model_path)
        for path in required:
            if path.is_symlink() or not path.exists():
                raise ValueError(f"local runtime resource is missing or unsafe: {path}")

    @property
    def effective_rewrite_model_path(self) -> Path:
        if self.rewrite_model_path is None:
            raise ValueError("rewrite model is unavailable for this runtime")
        return self.rewrite_model_path


def load_source_catalog(path: Path) -> Iterator[GateSourceCatalogRecord]:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("gate source catalog must be a local non-symlink JSONL file")
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if len(raw) > _MAX_CATALOG_LINE_BYTES:
                raise ValueError(f"source catalog line {line_number} exceeds the size limit")
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"source catalog line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict) or set(value) != _CATALOG_FIELDS:
                raise ValueError(f"source catalog line {line_number} schema mismatch")
            try:
                record = GateSourceCatalogRecord(**value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"source catalog line {line_number} is invalid: {exc}") from exc
            normalized = _identity(record.source_id)
            if normalized in seen:
                raise ValueError(f"source catalog line {line_number} duplicates source_id")
            seen.add(normalized)
            yield record


def phase_config_hash(config: Mapping[str, Any]) -> str:
    return _canonical_hash(config)


def experiment_contract_hash(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping):
        raise ValueError("resolved config must be a mapping")
    value = deepcopy(dict(config))
    gate_data = value.get("gate_data")
    if isinstance(gate_data, dict):
        gate_data.pop("scale", None)
    value.pop("runtime", None)
    value.pop("artifacts", None)
    return _canonical_hash(value)


def _canonical_hash(value: object) -> str:
    try:
        payload = json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("config must be canonical JSON data") from exc
    return hashlib.sha256(payload).hexdigest()


def _identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _gate_data_semantic_input_span(
    request: RewriteRequest, code: str, *, extractor: StatementUnitExtractor, window_contract_version: str,
) -> tuple[str, tuple[int, int]]:
    """Recover the same exact candidate-unit slice used by online scoring."""
    parsed = _parse_generation(
        request, RewriteGeneration((), code, "gate-data-span/v1", "gate-data-span/v1"),
        extractor=extractor, window_contract_version=window_contract_version,
    )
    if parsed.parse_status != "ok" or not parsed.same_parent_scope or not parsed.parser_spans:
        raise ValueError("gate-data candidate parser facts disagree with semantic input span")
    prefix_bytes = len(request.completed_prefix.encode("utf-8"))
    first = parsed.parser_spans[0][0] - prefix_bytes
    last = parsed.parser_spans[-1][1] - prefix_bytes
    raw = code.encode("utf-8")
    if not 0 <= first < last <= len(raw):
        raise ValueError("gate-data semantic input span escaped candidate bytes")
    return raw[first:last].decode("utf-8"), (first, last)


class LocalHFProductionAdapter:
    """Lazy production adapter; heavy local models load only in phase methods."""

    diagnostic_test_backend = False
    adapter_contract_version = "wfcllm-production-gate-adapter/v1"
    capabilities = frozenset(
        {
            "parse_statement_units",
            "generate_candidate_trajectories",
            "multi_key_lsh_probe_with_private_material",
            "split_groups",
            "audit_gate_data",
            "train_candidate",
        }
    )

    def __init__(self, options: LocalHFGateRuntimeOptions) -> None:
        if not isinstance(options, LocalHFGateRuntimeOptions):
            raise ValueError("local-hf-v1 requires LocalHFGateRuntimeOptions")
        self.options = options
        language = language_for_window_contract(
            options.window_contract_version
        )
        extractor = get_statement_unit_extractor(language)
        self._rewriter: object | None = (
            KeyBlindAstEquivalentWindowRewriter()
            if language == "python"
            else None
        )
        self._semantic_runtime: object | None = None
        self._parent_by_group: dict[str, str] = {}
        self._semantic_text_by_candidate: dict[tuple[str, int, int], str] = {}
        self._training_group_by_group: dict[str, GateTrainingGroup] = {}
        self._unit_metadata_by_group: dict[str, dict[str, Any]] = {}
        self._gate_tokenizer: object | None = None
        self._cache_config_hash: str | None = None
        self._selection_summary: dict[str, Any] | None = None

    def parse_statement_units(self, source_manifest, config):
        if not isinstance(source_manifest, Mapping):
            raise ValueError("gate source manifest must be a mapping")
        expected = source_manifest.get("catalog_sha256")
        actual = _sha256_file(self.options.source_catalog)
        if expected != actual:
            raise ValueError("gate source catalog hash does not match source manifest")
        records = tuple(load_source_catalog(self.options.source_catalog))
        if source_manifest.get("source_count") != len(records):
            raise ValueError("gate source manifest count does not match catalog")
        language = language_for_window_contract(
            self.options.window_contract_version
        )
        extractor = get_statement_unit_extractor(language)
        return tuple(
            _ParsedCatalogSource(record, tuple(extractor.extract(record.code)))
            for record in records
        )

    def generate_candidate_trajectories(self, parsed_units, config):
        if not isinstance(parsed_units, tuple) or any(
            not isinstance(item, _ParsedCatalogSource) for item in parsed_units
        ):
            raise ValueError("parsed source catalog has an invalid runtime shape")
        rewriter = self._rewriter
        language = language_for_window_contract(
            self.options.window_contract_version
        )
        if rewriter is None:
            rewriter = _load_causal_rewriter(self.options)
            self._rewriter = rewriter
        rewrite_identity = (
            "python-ast-equivalent/v1"
            if language == "python"
            else "cpp-structural-equivalent/v1"
            if self.options.rewrite_strategy == "cpp_structural_equivalent"
            else "js-structural-expressions/v1"
            if self.options.rewrite_strategy == "js_structural_equivalent"
            else "java-structural-expressions/v1"
            if self.options.rewrite_strategy == "java_structural_equivalent"
            else self.options.effective_rewrite_model_path.name
        )
        builder = GateDataBuilder(
            rewriter=rewriter,
            lsh_probe=_StructuralOnlyProbe(),
            rewrite_count=config.rewrite_count,
        )
        self.options.cache_dir.mkdir(parents=True, exist_ok=True)
        semantic_span_path = self.options.cache_dir / f"semantic-input-spans-{config.scale}-{config.config_hash}.jsonl"
        if semantic_span_path.exists() or semantic_span_path.is_symlink():
            raise ValueError("fresh gate-data semantic input span evidence already exists")
        cache_path = self._training_cache_path(config.config_hash)
        cache_path.unlink(missing_ok=True)
        self._cache_config_hash = config.config_hash
        max_groups = config.max_groups
        selected_starts = _select_source_stratified_starts(
            parsed_units,
            max_groups=max_groups,
        )
        split_assignments = _balanced_split_assignments(
            tuple(
                _record_split_group_id(selected.parsed.record)
                for selected in selected_starts
            )
        )
        self._selection_summary = _source_stratified_selection_summary(
            parsed_units,
            selected_starts,
            max_groups=max_groups,
            language=language_for_window_contract(config.parser_contract),
        )
        semantic_span_rows: list[dict[str, Any]] = []
        candidate_extractor = get_statement_unit_extractor(language)
        starts_by_source: dict[str, list[str]] = defaultdict(list)
        parsed_by_source: dict[str, _ParsedCatalogSource] = {}
        for selected in selected_starts:
            source_id = selected.parsed.record.source_id
            parsed_by_source[source_id] = selected.parsed
            starts_by_source[source_id].append(selected.start_unit_id)
        for source_id in dict.fromkeys(
            selected.parsed.record.source_id for selected in selected_starts
        ):
            parsed = parsed_by_source[source_id]
            record = parsed.record
            start_by_id = {unit.unit_id: unit for unit in parsed.units}
            groups = builder.build(
                parsed.units,
                context=GateBuildContext(
                    prompt=record.prompt,
                    source_id=record.source_id,
                    source_family=record.source_family,
                    repository_id=record.repository_id,
                    task_id=record.task_id,
                    function_id=record.function_id,
                    language=language_for_window_contract(
                        config.parser_contract
                    ),
                    parser_contract_version=config.parser_contract,
                ),
                source_text=record.code,
                selected_start_unit_ids=tuple(starts_by_source[source_id]),
                max_units=config.window_lengths[-1],
            )
            for built in groups:
                candidates = built.candidates_by_length
                lengths = config.window_lengths
                length_keys = {str(length) for length in lengths}
                if set(candidates) != length_keys:
                    continue
                training_group = built.training_group
                start = start_by_id[training_group.window_start_unit_id]
                start_index = next(
                    index
                    for index, unit in enumerate(parsed.units)
                    if unit.unit_id == training_group.window_start_unit_id
                )
                split = split_assignments[built.split_group_id]
                self._parent_by_group[built.group_id] = training_group.parent_descriptor
                for length in lengths:
                    request = _make_request(
                        parsed.units, start_index=start_index,
                        window=tuple(parsed.units[start_index : start_index + length]),
                        prompt=record.prompt, source_bytes=record.code.encode("utf-8"),
                        window_contract_version=config.parser_contract,
                    )
                    for observation in candidates[str(length)]:
                        if not (observation.parse_status == "ok" and observation.same_parent_scope
                                and observation.unit_count == length):
                            continue
                        if observation.candidate_index == 0:
                            if observation.code != request.original_window:
                                raise ValueError("candidate zero differs from its original source window")
                            semantic_text = request.original_window
                            semantic_span = (0, len(semantic_text.encode("utf-8")))
                        else:
                            semantic_text, semantic_span = _gate_data_semantic_input_span(
                                request, observation.code, extractor=candidate_extractor,
                                window_contract_version=config.parser_contract,
                            )
                        self._semantic_text_by_candidate[(built.group_id, length, observation.candidate_index)] = semantic_text
                        semantic_span_rows.append({
                            "contract_version": "wfcllm-gate-semantic-input-span/v1",
                            "group_id": built.group_id, "window_length": length,
                            "candidate_index": observation.candidate_index,
                            "semantic_span_in_candidate": list(semantic_span),
                            "raw_candidate_sha256": hashlib.sha256(observation.code.encode("utf-8")).hexdigest(),
                            "semantic_span_sha256": hashlib.sha256(semantic_text.encode("utf-8")).hexdigest(),
                            "semantic_input_sha256": hashlib.sha256(canonical_semantic_window_text(semantic_text).encode("utf-8")).hexdigest(),
                            "audit_only": True, "not_detector_input": True,
                        })
                self._training_group_by_group[built.group_id] = training_group
                self._unit_metadata_by_group[built.group_id] = {
                    "depth": start.depth,
                    "previous_unit_types": [
                        unit.node_type for unit in parsed.units[:start_index][-3:]
                    ],
                    "current_units": {
                        str(length): [
                            unit.text for unit in parsed.units[start_index : start_index + length]
                        ]
                        for length in lengths
                    },
                    "current_unit_types": {
                        str(length): [
                            unit.node_type for unit in parsed.units[start_index : start_index + length]
                        ]
                        for length in lengths
                    },
                    "source_family": record.source_family,
                }
                yield GatePipelineGroup(
                    group_id=built.group_id,
                    split_group_id=built.split_group_id,
                    split=split,
                    suitable_target=False,
                    close_target=False,
                    window_lengths=lengths,
                    statement_family=_statement_family(
                        start.text,
                        cpp_node_type=(start.node_type if config.parser_contract == "cpp-statement-window/v1" else None),
                    ),
                    r1_success_rate=0.0,
                    r3_success_rate=0.0,
                    holdout_success_rate=0.0,
                    repository_id=record.repository_id or record.task_id or record.function_id or record.source_id,
                    task_id=record.task_id or record.function_id or record.source_id,
                    generation_model_id=(
                        record.source_model_id
                        or rewrite_identity
                    ),
                    structural_invalid_rate=0.0,
                    numeric_instability_rate=0.0,
                    first_hit_candidate_position=None,
                    candidate_indices_by_window_length={
                        length: tuple(range(config.rewrite_count + 1))
                        for length in lengths
                    },
                    observed_training_key_ids=(),
                    observed_holdout_key_ids=(),
                    candidate_observations_by_length=candidates,
                    probe_results_by_length={
                        str(length): tuple(
                            {} for _index in range(config.rewrite_count + 1)
                        )
                        for length in lengths
                    },
                    row={
                        "schema_version": "wfcllm-gate-data/v1",
                        "group_id": built.group_id,
                        "split": split,
                    },
                )

        with semantic_span_path.open("x", encoding="utf-8") as handle:
            for row in semantic_span_rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self._selection_summary["semantic_input_span_audit"] = {
            "contract_version": "wfcllm-gate-semantic-input-span/v1",
            "filename": semantic_span_path.name,
            "sha256": _sha256_file(semantic_span_path),
            "candidate_count": len(semantic_span_rows),
        }

    def gate_data_selection_summary(self) -> dict[str, Any]:
        if self._selection_summary is None:
            raise ValueError("gate-data selection has not been computed")
        return deepcopy(self._selection_summary)

    def run_multi_key_lsh_probe(self, groups, *, training_keys, holdout_keys, config):
        runtime = self._semantic_runtime
        if runtime is None:
            runtime = _load_semantic_runtime(self.options)
            self._semantic_runtime = runtime
        all_ids = (*training_keys.key_ids, *holdout_keys.key_ids)
        materials = {
            key_id: bytes(
                training_keys.material_for(key_id)
                if key_id in training_keys.key_ids
                else holdout_keys.material_for(key_id)
            ).decode("utf-8")
            for key_id in all_ids
        }
        for group in groups:
            parent = self._parent_by_group.get(group.group_id)
            if parent is None:
                raise ValueError("semantic probe is missing the group parent descriptor")
            allowed_by_key = {
                key_id: WatermarkKeying(
                    material, self.options.lsh_dimension
                ).derive_descriptor(
                    contract_version=config.parser_contract,
                    parent_descriptor=parent,
                    k=max(
                        1,
                        round(
                            self.options.lsh_gamma
                            * (2 ** self.options.lsh_dimension)
                        ),
                    ),
                )
                for key_id, material in materials.items()
            }
            observations_by_length: dict[str, tuple[CandidateObservation, ...]] = {}
            results_by_length: dict[str, tuple[Mapping[str, LshProbeResult], ...]] = {}
            for length in group.window_lengths:
                observations: list[CandidateObservation] = []
                candidate_results: list[Mapping[str, LshProbeResult]] = []
                trajectory = group.candidate_observations_by_length[str(length)]
                reference_text = self._semantic_text_by_candidate.get((group.group_id, length, 0))
                if reference_text is None:
                    raise ValueError("gate-data original semantic input span is missing")
                for observation in trajectory:
                    exact_structural_candidate = (
                        observation.parse_status == "ok"
                        and observation.same_parent_scope
                        and observation.unit_count == length
                    )
                    if observation.parse_status == "ok" and not exact_structural_candidate:
                        raise ValueError(
                            "ok candidate parser facts contradict requested window length/scope"
                        )
                    if not exact_structural_candidate:
                        observations.append(
                            replace(
                                observation,
                                stable_across_precision_modes=False,
                                stable_across_batch_modes=False,
                                lsh_by_key_id={},
                                lsh_signature=None,
                                semantic_reference_cosine=None,
                                semantic_preservation_passed=None,
                                semantic_probe_pending=False,
                            )
                        )
                        candidate_results.append({})
                        continue
                    semantic_text = self._semantic_text_by_candidate.get(
                        (group.group_id, length, observation.candidate_index)
                    )
                    if semantic_text is None:
                        raise ValueError("gate-data candidate semantic input span is missing")
                    semantic_cosine = runtime.semantic_reference_cosine(reference_text, semantic_text)
                    semantic_preserved = bool(
                        semantic_cosine
                        >= self.options.semantic_preservation_threshold
                    )
                    if not semantic_preserved:
                        observations.append(
                            replace(
                                observation,
                                stable_across_precision_modes=False,
                                stable_across_batch_modes=False,
                                lsh_by_key_id={},
                                lsh_signature=None,
                                semantic_reference_cosine=float(
                                    semantic_cosine
                                ),
                                semantic_preservation_passed=False,
                                semantic_probe_pending=False,
                            )
                        )
                        candidate_results.append({})
                        continue
                    signature, margin, precision_stable, batch_stable = runtime.signature_and_margin(
                        semantic_text
                    )
                    if len(signature) != self.options.lsh_dimension:
                        raise ValueError(
                            "semantic signature dimension does not match runtime options"
                        )
                    stable = bool(precision_stable and batch_stable)
                    results: dict[str, LshProbeResult] = {}
                    for key_id, allowed in allowed_by_key.items():
                        results[key_id] = LshProbeResult(
                            signature=signature,
                            margin=float(margin),
                            hit=stable and signature in allowed,
                            stable=stable,
                            stable_across_precision_modes=bool(precision_stable),
                            stable_across_batch_modes=bool(batch_stable),
                        )
                    training_results = {key_id: results[key_id] for key_id in training_keys.key_ids}
                    observations.append(
                        replace(
                            observation,
                            stable_across_precision_modes=bool(precision_stable),
                            stable_across_batch_modes=bool(batch_stable),
                            lsh_by_key_id={
                                key_id: {
                                    "hit": result.hit,
                                    "stable": result.stable,
                                    "margin": result.margin,
                                }
                                for key_id, result in training_results.items()
                            },
                            lsh_signature=signature,
                            semantic_reference_cosine=float(semantic_cosine),
                            semantic_preservation_passed=True,
                            semantic_probe_pending=False,
                        )
                    )
                    candidate_results.append(results)
                observations_by_length[str(length)] = tuple(observations)
                results_by_length[str(length)] = tuple(candidate_results)

            max_units = group.window_lengths[-1]
            selected = build_gate_labels(
                observations_by_length[str(max_units)],
                training_key_count=32,
                thresholds=LabelThresholds(
                    configured_margin=self.options.semantic_margin,
                    max_units=max_units,
                ),
                rewrite_budgets=config.rewrite_budgets,
            )
            r1_budget = 1 if 1 in selected.budgets else max(selected.budgets)
            r3_budget = 3 if 3 in selected.budgets else max(selected.budgets)
            selected_budget = max(selected.budgets)
            r1, r3 = (
                selected.budgets[r1_budget].success_rate,
                selected.budgets[r3_budget].success_rate,
            )
            holdout_hits = sum(
                any(
                    key_id in results_by_length[str(max_units)][candidate]
                    and results_by_length[str(max_units)][candidate][key_id].is_reliable_hit(
                        configured_margin=self.options.semantic_margin
                    )
                    # Transfer degradation compares both banks at the R3 prefix.
                    for candidate in range(r3_budget + 1)
                )
                for key_id in holdout_keys.key_ids
            )
            first_hits = [
                value
                for value in selected.budgets[selected_budget].first_hit_by_key_id.values()
                if value is not None
            ]
            result = replace(
                group,
                suitable_target=selected.suitable_target,
                close_target=selected.close_target,
                r1_success_rate=r1,
                r3_success_rate=r3,
                holdout_success_rate=holdout_hits / len(holdout_keys.key_ids),
                structural_invalid_rate=(
                    1.0 - selected.budgets[selected_budget].structural_valid_rate
                ),
                numeric_instability_rate=selected.budgets[selected_budget].unstable_rate,
                first_hit_candidate_position=min(first_hits) if first_hits else None,
                observed_training_key_ids=tuple(training_keys.key_ids),
                observed_holdout_key_ids=tuple(holdout_keys.key_ids),
                candidate_observations_by_length=observations_by_length,
                probe_results_by_length=results_by_length,
            )
            self._append_training_examples(
                result,
                observations_by_length,
                rewrite_budgets=config.rewrite_budgets,
            )
            yield result

    def split_groups(self, groups, config):
        yield from groups

    def audit_gate_data(self, staging_dir, manifest):
        formal_identity = {
            "diagnostic_test_backend": False,
            "formal_eligible": True,
            "diagnostic_only": False,
            "not_official_method": False,
        }
        diagnostic_identity = {
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "diagnostic_only": True,
            "not_official_method": True,
        }
        if not isinstance(manifest, Mapping) or not any(
            all(manifest.get(field) is value for field, value in identity.items())
            for identity in (formal_identity, diagnostic_identity)
        ):
            raise ValueError(
                "local-hf-v1 requires a complete gate-data artifact identity"
            )
        config_hash = manifest.get("config_hash")
        if not isinstance(config_hash, str) or not self._training_cache_path(config_hash).is_file():
            raise ValueError("local-hf-v1 training example cache is missing")

    def train_candidate(
        self, *, config, data_manifest, data_jsonl, output_dir, learning_curve_plan
    ):
        import torch
        from transformers import AutoTokenizer

        from wfcllm.gate.config import GateTrainConfig
        from wfcllm.gate.losses import GateLoss, GateLossWeights
        from wfcllm.gate.model import GateModel
        from wfcllm.gate.trainer import GateTrainer, GateTrainerConfig, seed_gate_training

        del learning_curve_plan
        cache_rows = self._load_training_cache(config.config_hash)
        _validate_cache_against_data(cache_rows, data_jsonl.parent)
        examples = tuple(_gate_example_from_cache(row) for row in cache_rows)
        training, validation, validation_role = _partition_training_examples(
            examples,
            cache_rows,
        )
        if not training or not validation:
            raise ValueError("gate training requires non-empty train and validation examples")
        seed_gate_training(7)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.options.gate_base_model_path), local_files_only=True
        )
        model = GateModel.from_local_pretrained(
            GateTrainConfig(
                max_tokens=self.options.gate_max_tokens,
                base_model_path=self.options.gate_base_model_path,
            )
        )
        work = output_dir.parent / "_training_work"
        if work.exists():
            shutil.rmtree(work)
        trainer = GateTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir=work,
            config_hash=config.config_hash,
            dataset_manifest_hash=_sha256_file(data_jsonl.parent / "manifest.json"),
            config=GateTrainerConfig(
                epochs=self.options.gate_epochs,
                batch_size=self.options.gate_batch_size,
                learning_rate=float(self.options.gate_learning_rate),
                early_stopping_patience=self.options.gate_early_stopping_patience,
                max_tokens=self.options.gate_max_tokens,
                enable_consistency=False,
            ),
            loss_fn=GateLoss(
                GateLossWeights(
                    context_consistency=0.0,
                    batch_consistency=0.0,
                    quantization_consistency=0.0,
                )
            ),
            device=self.options.gate_device,
        )
        summary = trainer.fit(training, validation)
        checkpoint = torch.load(
            work / "checkpoints" / "best.pt",
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint.get("model_state")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("best gate checkpoint does not contain model state")
        output_dir.mkdir(parents=True, exist_ok=False)
        torch.save(state, output_dir / "gate_float.pt")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        # Keep the safe learning curves outside the published model bundle.
        for name, saved_name in (
            ("training_metrics.jsonl", "training_metrics.jsonl"),
            ("development_summary.json", "training_development_summary.json"),
        ):
            destination = output_dir.parent / saved_name
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"gate training evidence already exists: {name}")
            shutil.copyfile(work / name, destination)
        shutil.rmtree(work)
        return {
            "backend": LOCAL_HF_ADAPTER_NAME,
            "best_epoch": summary["best_epoch"],
            "epochs_completed": summary["epochs_completed"],
            "training_example_count": len(training),
            "validation_example_count": len(validation),
            "validation_split_role": validation_role,
            "suitable_accept_threshold": summary["best_validation"][
                "suitable_accept_threshold"
            ],
            "suitable_threshold_false_positive_rate": summary["best_validation"][
                "suitable_threshold_false_positive_rate"
            ],
            "suitable_threshold_true_positive_rate": summary["best_validation"][
                "suitable_threshold_true_positive_rate"
            ],
            "candidate_sha256": _sha256_file(output_dir / "gate_float.pt"),
        }


    def _training_cache_path(self, config_hash: str) -> Path:
        return self.options.cache_dir / f"gate-examples-{config_hash}.jsonl"

    def _append_training_examples(
        self,
        group: GatePipelineGroup,
        observations_by_length: Mapping[str, tuple[CandidateObservation, ...]],
        *,
        rewrite_budgets: tuple[int, ...],
    ) -> None:
        from wfcllm.gate.input import GateInput, serialize_gate_input

        training_group = self._training_group_by_group[group.group_id]
        metadata = self._unit_metadata_by_group[group.group_id]
        tokenizer = self._gate_tokenizer
        if tokenizer is None:
            tokenizer = _load_gate_tokenizer(self.options)
            self._gate_tokenizer = tokenizer
        rows: list[dict[str, Any]] = []
        for length in group.window_lengths:
            labels = build_gate_labels(
                observations_by_length[str(length)],
                training_key_count=32,
                thresholds=LabelThresholds(
                    configured_margin=self.options.semantic_margin,
                    max_units=group.window_lengths[-1],
                ),
                rewrite_budgets=rewrite_budgets,
            )
            current_units = tuple(metadata["current_units"][str(length)])
            normalized = "\n".join(normalize_unit_text(unit) for unit in current_units)
            token_count = _token_count(tokenizer, normalized)
            gate_input = GateInput(
                normalization_version=WINDOW_NORMALIZATION_VERSION,
                parent_descriptor=training_group.parent_descriptor,
                depth=metadata["depth"],
                previous_units=training_group.previous_units,
                previous_unit_types=tuple(metadata["previous_unit_types"]),
                current_units=current_units,
                current_unit_types=tuple(metadata["current_unit_types"][str(length)]),
                current_unit_count=length,
                current_token_count=token_count,
            )
            for budget in rewrite_budgets:
                rows.append(
                    {
                        "contract_version": "wfcllm-local-gate-cache/v1",
                        "config_hash": self._cache_config_hash,
                        "group_id": group.group_id,
                        "window_start_unit_id": training_group.window_start_unit_id,
                        "split": group.split,
                        "context_length": length,
                        "budget": budget,
                        "serialized_gate_input": serialize_gate_input(gate_input),
                        "gate_input": {
                            "normalization_version": gate_input.normalization_version,
                            "parent_descriptor": gate_input.parent_descriptor,
                            "depth": gate_input.depth,
                            "previous_units": list(gate_input.previous_units),
                            "previous_unit_types": list(gate_input.previous_unit_types),
                            "current_units": list(gate_input.current_units),
                            "current_unit_types": list(gate_input.current_unit_types),
                            "current_unit_count": gate_input.current_unit_count,
                            "current_token_count": gate_input.current_token_count,
                        },
                        "close_target": labels.close_target,
                        "suitable_target": labels.suitable_target,
                        "dangerous_negative": labels.close_target and not labels.suitable_target,
                        "source_family": metadata["source_family"],
                        "generation_model_id": group.generation_model_id,
                        "span": list(observations_by_length[str(length)][0].boundary_span),
                    }
                )
        path = self._training_cache_path(str(self._cache_config_hash))
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")

    def _load_training_cache(self, config_hash: str) -> list[dict[str, Any]]:
        path = self._training_cache_path(config_hash)
        if path.is_symlink() or not path.is_file():
            raise ValueError("local gate training cache is missing")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"local gate cache line {line_number} is invalid") from exc
                if not isinstance(row, dict) or row.get("config_hash") != config_hash:
                    raise ValueError("local gate cache provenance mismatch")
                rows.append(row)
        if not rows:
            raise ValueError("local gate training cache is empty")
        return rows


class HFCausalRewriteBackend:
    """Key-blind whole-window generation backed by one local HF model."""

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        device: str,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        language: str = "python",
    ) -> None:
        if not callable(getattr(model, "generate", None)):
            raise ValueError("rewrite model must expose generate")
        if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
            raise ValueError("rewrite tokenizer must be callable and expose decode")
        if not isinstance(device, str) or not device:
            raise ValueError("rewrite device must be non-empty")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("rewrite max_new_tokens must be positive")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or temperature <= 0
        ):
            raise ValueError("rewrite temperature must be positive")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0 < top_p <= 1
        ):
            raise ValueError("rewrite top_p must be in (0, 1]")
        self.model = model
        self.tokenizer = tokenizer
        if (
            not bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
            and getattr(tokenizer, "padding_side", None) == "right"
        ):
            # Causal generation reads the final position of every batch row.
            self.tokenizer.padding_side = "left"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        _rewrite_language_label(language)
        self.language = language

    def generate_window(
        self,
        *,
        prompt: str,
        completed_prefix: str,
        original_window: str,
        candidate_index: int,
        max_units: int,
    ) -> RewriteGeneration:
        import torch

        instruction = self._render_instruction(
            prompt=prompt,
            completed_prefix=completed_prefix,
            original_window=original_window,
            max_units=max_units,
            candidate_index=candidate_index,
        )
        encoded = self.tokenizer(instruction, return_tensors="pt", truncation=True)
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in dict(encoded).items()
        }
        seed_payload = (
            f"local-hf-v1\0{candidate_index}\0{prompt}\0{completed_prefix}\0{original_window}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
        torch.manual_seed(seed)
        generated = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        if not isinstance(generated, torch.Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
            raise ValueError("rewrite model must return one token sequence")
        token_ids = generated[0]
        is_encoder_decoder = bool(getattr(getattr(self.model, "config", None), "is_encoder_decoder", False))
        if not is_encoder_decoder:
            input_ids = inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
                raise ValueError("causal rewrite input_ids are missing")
            token_ids = token_ids[input_ids.shape[1] :]
        ids = tuple(int(value) for value in token_ids.detach().cpu().tolist())
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        if not isinstance(text, str):
            raise ValueError("rewrite tokenizer decode must return text")
        return RewriteGeneration(
            token_ids=ids,
            text=_extract_rewrite_code(
                text,
                original_window=original_window,
                completed_prefix=completed_prefix,
                language=self.language,
            ),
            generation_seed_id=f"local-hf-v1:{seed:016x}",
            rewrite_config_id=(
                f"local-hf-v1:max-new-tokens={self.max_new_tokens}:"
                f"strategy={_rewrite_strategy_id(candidate_index)}:"
                f"temperature={self.temperature}:top-p={self.top_p}"
            ),
        )

    def generate_windows(
        self,
        *,
        prompt: str,
        completed_prefix: str,
        original_window: str,
        candidate_indices: tuple[int, ...],
        max_units: int,
    ) -> tuple[RewriteGeneration, ...]:
        """Sample one contiguous candidate trajectory in one generate call."""

        import torch

        if not candidate_indices or candidate_indices != tuple(
            range(1, len(candidate_indices) + 1)
        ):
            raise ValueError("batched candidate indices must be contiguous from 1")
        instructions = [
            self._render_instruction(
                prompt=prompt,
                completed_prefix=completed_prefix,
                original_window=original_window,
                max_units=max_units,
                candidate_index=candidate_index,
            )
            for candidate_index in candidate_indices
        ]
        encoded = self.tokenizer(
            instructions,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in dict(encoded).items()
        }
        seed_payload = (
            f"local-hf-v1-batch\0{prompt}\0{completed_prefix}\0{original_window}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
        torch.manual_seed(seed)
        generated = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        if (
            not isinstance(generated, torch.Tensor)
            or generated.ndim != 2
            or generated.shape[0] != len(candidate_indices)
        ):
            raise ValueError("rewrite model must return one sequence per candidate")
        is_encoder_decoder = bool(
            getattr(getattr(self.model, "config", None), "is_encoder_decoder", False)
        )
        prompt_length = inputs["input_ids"].shape[1]
        results: list[RewriteGeneration] = []
        for candidate_index, token_ids in zip(
            candidate_indices,
            generated,
            strict=True,
        ):
            if not is_encoder_decoder:
                token_ids = token_ids[prompt_length:]
            ids = tuple(int(value) for value in token_ids.detach().cpu().tolist())
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            if not isinstance(text, str):
                raise ValueError("rewrite tokenizer decode must return text")
            results.append(
                RewriteGeneration(
                    token_ids=ids,
                    text=_extract_rewrite_code(
                        text,
                        original_window=original_window,
                        completed_prefix=completed_prefix,
                        language=self.language,
                    ),
                    generation_seed_id=f"local-hf-v1-batch:{seed:016x}:{candidate_index}",
                    rewrite_config_id=(
                        f"local-hf-v1-batch:count={len(candidate_indices)}:"
                        f"strategy={_rewrite_strategy_id(candidate_index)}:"
                        f"max-new-tokens={self.max_new_tokens}:"
                        f"temperature={self.temperature}:top-p={self.top_p}"
                    ),
                )
            )
        return tuple(results)

    def _render_instruction(
        self,
        *,
        prompt: str,
        completed_prefix: str,
        original_window: str,
        max_units: int,
        candidate_index: int,
    ) -> str:
        bounded_task = prompt[-2048:]
        bounded_prefix = completed_prefix[-4096:]
        language_label = _rewrite_language_label(self.language)
        strategy = _rewrite_strategy_instruction(candidate_index, language=self.language)
        content = (
            "Task description (context only):\n"
            f"{bounded_task}\n\n"
            "Completed prefix (context only; do not output it):\n"
            f"{bounded_prefix}\n\n"
            f"Rewrite the target window into exactly {max_units} complete {language_label} statements. "
            "Preserve every referenced name, assigned name, called function, attribute name, "
            "literal value, side effect, and control-flow outcome. Do not add imports, "
            "definitions, returns, raises, or calls. Do not rename anything. Use a natural "
            "behavior-preserving alternative only if one exists; otherwise copy the target. "
            f"{strategy} "
            f"Return code only, with exactly {max_units} complete {language_label} statements.\n\n"
            f"Target window:\n{original_window}"
        )
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            return apply_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return content


def _rewrite_strategy_id(candidate_index: int) -> str:
    if type(candidate_index) is not int or candidate_index <= 0:
        raise ValueError("candidate_index must be a positive integer")
    return ("a", "b", "c")[(candidate_index - 1) % 3]


def _rewrite_language_label(language: str) -> str:
    labels = {"python": "Python", "cpp": "C++", "java": "Java", "js": "JavaScript"}
    if not isinstance(language, str) or language not in labels:
        raise ValueError("unsupported rewrite language")
    return labels[language]


def _rewrite_strategy_instruction(candidate_index: int, *, language: str = "python") -> str:
    strategy = _rewrite_strategy_id(candidate_index)
    language_label = _rewrite_language_label(language)
    instructions = {
        "a": (
            "Conservative plan A: make the smallest natural equivalent "
            "expression-level rewrite while retaining evaluation order."
        ),
        "b": (
            f"Conservative plan B: use an independently phrased {language_label} idiom "
            "only when its runtime behavior is identical."
        ),
        "c": (
            "Conservative plan C: re-derive the same statements from the "
            "task and prefix, and copy any statement whose equivalence is uncertain."
        ),
    }
    return instructions[strategy]


def _extract_rewrite_code(
    text: str,
    *,
    original_window: str,
    completed_prefix: str,
    language: str = "python",
) -> str:
    """Extract code blocks and restore indentation supplied by the prefix.

    The source slice starts at the first statement's AST byte, so the completed
    prefix already contains indentation for the generated first line. Every
    later physical line must receive that same public base indentation. Model
    prose is discarded only when it explicitly encloses code in Markdown
    fences; all fenced blocks are retained in their original order.
    """

    _rewrite_language_label(language)
    fence_labels = {
        "python": ("python", "py"),
        "cpp": ("cpp", "c++", "cxx", "cc", "c"),
        "java": ("java",),
        "js": ("javascript", "js"),
    }
    fence_pattern = "|".join(re.escape(label) for label in fence_labels[language])
    fenced_blocks = re.findall(
        rf"```(?:{fence_pattern})?[ \t]*\r?\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = (
        "\n".join(block.strip("\r\n") for block in fenced_blocks)
        if fenced_blocks
        else text
    )
    body = textwrap.dedent(body.strip("\n"))
    prefix_tail = completed_prefix.rsplit("\n", 1)[-1]
    prefix_supplies_indent = bool(prefix_tail) and not prefix_tail.strip()
    original_line = next(
        (line for line in original_window.splitlines() if line.strip()),
        "",
    )
    original_indent = original_line[: len(original_line) - len(original_line.lstrip())]
    base_indent = prefix_tail if prefix_supplies_indent else original_indent
    lines = body.splitlines()
    restored_lines = (
        lines[:1]
        if prefix_supplies_indent
        else [base_indent + lines[0]] if lines else []
    )
    restored_lines.extend(
        base_indent + line if line.strip() else line
        for line in lines[1:]
    )
    restored = "\n".join(restored_lines)
    return restored + ("\n" if restored else "")


class LocalSemanticRuntime:
    """Key-independent semantic runtime with measured mode stability."""

    def __init__(self, verifier: object) -> None:
        required = (
            "semantic_reference_cosine",
            "signature_and_margin_modes",
        )
        if any(not callable(getattr(verifier, name, None)) for name in required):
            raise ValueError(
                "semantic verifier must expose cosine and real mode measurements"
            )
        self.verifier = verifier

    def semantic_reference_cosine(
        self, reference_text: str, candidate_text: str
    ) -> float:
        compare = getattr(self.verifier, "semantic_reference_cosine", None)
        if not callable(compare):
            raise ValueError(
                "semantic verifier must expose semantic_reference_cosine"
            )
        cosine = compare(
            canonical_semantic_window_text(reference_text),
            canonical_semantic_window_text(candidate_text),
        )
        if (
            isinstance(cosine, bool)
            or not isinstance(cosine, (int, float))
            or not -1.0 <= cosine <= 1.0
        ):
            raise ValueError("semantic reference cosine must be in [-1, 1]")
        return float(cosine)

    def signature_and_margin(self, window_text: str):
        measure = getattr(self.verifier, "signature_and_margin_modes", None)
        if not callable(measure):
            raise ValueError(
                "semantic verifier must expose real precision/batch mode measurements"
            )
        signature, margin, precision_stable, batch_stable = measure(
            canonical_semantic_window_text(window_text)
        )
        return (
            tuple(signature),
            float(margin),
            bool(precision_stable),
            bool(batch_stable),
        )


def _gate_example_from_cache(row: Mapping[str, Any]):
    from wfcllm.gate.dataset import GateExample
    from wfcllm.gate.input import GateInput

    raw_input = row.get("gate_input")
    if not isinstance(raw_input, Mapping):
        raise ValueError("local gate cache input is invalid")
    gate_input = GateInput(
        normalization_version=raw_input["normalization_version"],
        parent_descriptor=raw_input["parent_descriptor"],
        depth=raw_input["depth"],
        previous_units=tuple(raw_input["previous_units"]),
        previous_unit_types=tuple(raw_input["previous_unit_types"]),
        current_units=tuple(raw_input["current_units"]),
        current_unit_types=tuple(raw_input["current_unit_types"]),
        current_unit_count=raw_input["current_unit_count"],
        current_token_count=raw_input["current_token_count"],
    )
    return GateExample.from_gate_input(
        group_id=row["group_id"],
        window_start_unit_id=row["window_start_unit_id"],
        context_length=row["context_length"],
        budget=row["budget"],
        gate_input=gate_input,
        close_target=row["close_target"],
        suitable_target=row["suitable_target"],
        dangerous_negative=row["dangerous_negative"],
        source_family=row["source_family"],
        generation_model_id=row["generation_model_id"],
    )


def _load_gate_tokenizer(options: LocalHFGateRuntimeOptions):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(options.gate_base_model_path), local_files_only=True
    )


def _token_count(tokenizer: object, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("gate tokenizer did not return input_ids")
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("gate tokenizer returned an unexpected batch")
        input_ids = input_ids[0]
    if not isinstance(input_ids, list):
        raise ValueError("gate tokenizer returned invalid input_ids")
    return len(input_ids)


@dataclass(frozen=True)
class _CandidateGateManifest:
    window_contract_version: str
    gate_input_contract_version: str
    tokenizer_sha256: str
    close_low_threshold: float
    close_high_threshold: float
    suitable_accept_threshold: float
    max_tokens: int
    max_units: int


class _FloatGatePredictor:
    def __init__(
        self, *, model: object, tokenizer: object, max_tokens: int, device: str = "cpu"
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device
        self._prediction_cache: OrderedDict[tuple[object, ...], GateScores] = OrderedDict()

    def predict(self, serialized_input: str) -> GateScores:
        import torch

        if not isinstance(serialized_input, str) or not serialized_input:
            raise ValueError("serialized_input must be a non-empty string")
        device_type = str(self.device).split(":", 1)[0]
        cache_key = (
            serialized_input, self.max_tokens, str(self.device),
            torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type),
        )
        cache_enabled = not getattr(self.model, "training", False)
        if not cache_enabled:
            self._prediction_cache.clear()
        elif cache_key in self._prediction_cache:
            self._prediction_cache.move_to_end(cache_key)
            return self._prediction_cache[cache_key]
        encoded = self.tokenizer(
            serialized_input,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
            )
        close = float(torch.sigmoid(output.close_logits)[0].cpu())
        suitable = float(torch.sigmoid(output.suitable_logits)[0].cpu())
        scores = GateScores(
            close_probability=close,
            suitable_probability=suitable,
            stable=True,
            precision_delta=0.0,
            decision_agreement=True,
        )
        if cache_enabled:
            self._prediction_cache[cache_key] = scores
            if len(self._prediction_cache) > 4096:
                self._prediction_cache.popitem(last=False)
        return scores


class _FloatRuntimeGateBundle:
    """Load the float gate candidate produced by the current fresh run."""

    def __init__(
        self,
        *,
        root: Path,
        base_model_path: Path,
        bundle_sha256: str,
        device: str = "cpu",
        max_tokens: int = 256,
        window_contract_version: str = WINDOW_CONTRACT_VERSION,
        close_low_threshold: float = 0.45,
        close_high_threshold: float = 0.55,
        suitable_accept_threshold: float = 0.5,
        max_units: int = 3,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        from wfcllm.gate.config import GateTrainConfig
        from wfcllm.gate.model import GateModel

        float_path = root / "gate_float.pt"
        tokenizer_path = root / "tokenizer"
        if not float_path.is_file() or not tokenizer_path.is_dir():
            raise ValueError("gate candidate is incomplete")
        state = torch.load(float_path, map_location="cpu", weights_only=True)
        model = GateModel.from_local_pretrained(
            GateTrainConfig(max_tokens=max_tokens, base_model_path=base_model_path)
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )
        tokenizer_hash = _sha256_directory(tokenizer_path)
        self.root = root
        self.bundle_sha256 = bundle_sha256
        self.tokenizer_sha256 = tokenizer_hash
        self.manifest = _CandidateGateManifest(
            window_contract_version=window_contract_version,
            gate_input_contract_version=GATE_INPUT_CONTRACT_VERSION,
            tokenizer_sha256=tokenizer_hash,
            close_low_threshold=close_low_threshold,
            close_high_threshold=close_high_threshold,
            suitable_accept_threshold=suitable_accept_threshold,
            max_tokens=max_tokens,
            max_units=max_units,
        )
        self.stable_gate_predictor = _FloatGatePredictor(
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            device=device,
        )
        self._tokenizer = tokenizer

    def tokenizer_counter(self, text: str) -> int:
        return _token_count(self._tokenizer, text)


class LocalCandidateRuntimeGateBundle(_FloatRuntimeGateBundle):
    """Gate-train candidate bound by its current-run manifest tree hash."""


class LocalHFProgramGenerator:
    """One local HF model shared by base generation and window rewrites."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        rewrite_max_new_tokens: int,
        program_prompt_mode: str = "completion",
        rewrite_generation_attempts: int = 3,
        rewrite_temperature: float = 0.8,
        rewrite_top_p: float = 0.95,
        load_in_4bit: bool = False,
        torch_dtype: str = "bf16",
    ) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        model_class = (
            AutoModelForSeq2SeqLM
            if bool(getattr(model_config, "is_encoder_decoder", False))
            else AutoModelForCausalLM
        )
        model_kwargs: dict[str, Any] = {"local_files_only": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float16
            model_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map="auto",
            )
        else:
            dtype_by_name = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            if torch_dtype not in dtype_by_name:
                raise ValueError("torch_dtype must be bf16, fp16, or fp32")
            model_kwargs["torch_dtype"] = dtype_by_name[torch_dtype]
        self.model = model_class.from_pretrained(str(model_path), **model_kwargs)
        if not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        if program_prompt_mode not in {"completion", "mbpp_chat", "humanevalpack_chat"}:
            raise ValueError(
                "program_prompt_mode must be completion, mbpp_chat or humanevalpack_chat"
            )
        self.program_prompt_mode = program_prompt_mode
        self.is_encoder_decoder = bool(getattr(model_config, "is_encoder_decoder", False))
        self.rewriter = CausalWindowRewriter(
            HFCausalRewriteBackend(
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                max_new_tokens=rewrite_max_new_tokens,
                temperature=rewrite_temperature,
                top_p=rewrite_top_p,
            ),
            generation_attempts=rewrite_generation_attempts,
        )

    def generate_program(self, *, prompt: str, sample_id: str) -> str:
        import torch

        self.last_program_generation_trace = None
        native_chat = getattr(self, "program_prompt_mode", "completion") == "humanevalpack_chat"
        native_language = None
        if native_chat:
            native_language = {"java": "Java", "javascript": "JavaScript"}.get(sample_id.split("/", 1)[0].casefold())
            if native_language is None:
                raise ValueError("humanevalpack_chat requires a Java/ or JavaScript/ sample ID")
        mbpp_chat = (
            _is_mbpp_sample(sample_id)
            and _is_mbpp_interface_prefix(prompt)
            and getattr(self, "program_prompt_mode", "completion")
            == "mbpp_chat"
        )
        lm_prompt = (
            _native_chat_program_prompt(self.tokenizer, prompt, native_language)
            if native_language is not None
            else _mbpp_chat_program_prompt(self.tokenizer, prompt)
            if mbpp_chat
            else _program_generation_prompt(prompt, sample_id=sample_id)
        )
        encoded = self.tokenizer(lm_prompt, return_tensors="pt", truncation=True)
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in dict(encoded).items()
        }
        digest = hashlib.sha256(f"{self.seed}\0{sample_id}\0{lm_prompt}".encode("utf-8")).digest()
        torch.manual_seed(int.from_bytes(digest[:8], "big"))
        generated = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-6),
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        token_ids = generated[0]
        if not self.is_encoder_decoder:
            token_ids = token_ids[inputs["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not isinstance(completion, str):
            raise ValueError("generation tokenizer decode must return text")
        if native_language is not None:
            self.last_program_generation_trace = {
                "sample_id": sample_id,
                "prompt_mode": "humanevalpack_chat",
                "model_input": lm_prompt,
                "model_response": completion,
                "model_input_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
                "model_continuation_token_ids": token_ids.detach().cpu().tolist(),
            }
            return _extract_native_chat_code(completion, native_language)
        if _is_mbpp_sample(sample_id):
            if _is_mbpp_interface_prefix(prompt) and not mbpp_chat:
                return prompt + completion
            code = _extract_python_code_completion(completion)
            return (
                _normalize_mbpp_chat_interface(prompt, code)
                if mbpp_chat
                else code
            )
        return prompt + completion


def _is_mbpp_sample(sample_id: str) -> bool:
    return sample_id.lower().startswith("mbpp/")


def _native_chat_program_prompt(tokenizer: Any, prompt: str, language: str) -> str:
    if language not in {"Java", "JavaScript"}:
        raise ValueError("native chat requires Java or JavaScript")
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError("humanevalpack_chat requires a tokenizer chat template")
    instruction = (
        f"Return one complete {language} implementation of the public code prefix below. "
        "Preserve its exact function, class and parameter names and interface. "
        "Include required imports and helper definitions. Return code only. "
        "Do not include Markdown, explanations, examples, tests, printing, input calls, "
        "or a main function. Do not add unrelated declarations.\n\n" + prompt
    )
    if language == "Java":
        instruction = "Preserve the class modifiers from the prefix; do not add public to its class. " + instruction
    rendered = apply_template([{"role": "user", "content": instruction}],
                              tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template returned an invalid prompt")
    return rendered


def _extract_native_chat_code(text: str, language: str) -> str:
    """Unwrap the first native code fence, without executing or repairing code."""
    if language not in {"Java", "JavaScript"}:
        raise ValueError("native chat requires Java or JavaScript")
    labels = {"", "java"} if language == "Java" else {"", "javascript", "js"}
    fence = re.search(r"(?m)^[ \t]*```([A-Za-z0-9_-]*)[ \t]*\r?\n", text)
    if fence is not None and fence.group(1).casefold() in labels:
        content = text[fence.end():]
        closing = re.search(r"(?m)^[ \t]*```[ \t]*$", content)
        text = content[:closing.start()] if closing is not None else content
    stripped = text.strip()
    return stripped + ("\n" if stripped else "")


def _program_generation_prompt(prompt: str, *, sample_id: str) -> str:
    if not _is_mbpp_sample(sample_id):
        return prompt
    if _is_mbpp_interface_prefix(prompt):
        return prompt
    return (
        "Write executable Python code only. Return the result; do not print it. "
        "Do not include Markdown, prose, tests, print calls, or input calls.\n\n"
        f"Task:\n{prompt}\n\nPython code:\n"
    )


def _mbpp_chat_program_prompt(tokenizer: Any, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("mbpp_chat requires a tokenizer chat template")
    user_prompt = (
        "Return one complete executable Python program that implements the "
        "function prefix below. Preserve the exact function and parameter "
        "names. Return the result instead of printing it. Do not include "
        "Markdown, explanations, examples, tests, print calls, or input "
        "calls. Use at least four semantically meaningful statements, "
        "excluding docstrings and imports. Intermediate variables must "
        "contribute to the returned value. Do not use a one-line "
        "implementation; do not add dead code.\n\n"
        f"{prompt}"
    )
    rendered = apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template returned an invalid prompt")
    return rendered


def _normalize_mbpp_chat_interface(prompt: str, code: str) -> str:
    """Restore the sanitized prompt interface without consulting MBPP tests."""

    try:
        prompt_tree = ast.parse(prompt)
        code_tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return code
    expected_functions = [
        node
        for node in prompt_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    actual_functions = [
        node
        for node in code_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not expected_functions or not actual_functions:
        return code
    expected = expected_functions[-1]
    same_name = [node for node in actual_functions if node.name == expected.name]
    if same_name:
        actual = same_name[0]
    else:
        expected_arity = len(expected.args.posonlyargs) + len(expected.args.args)
        same_arity = [
            node
            for node in actual_functions
            if type(node) is type(expected)
            and node.args.vararg is None
            and len(node.args.posonlyargs) + len(node.args.args)
            == expected_arity
        ]
        if len(same_arity) != 1:
            return code
        actual = same_arity[0]

    actual_parameters = actual.args.posonlyargs + actual.args.args
    expected_parameters = expected.args.posonlyargs + expected.args.args
    if (
        len(actual_parameters) > len(expected_parameters)
        or actual.args.vararg is not None
        or actual.args.kwarg is not None
        or actual.args.kwonlyargs
        or expected.args.vararg is not None
        or expected.args.kwarg is not None
        or expected.args.kwonlyargs
    ):
        return code
    if len(actual_parameters) < len(expected_parameters) and any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {actual.name, expected.name}
        for node in ast.walk(actual)
    ):
        # Additional required public parameters would break recursive calls.
        return code
    parameter_bindings = [
        (old.arg, new.arg)
        for old, new in zip(
            actual_parameters, expected_parameters[:len(actual_parameters)], strict=True
        )
        if old.arg != new.arg
    ]
    old_name = actual.name
    actual.name = expected.name
    actual.args = deepcopy(expected.args)
    if old_name != expected.name:
        class _RenameRecursiveCall(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == old_name:
                    return ast.copy_location(
                        ast.Name(id=expected.name, ctx=node.ctx),
                        node,
                    )
                return node

        _RenameRecursiveCall().visit(actual)

    expected_docstring = (
        expected.body[0]
        if expected.body
        and isinstance(expected.body[0], ast.Expr)
        and isinstance(expected.body[0].value, ast.Constant)
        and isinstance(expected.body[0].value.value, str)
        else None
    )
    actual_has_docstring = bool(
        actual.body
        and isinstance(actual.body[0], ast.Expr)
        and isinstance(actual.body[0].value, ast.Constant)
        and isinstance(actual.body[0].value.value, str)
    )
    if expected_docstring is not None:
        if actual_has_docstring:
            actual.body[0] = deepcopy(expected_docstring)
        else:
            actual.body.insert(0, deepcopy(expected_docstring))
    if parameter_bindings:
        # Restore generated local names simultaneously, including swaps.  Keep
        # nested scopes untouched instead of renaming their bound identifiers.
        aliases = ast.Assign(
            targets=[ast.Tuple(
                elts=[ast.Name(id=old, ctx=ast.Store()) for old, _ in parameter_bindings],
                ctx=ast.Store(),
            )],
            value=ast.Tuple(
                elts=[ast.Name(id=new, ctx=ast.Load()) for _, new in parameter_bindings],
                ctx=ast.Load(),
            ),
        )
        actual.body.insert(int(expected_docstring is not None or actual_has_docstring), aliases)
    ast.fix_missing_locations(code_tree)
    try:
        return ast.unparse(code_tree).rstrip() + "\n"
    except (RecursionError, TypeError, ValueError):
        return code


def _is_mbpp_interface_prefix(prompt: str) -> bool:
    return bool(
        re.search(r"(?m)^def\s+[A-Za-z_]\w*\s*\(", prompt)
        and prompt.lstrip().startswith(("class ", "def "))
    )


def _extract_python_code_completion(text: str) -> str:
    fenced = _first_python_fence(text)
    if fenced is not None:
        return _trim_to_compilable_python(fenced)
    cleaned = text.replace("<jupyter>", "")
    lines = cleaned.splitlines()
    start = _first_python_code_line(lines)
    if start is None:
        return cleaned.strip() + ("\n" if cleaned.strip() else "")
    return _trim_to_compilable_python("\n".join(lines[start:]))


def _first_python_fence(text: str) -> str | None:
    match = re.search(
        r"```(?:python|py)?\s*\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else None


def _first_python_code_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:from\s+\S+\s+import\s+|import\s+|def\s+|class\s+|@)", line):
            return index
    return None


def _trim_to_compilable_python(text: str) -> str:
    lines: list[str] = []
    for line in text.replace("<jupyter>", "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            break
        if line.startswith(
            (
                "This function",
                "You can",
                "The ",
                "In this",
                "Explanation:",
                "Output:",
                "Input:",
            )
        ):
            break
        lines.append(line)

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            compile(candidate + "\n", "<wfcllm-mbpp-final-code>", "exec")
        except SyntaxError:
            continue
        return candidate + "\n"
    return ("\n".join(lines).strip() + "\n") if lines else ""


def build_local_semantic_window_scorer(
    options: LocalHFGateRuntimeOptions, deployment_key: bytes
):
    from wfcllm.semantic.window_lsh import SemanticWindowScorer

    components = _load_public_semantic_components(options)
    return SemanticWindowScorer(
        verifier=components.verifier,
        keying=WatermarkKeying(deployment_key.hex(), options.lsh_dimension),
        contract_version=options.window_contract_version,
        k=max(1, round(options.lsh_gamma * (2 ** options.lsh_dimension))),
        margin=float(options.semantic_margin),
        semantic_preservation_threshold=(
            options.semantic_preservation_threshold
        ),
    )


def local_semantic_runtime_hash(options: LocalHFGateRuntimeOptions) -> str:
    digest = hashlib.sha256(
        b"wfcllm-local-semantic-runtime/v2\0"
        + _PUBLIC_SEMANTIC_RUNTIME_VERSION.encode("utf-8")
        + _PUBLIC_SEMANTIC_INITIALIZATION_SEED.to_bytes(8, "big")
        + options.semantic_embed_dim.to_bytes(4, "big")
        + options.lsh_dimension.to_bytes(4, "big")
        + repr(float(options.lsh_gamma)).encode("ascii")
        + b"\0"
        + repr(float(options.semantic_margin)).encode("ascii")
        + b"\0"
        + options.semantic_evidence_rule.encode("utf-8")
        + b"\0"
        + options.window_contract_version.encode("utf-8")
        + repr(float(options.semantic_preservation_threshold)).encode("ascii")
        + b"\0"
    )
    for path in (
        options.semantic_encoder_model_path,
        options.semantic_encoder_checkpoint_path,
    ):
        if path.is_file():
            digest.update(path.name.encode("utf-8") + bytes.fromhex(_sha256_file(path)))
        elif path.is_dir():
            for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
                if child.is_file():
                    relative = child.relative_to(path).as_posix().encode("utf-8")
                    digest.update(relative + bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def load_local_causal_rewriter(options: LocalHFGateRuntimeOptions):
    return _load_causal_rewriter(options)


def _validate_cache_against_data(
    cache_rows: list[dict[str, Any]], data_dir: Path
) -> None:
    window_lengths = _gate_data_window_lengths(data_dir)
    label_context_length = window_lengths[-1]
    configured_budgets = _gate_data_rewrite_budgets(data_dir)
    index_path = data_dir / "group_index.jsonl"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("gate data group index is missing")
    indexed: dict[str, dict[str, Any]] = {}
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"gate group index line {line_number} is invalid") from exc
            group_id = row.get("group_id") if isinstance(row, Mapping) else None
            if not isinstance(group_id, str) or group_id in indexed:
                raise ValueError("gate group index identity is invalid")
            indexed[group_id] = dict(row)
    cache_ids = {row.get("group_id") for row in cache_rows}
    if cache_ids != set(indexed):
        raise ValueError("local gate cache group set does not match formal gate data")
    variants: set[tuple[str, int, int]] = set()
    group_variants: dict[str, set[tuple[int, int]]] = {
        group_id: set() for group_id in indexed
    }
    for row in cache_rows:
        group_id = row["group_id"]
        context_length = row.get("context_length")
        budget = row.get("budget")
        if context_length not in window_lengths:
            raise ValueError("local gate cache context_length contradicts gate data")
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or budget < 0
            or budget > 48
        ):
            raise ValueError("local gate cache budget is invalid")
        variant = (group_id, context_length, budget)
        if variant in variants:
            raise ValueError("local gate cache contains a duplicate variant")
        variants.add(variant)
        group_variants[group_id].add((context_length, budget))
        index = indexed[group_id]
        if row.get("split") != index.get("split"):
            raise ValueError("local gate cache split contradicts formal gate data")
    observed_budget_sets = {
        tuple(sorted({budget for _length, budget in variants_for_group}))
        for variants_for_group in group_variants.values()
    }
    if len(observed_budget_sets) != 1:
        raise ValueError("local gate cache budget set contradicts gate data")
    observed_budgets = observed_budget_sets.pop()
    rewrite_budgets = configured_budgets or observed_budgets
    expected_variants = {
        (length, budget) for length in window_lengths for budget in rewrite_budgets
    }
    if any(
        variants_for_group != expected_variants
        for variants_for_group in group_variants.values()
    ):
        raise ValueError(
            "local gate cache variant count does not match gate data window_lengths"
        )
    label_budget = rewrite_budgets[-1]
    for row in cache_rows:
        index = indexed[row["group_id"]]
        if (
            row.get("context_length") == label_context_length
            and row.get("budget") == label_budget
            and (
                row.get("close_target") != index.get("close_target")
                or row.get("suitable_target") != index.get("suitable_target")
            )
        ):
            raise ValueError("local gate cache label contradicts formal gate data")


def _gate_data_window_lengths(data_dir: Path) -> tuple[int, ...]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return (1, 2, 3)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("gate data manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("gate data manifest must be a JSON object")
    raw_lengths = manifest.get("window_lengths")
    if raw_lengths not in ([1], [1, 2], [1, 2, 3]):
        raise ValueError("gate data manifest window_lengths is invalid")
    return tuple(raw_lengths)


def _gate_data_rewrite_budgets(data_dir: Path) -> tuple[int, ...] | None:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("gate data manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("gate data manifest must be a JSON object")
    raw_budgets = manifest.get("rewrite_budgets")
    if raw_budgets is None:
        return None
    if (
        not isinstance(raw_budgets, list)
        or not raw_budgets
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 48
            for value in raw_budgets
        )
        or raw_budgets != sorted(raw_budgets)
        or len(set(raw_budgets)) != len(raw_budgets)
    ):
        raise ValueError("gate data manifest rewrite_budgets is invalid")
    return tuple(raw_budgets)


def _partition_training_examples(
    examples: tuple[Any, ...],
    cache_rows: list[dict[str, Any]],
) -> tuple[tuple[Any, ...], tuple[Any, ...], str]:
    if len(examples) != len(cache_rows):
        raise ValueError("gate examples and cache rows must have equal length")
    paired = tuple(zip(examples, cache_rows, strict=True))
    training = tuple(
        example for example, row in paired if row.get("split") == "train"
    )
    validation = tuple(
        example for example, row in paired if row.get("split") == "validation"
    )
    validation_role = "validation"
    return training, validation, validation_role


@dataclass(frozen=True)
class _ParsedCatalogSource:
    record: GateSourceCatalogRecord
    units: tuple[Any, ...]


@dataclass(frozen=True)
class _CatalogWindowStart:
    parsed: _ParsedCatalogSource
    start_unit_id: str


def _select_source_stratified_starts(
    parsed_sources: Sequence[_ParsedCatalogSource],
    *,
    max_groups: int | None,
) -> tuple[_CatalogWindowStart, ...]:
    """Select legal W1/W2/W3 starts with deterministic source/repo coverage."""

    if not isinstance(parsed_sources, Sequence) or any(
        not isinstance(parsed, _ParsedCatalogSource) for parsed in parsed_sources
    ):
        raise ValueError("parsed source catalog has an invalid runtime shape")
    if max_groups is not None and (type(max_groups) is not int or max_groups <= 0):
        raise ValueError("max_groups must be a positive integer or None")

    source_queues: list[tuple[_ParsedCatalogSource, tuple[str, ...]]] = []
    for parsed in parsed_sources:
        legal = complete_w1_w2_w3_start_unit_ids(parsed.units)
        if not legal:
            continue
        ordered = tuple(
            sorted(
                legal,
                key=lambda unit_id: _selection_order_key(
                    "window", parsed.record.source_id, unit_id
                ),
            )
        )
        source_queues.append((parsed, ordered))

    by_repository: dict[
        str, list[tuple[_ParsedCatalogSource, tuple[str, ...]]]
    ] = defaultdict(list)
    for parsed, starts in source_queues:
        record = parsed.record
        repository = (
            record.repository_id
            or record.task_id
            or record.function_id
            or record.source_id
        )
        by_repository[repository].append((parsed, starts))
    for repository, sources in by_repository.items():
        sources.sort(
            key=lambda item: _selection_order_key(
                "source", repository, item[0].record.source_id
            )
        )

    repositories = sorted(
        by_repository,
        key=lambda repository: _selection_order_key("repository", repository),
    )
    balanced_sources: list[tuple[_ParsedCatalogSource, tuple[str, ...]]] = []
    max_sources_per_repository = max(
        (len(by_repository[repository]) for repository in repositories),
        default=0,
    )
    for source_offset in range(max_sources_per_repository):
        for repository in repositories:
            sources = by_repository[repository]
            if source_offset < len(sources):
                balanced_sources.append(sources[source_offset])

    limit = sum(len(starts) for _parsed, starts in balanced_sources)
    if max_groups is not None:
        limit = min(limit, max_groups)
    selected: list[_CatalogWindowStart] = []
    max_starts_per_source = max(
        (len(starts) for _parsed, starts in balanced_sources),
        default=0,
    )
    for start_offset in range(max_starts_per_source):
        for parsed, starts in balanced_sources:
            if start_offset < len(starts):
                selected.append(_CatalogWindowStart(parsed, starts[start_offset]))
                if len(selected) == limit:
                    return tuple(selected)
    return tuple(selected)


def _selection_order_key(*parts: str) -> bytes:
    digest = hashlib.sha256(b"wfcllm-source-stratified-window-selection/v2\0")
    for part in parts:
        digest.update(part.encode("utf-8") + b"\0")
    return digest.digest()


def _source_stratified_selection_summary(
    parsed_sources: Sequence[_ParsedCatalogSource],
    selected: Sequence[_CatalogWindowStart],
    *,
    max_groups: int | None,
    language: str = "python",
) -> dict[str, Any]:
    candidate_sources: set[str] = set()
    candidate_repositories: set[str] = set()
    candidate_window_count = 0
    for parsed in parsed_sources:
        count = len(complete_w1_w2_w3_start_unit_ids(parsed.units))
        if not count:
            continue
        candidate_window_count += count
        record = parsed.record
        candidate_sources.add(record.source_id)
        candidate_repositories.add(
            record.repository_id
            or record.task_id
            or record.function_id
            or record.source_id
        )

    selected_sources = {item.parsed.record.source_id for item in selected}
    selected_repositories = {
        item.parsed.record.repository_id
        or item.parsed.record.task_id
        or item.parsed.record.function_id
        or item.parsed.record.source_id
        for item in selected
    }
    selected_tasks = {
        item.parsed.record.task_id
        for item in selected
        if item.parsed.record.task_id is not None
    }
    selected_functions = {
        item.parsed.record.function_id
        for item in selected
        if item.parsed.record.function_id is not None
    }
    per_source = Counter(item.parsed.record.source_id for item in selected)
    statement_families: Counter[str] = Counter()
    selection_records: list[dict[str, str]] = []
    for item in selected:
        unit_by_id = {unit.unit_id: unit for unit in item.parsed.units}
        statement_families[
            _statement_family(
                unit_by_id[item.start_unit_id].text,
                cpp_node_type=(unit_by_id[item.start_unit_id].node_type if language == "cpp" else None),
            )
        ] += 1
        record = item.parsed.record
        selection_records.append(
            {
                "source_id": record.source_id,
                "repository_id": record.repository_id or "",
                "task_id": record.task_id or "",
                "function_id": record.function_id or "",
                "start_unit_id": item.start_unit_id,
            }
        )
    algorithm_version = "wfcllm-source-stratified-window-selection/v2"
    selection_sha256 = hashlib.sha256(
        json.dumps(
            selection_records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selection_config_sha256 = _canonical_hash(
        {
            "algorithm_version": algorithm_version,
            "max_groups": max_groups,
        }
    )
    return {
        "algorithm_version": algorithm_version,
        "selection_config_sha256": selection_config_sha256,
        "selection_sha256": selection_sha256,
        "candidate_window_count": candidate_window_count,
        "candidate_source_count": len(candidate_sources),
        "candidate_repository_count": len(candidate_repositories),
        "selected_group_count": len(selected),
        "selected_source_count": len(selected_sources),
        "selected_repository_count": len(selected_repositories),
        "selected_task_count": len(selected_tasks),
        "selected_function_count": len(selected_functions),
        "max_selected_per_source": max(per_source.values(), default=0),
        "statement_family_counts": dict(sorted(statement_families.items())),
    }


class _StructuralOnlyProbe:
    """Key-independent placeholder replaced before labels are computed."""

    semantic_probe_pending = True

    def probe(self, *, window_text: str, parent_descriptor: str, key_ids: tuple[str, ...]):
        return {
            key_id: LshProbeResult((0, 0, 0, 0), 0.0, False, False, False, False)
            for key_id in key_ids
        }


def _statement_family(text: str, *, cpp_node_type: str | None = None) -> str:
    """Classify public statement syntax without consulting labels or keys."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("statement text must be a non-empty string")
    if cpp_node_type is not None:
        return _cpp_statement_family(text, cpp_node_type)
    try:
        statement = ast.parse(textwrap.dedent(text).strip()).body[0]
    except (IndentationError, SyntaxError, IndexError):
        return "other"
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        value = getattr(statement, "value", None)
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call):
            return "assignment_call"
        if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
            return "assignment_reference"
        if isinstance(
            value,
            (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict),
        ):
            return "assignment_literal"
        return "assignment_expression"
    if isinstance(statement, ast.Expr):
        value = statement.value
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call):
            return "call"
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return "docstring"
        return "expression"
    if isinstance(statement, (ast.If, ast.Match)):
        return "branch"
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        return "loop"
    if isinstance(statement, ast.Return):
        return "return"
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return "import"
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "definition"
    if isinstance(statement, ast.Raise):
        return "raise"
    return "control"


def _cpp_statement_family(text: str, node_type: str) -> str:
    from wfcllm.lang.cpp.parser import CppParser

    if node_type in {"if_statement", "switch_statement"}:
        return "branch"
    if node_type in {"for_statement", "for_range_loop", "while_statement", "do_statement"}:
        return "loop"
    if node_type in {"return_statement", "co_return_statement"}:
        return "return"
    if node_type == "throw_statement":
        return "raise"
    if node_type in {"break_statement", "continue_statement", "goto_statement", "try_statement"}:
        return "control"
    root = CppParser().parse(text).root_node
    statements = [node for node in root.named_children if node.type != "comment"]
    if root.has_error or len(statements) != 1:
        return "other"
    statement = statements[0]
    values = []
    if statement.type == "declaration":
        values = [value for child in statement.named_children if child.type == "init_declarator"
                  if (value := child.child_by_field_name("value")) is not None]
        if not values:
            return "declaration"
    elif statement.type == "expression_statement" and statement.named_children:
        expression = statement.named_children[0]
        if expression.type == "assignment_expression":
            value = expression.child_by_field_name("right")
            if value is not None:
                values = [value]
        elif expression.type == "update_expression":
            return "assignment_expression"
        elif expression.type == "call_expression":
            return "call"
        else:
            return "expression"
    if not values:
        return "other"
    types = {value.type for value in values}
    if "call_expression" in types:
        return "assignment_call"
    if types <= {"identifier", "qualified_identifier", "field_expression", "subscript_expression"}:
        return "assignment_reference"
    if types <= {"number_literal", "char_literal", "string_literal", "concatenated_string", "true", "false", "nullptr"}:
        return "assignment_literal"
    return "assignment_expression"


def _record_split_group_id(record: GateSourceCatalogRecord) -> str:
    if record.repository_id is not None:
        return "repository:" + canonical_gate_source_identity(record.repository_id)
    if record.task_id is not None:
        return "task:" + canonical_gate_source_identity(record.task_id)
    if record.function_id is not None:
        return "function:" + canonical_gate_source_identity(record.function_id)
    raise ValueError("source catalog record has no split identity")


def _balanced_split_assignments(group_ids: Sequence[str]) -> dict[str, str]:
    """Assign whole source groups with deterministic 60/20/20 coverage."""

    unique = set(group_ids)
    if any(not isinstance(group_id, str) or not group_id for group_id in unique):
        raise ValueError("split group IDs must be non-empty strings")
    ordered = sorted(
        unique,
        key=lambda group_id: _selection_order_key("split", group_id),
    )
    if len(ordered) < 3:
        return {group_id: "train" for group_id in ordered}
    holdout_count = max(1, (len(ordered) + 4) // 5)
    holdout_count = min(holdout_count, (len(ordered) - 1) // 2)
    validation = set(ordered[:holdout_count])
    test = set(ordered[holdout_count : 2 * holdout_count])
    return {
        group_id: (
            "validation"
            if group_id in validation
            else "test"
            if group_id in test
            else "train"
        )
        for group_id in ordered
    }


def _split_for(group_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(group_id.encode("utf-8")).digest()[:8], "big") % 10
    return "validation" if bucket == 0 else "test" if bucket == 1 else "train"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("hash input must be a non-symlink directory")
    digest = hashlib.sha256()
    found_file = False
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ValueError("hash input directory cannot contain symlinks")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_dir():
            digest.update(b"D")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            continue
        if not path.is_file():
            raise ValueError("hash input directory contains an unsupported entry")
        found_file = True
        digest.update(b"F")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    if not found_file:
        raise ValueError("hash input directory must contain at least one file")
    return digest.hexdigest()


def _load_causal_rewriter(options: LocalHFGateRuntimeOptions):
    language = language_for_window_contract(options.window_contract_version)
    if options.rewrite_strategy == "java_structural_equivalent":
        from wfcllm.generation.java_structural_rewriter import KeyBlindJavaStructuralWindowRewriter

        return KeyBlindJavaStructuralWindowRewriter(
            extractor=get_statement_unit_extractor(language),
            window_contract_version=options.window_contract_version,
        )
    if options.rewrite_strategy == "js_structural_equivalent":
        from wfcllm.generation.js_structural_rewriter import KeyBlindJavaScriptStructuralWindowRewriter

        return KeyBlindJavaScriptStructuralWindowRewriter(
            extractor=get_statement_unit_extractor(language),
            window_contract_version=options.window_contract_version,
        )
    if options.rewrite_strategy == "cpp_structural_equivalent":
        from wfcllm.generation.cpp_structural_rewriter import KeyBlindCppStructuralWindowRewriter

        return KeyBlindCppStructuralWindowRewriter(
            extractor=get_statement_unit_extractor(language),
            window_contract_version=options.window_contract_version,
        )
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    path = options.effective_rewrite_model_path
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    model_class = AutoModelForSeq2SeqLM if bool(getattr(config, "is_encoder_decoder", False)) else AutoModelForCausalLM
    model = model_class.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    model.to(options.model_device)
    model.eval()
    return CausalWindowRewriter(
        HFCausalRewriteBackend(
            model=model,
            tokenizer=tokenizer,
            device=options.model_device,
            max_new_tokens=options.rewrite_max_new_tokens,
            temperature=options.rewrite_temperature,
            top_p=options.rewrite_top_p,
            language=language,
        ),
        extractor=get_statement_unit_extractor(language),
        generation_attempts=options.rewrite_generation_attempts,
        window_contract_version=options.window_contract_version,
    )


def _load_semantic_runtime(options: LocalHFGateRuntimeOptions):
    components = _load_public_semantic_components(options)
    return LocalSemanticRuntime(components.verifier)


def _load_public_semantic_components(options: LocalHFGateRuntimeOptions):
    import torch

    from wfcllm.semantic.lsh import load_semantic_lsh_components

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_PUBLIC_SEMANTIC_INITIALIZATION_SEED)
        return load_semantic_lsh_components(
            encoder_model_path=str(options.semantic_encoder_model_path),
            encoder_checkpoint_path=str(options.semantic_encoder_checkpoint_path),
            embed_dim=options.semantic_embed_dim,
            device=options.model_device,
            use_lora=False,
            use_bf16=False,
            secret_key="wfcllm-public-window-plane/v1",
            lsh_d=options.lsh_dimension,
        )


__all__ = [
    "LOCAL_HF_ADAPTER_NAME",
    "GateSourceCatalogRecord",
    "LocalHFGateRuntimeOptions",
    "LocalHFProductionAdapter",
    "HFCausalRewriteBackend",
    "LocalHFProgramGenerator",
    "LocalCandidateRuntimeGateBundle",
    "LocalSemanticRuntime",
    "build_local_semantic_window_scorer",
    "experiment_contract_hash",
    "load_source_catalog",
    "load_local_causal_rewriter",
    "local_semantic_runtime_hash",
    "phase_config_hash",
]
