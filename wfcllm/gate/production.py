"""Single-machine, local-only production runtime for gated experiments."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
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
    complete_w1_w2_w3_start_unit_ids,
)
from wfcllm.generation.window_rewriter import (
    CausalWindowRewriter,
    KeyBlindAstEquivalentWindowRewriter,
    KeyBlindCppEquivalentWindowRewriter,
    RewriteGeneration,
)
from wfcllm.gate.labels import LabelThresholds, build_gate_labels
from wfcllm.gate.pipeline import GatePipelineGroup, ValidationOutcome
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
    semantic_encoder_checkpoint_path: Path | None
    semantic_whitening_path: Path | None
    gate_base_model_path: Path
    model_device: str = "cuda"
    gate_device: str = "cuda"
    cache_dir: Path = Path("data/gate-cache")
    semantic_embed_dim: int = 128
    lsh_dimension: int = 4
    lsh_gamma: float = 0.25
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
    gate_resume_checkpoint: Path | None = None
    window_contract_version: str = WINDOW_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_catalog",
            "generation_model_path",
            "semantic_encoder_model_path",
            "gate_base_model_path",
            "cache_dir",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a pathlib.Path")
        for name in (
            "rewrite_model_path",
            "semantic_encoder_checkpoint_path",
            "semantic_whitening_path",
            "gate_resume_checkpoint",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise ValueError(f"{name} must be a pathlib.Path or None")
        for name in ("model_device", "gate_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.semantic_evidence_rule not in {
            "semantic_lsh",
            "keyed_text_region",
        }:
            raise ValueError("unsupported semantic evidence rule")
        if not is_supported_window_contract(self.window_contract_version):
            raise ValueError("unsupported window contract version")
        if (
            isinstance(self.lsh_gamma, bool)
            or not isinstance(self.lsh_gamma, (int, float))
            or not 0.0 < self.lsh_gamma < 1.0
        ):
            raise ValueError("lsh_gamma must be in (0, 1)")
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
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gate_batch_size < 3:
            raise ValueError("gate_batch_size must be at least 3")
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
            self.gate_base_model_path,
        )
        for path in required:
            if path.is_symlink() or not path.exists():
                raise ValueError(f"local runtime resource is missing or unsafe: {path}")

    @property
    def effective_rewrite_model_path(self) -> Path:
        return self.rewrite_model_path or self.generation_model_path


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
            "validate_candidate",
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
            if options.semantic_evidence_rule == "semantic_lsh"
            and language == "python"
            else KeyBlindCppEquivalentWindowRewriter(
                extractor=extractor,
                window_contract_version=options.window_contract_version,
            )
            if options.semantic_evidence_rule == "semantic_lsh"
            and language == "cpp"
            else None
        )
        self._semantic_runtime: object | None = None
        self._parent_by_group: dict[str, str] = {}
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
        if rewriter is None:
            rewriter = _load_causal_rewriter(self.options)
            self._rewriter = rewriter
        builder = GateDataBuilder(rewriter=rewriter, lsh_probe=_StructuralOnlyProbe())
        self.options.cache_dir.mkdir(parents=True, exist_ok=True)
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
        )
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
            )
            for built in groups:
                candidates = built.candidates_by_length
                if set(candidates) != {"1", "2", "3"}:
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
                        for length in (1, 2, 3)
                    },
                    "current_unit_types": {
                        str(length): [
                            unit.node_type for unit in parsed.units[start_index : start_index + length]
                        ]
                        for length in (1, 2, 3)
                    },
                    "source_family": record.source_family,
                }
                yield GatePipelineGroup(
                    group_id=built.group_id,
                    split_group_id=built.split_group_id,
                    split=split,
                    suitable_target=False,
                    close_target=False,
                    window_lengths=(1, 2, 3),
                    statement_family=_statement_family(start.text),
                    r1_success_rate=0.0,
                    r3_success_rate=0.0,
                    holdout_success_rate=0.0,
                    repository_id=record.repository_id or record.task_id or record.function_id or record.source_id,
                    task_id=record.task_id or record.function_id or record.source_id,
                    generation_model_id=(
                        record.source_model_id
                        or self.options.effective_rewrite_model_path.name
                        or "local-hf-rewriter"
                    ),
                    structural_invalid_rate=0.0,
                    numeric_instability_rate=0.0,
                    first_hit_candidate_position=None,
                    candidate_indices_by_window_length={length: tuple(range(4)) for length in (1, 2, 3)},
                    observed_training_key_ids=(),
                    observed_holdout_key_ids=(),
                    candidate_observations_by_length=candidates,
                    probe_results_by_length={str(length): tuple({} for _ in range(4)) for length in (1, 2, 3)},
                    row={
                        "schema_version": "wfcllm-gate-data/v1",
                        "group_id": built.group_id,
                        "split": split,
                    },
                )

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
            for length in (1, 2, 3):
                observations: list[CandidateObservation] = []
                candidate_results: list[Mapping[str, LshProbeResult]] = []
                trajectory = group.candidate_observations_by_length[str(length)]
                reference_text = trajectory[0].code
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
                    semantic_cosine = runtime.semantic_reference_cosine(
                        reference_text, observation.code
                    )
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
                        observation.code
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

            label_thresholds = (
                LabelThresholds(r3_hit_rate=0.25)
                if config.fast_experimental
                else None
            )
            selected = build_gate_labels(
                observations_by_length["3"],
                training_key_count=32,
                thresholds=label_thresholds,
            )
            r1, r3 = (selected.budgets[budget].success_rate for budget in (1, 3))
            holdout_hits = sum(
                any(
                    key_id in results_by_length["3"][candidate]
                    and results_by_length["3"][candidate][key_id].is_reliable_hit(
                        configured_margin=0.0
                    )
                    for candidate in range(4)
                )
                for key_id in holdout_keys.key_ids
            )
            first_hits = [
                value for value in selected.budgets[3].first_hit_by_key_id.values() if value is not None
            ]
            result = replace(
                group,
                suitable_target=selected.suitable_target,
                close_target=selected.close_target,
                r1_success_rate=r1,
                r3_success_rate=r3,
                holdout_success_rate=holdout_hits / len(holdout_keys.key_ids),
                structural_invalid_rate=1.0 - selected.budgets[3].structural_valid_rate,
                numeric_instability_rate=selected.budgets[3].unstable_rate,
                first_hit_candidate_position=min(first_hits) if first_hits else None,
                observed_training_key_ids=tuple(training_keys.key_ids),
                observed_holdout_key_ids=tuple(holdout_keys.key_ids),
                candidate_observations_by_length=observations_by_length,
                probe_results_by_length=results_by_length,
            )
            self._append_training_examples(
                result,
                observations_by_length,
                fast_experimental=config.fast_experimental,
            )
            yield result

    def split_groups(self, groups, config):
        yield from groups

    def audit_gate_data(self, staging_dir, manifest):
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("formal_eligible") is not True
            or manifest.get("diagnostic_test_backend") is not False
        ):
            raise ValueError("local-hf-v1 requires a formal gate-data manifest")
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
            fast_experimental=config.fast_experimental,
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
                save_checkpoints=not config.fast_experimental,
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
        summary = trainer.fit(
            training,
            validation,
            resume_from=self.options.gate_resume_checkpoint,
        )
        if config.fast_experimental:
            state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }
        else:
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
        if config.fast_experimental:
            (output_dir / "runtime_thresholds.json").write_text(
                json.dumps(
                    {
                        "schema_version": _UNVALIDATED_RUNTIME_SCHEMA,
                        "close_low_threshold": 0.0,
                        "close_high_threshold": 0.000001,
                        "suitable_accept_threshold": 0.0,
                        "max_units": 3,
                        "runtime_profile": "fast-experimental-low-threshold/v1",
                    },
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        shutil.rmtree(work)
        return {
            "backend": LOCAL_HF_ADAPTER_NAME,
            "best_epoch": summary["best_epoch"],
            "epochs_completed": summary["epochs_completed"],
            "training_example_count": len(training),
            "validation_example_count": len(validation),
            "validation_split_role": validation_role,
            "candidate_sha256": _sha256_file(output_dir / "gate_float.pt"),
        }

    def validate_candidate(
        self,
        *,
        config,
        candidate_bundle,
        data_manifest,
        threshold_fit_group_ids,
        agreement_group_ids,
        output_dir,
    ):
        import torch

        from wfcllm.gate.bundle import (
            GateBundle,
            quantize_gate_model_dynamic,
            sha256_file,
        )
        from wfcllm.gate.config import GateTrainConfig, GateValidateConfig
        from wfcllm.gate.model import GateModel
        from wfcllm.gate.validation import (
            GateValidationArtifacts,
            GateValidator,
        )

        rows = self._load_training_cache(config.config_hash)
        _validate_cache_against_data(rows, config.data_dir)
        fit_ids = set(threshold_fit_group_ids)
        agreement_ids = set(agreement_group_ids)
        fit = tuple(
            _validation_example_from_cache(row, "threshold_fit")
            for row in rows
            if row["group_id"] in fit_ids and row["budget"] == 3
        )
        agreement = tuple(
            _validation_example_from_cache(row, "agreement")
            for row in rows
            if row["group_id"] in agreement_ids and row["budget"] == 3
        )
        if not fit or not agreement:
            raise ValueError("gate validation cache does not cover both holdout subsets")
        float_path = candidate_bundle / "gate_float.pt"
        float_state = torch.load(float_path, map_location="cpu", weights_only=True)
        model = GateModel.from_local_pretrained(
            GateTrainConfig(base_model_path=self.options.gate_base_model_path)
        ).cpu().eval()
        model.load_state_dict(float_state, strict=True)
        quantized = quantize_gate_model_dynamic(model)
        inputs = output_dir / "_validation_inputs"
        inputs.mkdir()
        int8_path = inputs / "gate_int8.pt"
        torch.save(quantized.state_dict(), int8_path)
        artifacts = GateValidationArtifacts(
            float_model_path=float_path,
            int8_model_path=int8_path,
            float_model_sha256=sha256_file(float_path),
            int8_model_sha256=sha256_file(int8_path),
        )
        factory = _LocalGatePredictorFactory(
            base_model_path=self.options.gate_base_model_path,
            tokenizer_path=candidate_bundle / "tokenizer",
            artifacts=artifacts,
        )
        summary = GateValidator(GateValidateConfig()).validate(
            threshold_fit_examples=fit,
            agreement_examples=agreement,
            max_tokens=512,
            predictor_factory=factory,
            gpu_available=torch.cuda.is_available(),
        )
        if not summary.validated:
            shutil.rmtree(inputs)
            return ValidationOutcome(validated=False, summary=summary, bundle=None)
        bundle = GateBundle.create(
            root=output_dir / "bundle",
            validated_float_artifact=float_path,
            validated_int8_artifact=int8_path,
            tokenizer_source=candidate_bundle / "tokenizer",
            validation_summary=summary.to_dict(),
            model_architecture="local-transformer-dual-head",
            base_model_id=self.options.gate_base_model_path.as_posix(),
            thresholds=summary.thresholds,
            training_data_manifest_sha256=_sha256_file(config.data_dir / "manifest.json"),
            training_key_bank_id=data_manifest["training_key_bank_id"],
            holdout_key_bank_id=data_manifest["holdout_key_bank_id"],
            window_contract_version=self.options.window_contract_version,
        )
        shutil.rmtree(inputs)
        return ValidationOutcome(validated=True, summary=summary, bundle=bundle)

    def _training_cache_path(self, config_hash: str) -> Path:
        return self.options.cache_dir / f"gate-examples-{config_hash}.jsonl"

    def _append_training_examples(
        self,
        group: GatePipelineGroup,
        observations_by_length: Mapping[str, tuple[CandidateObservation, ...]],
        *,
        fast_experimental: bool = False,
    ) -> None:
        from wfcllm.gate.input import GateInput, serialize_gate_input

        training_group = self._training_group_by_group[group.group_id]
        metadata = self._unit_metadata_by_group[group.group_id]
        tokenizer = self._gate_tokenizer
        if tokenizer is None:
            tokenizer = _load_gate_tokenizer(self.options)
            self._gate_tokenizer = tokenizer
        rows: list[dict[str, Any]] = []
        for length in (1, 2, 3):
            thresholds = LabelThresholds(r3_hit_rate=0.25) if fast_experimental else None
            labels = build_gate_labels(
                observations_by_length[str(length)],
                training_key_count=32,
                thresholds=thresholds,
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
            for budget in (3,):
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
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)

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
        strategy = _rewrite_strategy_instruction(candidate_index)
        content = (
            "Task description (context only):\n"
            f"{bounded_task}\n\n"
            "Completed prefix (context only; do not output it):\n"
            f"{bounded_prefix}\n\n"
            f"Rewrite the target window into exactly {max_units} complete Python statements. "
            "Preserve every referenced name, assigned name, called function, attribute name, "
            "literal value, side effect, and control-flow outcome. Do not add imports, "
            "definitions, returns, raises, or calls. Do not rename anything. Use a natural "
            "behavior-preserving alternative only if one exists; otherwise copy the target. "
            f"{strategy} "
            f"Return code only, with exactly {max_units} complete Python statements.\n\n"
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


def _rewrite_strategy_instruction(candidate_index: int) -> str:
    strategy = _rewrite_strategy_id(candidate_index)
    instructions = {
        "a": (
            "Conservative plan A: make the smallest natural equivalent "
            "expression-level rewrite while retaining evaluation order."
        ),
        "b": (
            "Conservative plan B: use an independently phrased Python idiom "
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
) -> str:
    """Extract code blocks and restore indentation supplied by the prefix.

    The source slice starts at the first statement's AST byte, so the completed
    prefix already contains indentation for the generated first line. Every
    later physical line must receive that same public base indentation. Model
    prose is discarded only when it explicitly encloses code in Markdown
    fences; all fenced blocks are retained in their original order.
    """

    fenced_blocks = re.findall(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
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


@dataclass(frozen=True)
class _LocalGatePredictorFactory:
    base_model_path: Path
    tokenizer_path: Path
    artifacts: Any

    def __call__(self, mode, state):
        from transformers import AutoTokenizer

        from wfcllm.gate.bundle import quantize_gate_model_dynamic
        from wfcllm.gate.config import GateTrainConfig
        from wfcllm.gate.model import GateModel

        model = GateModel.from_local_pretrained(
            GateTrainConfig(base_model_path=self.base_model_path)
        ).cpu().eval()
        if mode.precision == "int8":
            model = quantize_gate_model_dynamic(model)
        model.load_state_dict(state, strict=True)
        device = "cuda" if mode.device == "gpu" else "cpu"
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path), local_files_only=True
        )
        return _LocalGatePredictor(model=model, tokenizer=tokenizer, device=device)


class _LocalGatePredictor:
    def __init__(self, *, model: object, tokenizer: object, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode_batch(self, examples):
        encoded = self.tokenizer(
            [example.serialized_input for example in examples],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }


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


def _validation_example_from_cache(row: Mapping[str, Any], role: str):
    from wfcllm.gate.validation import GateValidationExample

    span = row.get("span")
    return GateValidationExample(
        example_id=(
            f"{row['group_id']}:context-{row['context_length']}:budget-{row['budget']}"
        ),
        group_id=row["group_id"],
        serialized_input=row["serialized_gate_input"],
        span=(int(span[0]), int(span[1])),
        close_target=row["close_target"],
        suitable_target=row["suitable_target"],
        validation_role=role,
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


class _BundlePredictorAdapter:
    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        tokenizer_sha256: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_sha256 = tokenizer_sha256

    def encode_input(self, serialized_input: str):
        encoded = self.tokenizer(
            serialized_input,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].cpu(),
            "attention_mask": encoded["attention_mask"].cpu(),
        }


@dataclass(frozen=True)
class _BundlePredictorLoader:
    base_model_path: Path
    tokenizer_path: Path

    def __call__(self, *, precision, state, tokenizer_snapshot, manifest):
        from transformers import AutoTokenizer

        from wfcllm.gate.bundle import quantize_gate_model_dynamic
        from wfcllm.gate.config import GateTrainConfig
        from wfcllm.gate.model import GateModel

        model = GateModel.from_local_pretrained(
            GateTrainConfig(base_model_path=self.base_model_path)
        ).cpu().eval()
        if precision == "int8":
            model = quantize_gate_model_dynamic(model)
        model.load_state_dict(state, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path), local_files_only=True
        )
        return _BundlePredictorAdapter(
            model=model.eval(),
            tokenizer=tokenizer,
            tokenizer_sha256=tokenizer_snapshot.sha256,
        )


class LocalRuntimeGateBundle:
    """Validated bundle with its CPU predictor and tokenizer bound once."""

    def __init__(self, *, root: Path, base_model_path: Path, bundle_sha256: str) -> None:
        from transformers import AutoTokenizer

        from wfcllm.gate.bundle import GateBundle

        bundle = GateBundle.load(root)
        self.root = root
        self.bundle_sha256 = bundle_sha256
        self.manifest = bundle.manifest
        self.validation_summary = bundle.validation_summary
        self.tokenizer_sha256 = bundle.manifest.tokenizer_sha256
        self.stable_gate_predictor = bundle.stable_predictor(
            _BundlePredictorLoader(
                base_model_path=base_model_path,
                tokenizer_path=root / "tokenizer",
            )
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(root / "tokenizer"), local_files_only=True
        )

    def tokenizer_counter(self, text: str) -> int:
        return _token_count(self._tokenizer, text)


@dataclass(frozen=True)
class _ExperimentalGateManifest:
    window_contract_version: str
    gate_input_contract_version: str
    tokenizer_sha256: str
    close_low_threshold: float
    close_high_threshold: float
    suitable_accept_threshold: float
    max_tokens: int
    max_units: int
    runtime_profile: str


_UNVALIDATED_RUNTIME_SCHEMA = "wfcllm-unvalidated-runtime-thresholds/v1"
_UNVALIDATED_RUNTIME_FIELDS = {
    "schema_version",
    "runtime_profile",
    "close_low_threshold",
    "close_high_threshold",
    "suitable_accept_threshold",
    "max_units",
}


def _load_unvalidated_runtime_manifest(
    root: Path,
    *,
    tokenizer_sha256: str,
    max_tokens: int,
    window_contract_version: str = WINDOW_CONTRACT_VERSION,
) -> tuple[_ExperimentalGateManifest, str | None]:
    """Load an optional runtime profile whose bytes are bound by the bundle hash."""

    defaults = _ExperimentalGateManifest(
        window_contract_version=window_contract_version,
        gate_input_contract_version=GATE_INPUT_CONTRACT_VERSION,
        tokenizer_sha256=tokenizer_sha256,
        close_low_threshold=0.45,
        close_high_threshold=0.55,
        suitable_accept_threshold=0.5,
        max_tokens=max_tokens,
        max_units=3,
        runtime_profile="learned-gate-default/v1",
    )
    path = root / "runtime_thresholds.json"
    if not path.is_file():
        return defaults, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("unvalidated runtime threshold artifact is invalid") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _UNVALIDATED_RUNTIME_FIELDS
        or payload.get("schema_version") != _UNVALIDATED_RUNTIME_SCHEMA
    ):
        raise ValueError("unvalidated runtime threshold schema mismatch")
    profile = payload.get("runtime_profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("unvalidated runtime profile must be non-empty")
    thresholds = GateThresholds(
        close_low=payload.get("close_low_threshold"),
        close_high=payload.get("close_high_threshold"),
        suitable_accept=payload.get("suitable_accept_threshold"),
        max_units=payload.get("max_units"),
        max_input_tokens=max_tokens,
    )
    return (
        replace(
            defaults,
            close_low_threshold=thresholds.close_low,
            close_high_threshold=thresholds.close_high,
            suitable_accept_threshold=thresholds.suitable_accept,
            max_units=thresholds.max_units,
            runtime_profile=profile,
        ),
        _sha256_file(path),
    )


class _ExperimentalFloatGatePredictor:
    def __init__(self, *, model: object, tokenizer: object, max_tokens: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def predict(self, serialized_input: str) -> GateScores:
        import torch

        if not isinstance(serialized_input, str) or not serialized_input:
            raise ValueError("serialized_input must be a non-empty string")
        encoded = self.tokenizer(
            serialized_input,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=encoded["input_ids"].cpu(),
                attention_mask=encoded["attention_mask"].cpu(),
            )
        close = float(torch.sigmoid(output.close_logits)[0].cpu())
        suitable = float(torch.sigmoid(output.suitable_logits)[0].cpu())
        return GateScores(
            close_probability=close,
            suitable_probability=suitable,
            stable=True,
            precision_delta=0.0,
            decision_agreement=True,
        )


class LocalExperimentalRuntimeGateBundle:
    """Float-only diagnostic gate candidate; never a formal gate bundle."""

    experimental_only = True

    def __init__(
        self,
        *,
        root: Path,
        base_model_path: Path,
        bundle_sha256: str,
        max_tokens: int = 256,
        window_contract_version: str = WINDOW_CONTRACT_VERSION,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        from wfcllm.gate.bundle import sha256_directory
        from wfcllm.gate.config import GateTrainConfig
        from wfcllm.gate.model import GateModel

        float_path = root / "gate_float.pt"
        tokenizer_path = root / "tokenizer"
        if not float_path.is_file() or not tokenizer_path.is_dir():
            raise ValueError("experimental gate candidate is incomplete")
        state = torch.load(float_path, map_location="cpu", weights_only=True)
        model = GateModel.from_local_pretrained(
            GateTrainConfig(max_tokens=max_tokens, base_model_path=base_model_path)
        ).cpu().eval()
        model.load_state_dict(state, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )
        tokenizer_hash = sha256_directory(tokenizer_path)
        self.root = root
        self.bundle_sha256 = bundle_sha256
        self.tokenizer_sha256 = tokenizer_hash
        self.validation_summary = {
            "validated": False,
            "experimental_only": True,
            "diagnostic_only": True,
            "not_official_method": True,
        }
        self.manifest = _ExperimentalGateManifest(
            window_contract_version=window_contract_version,
            gate_input_contract_version=GATE_INPUT_CONTRACT_VERSION,
            tokenizer_sha256=tokenizer_hash,
            close_low_threshold=0.45,
            close_high_threshold=0.55,
            suitable_accept_threshold=0.5,
            max_tokens=max_tokens,
            max_units=3,
            runtime_profile="learned-gate-default/v1",
        )
        self.runtime_thresholds_sha256 = None
        self.stable_gate_predictor = _ExperimentalFloatGatePredictor(
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        self._tokenizer = tokenizer

    def tokenizer_counter(self, text: str) -> int:
        return _token_count(self._tokenizer, text)


class LocalUnvalidatedRuntimeGateBundle(LocalExperimentalRuntimeGateBundle):
    """Formal float candidate used when validation is explicitly out of scope."""

    experimental_only = False
    unvalidated_candidate = True

    def __init__(self, **kwargs: Any) -> None:
        window_contract_version = kwargs.get(
            "window_contract_version", WINDOW_CONTRACT_VERSION
        )
        super().__init__(**kwargs)
        self.manifest, self.runtime_thresholds_sha256 = (
            _load_unvalidated_runtime_manifest(
                self.root,
                tokenizer_sha256=self.tokenizer_sha256,
                max_tokens=self.manifest.max_tokens,
                window_contract_version=window_contract_version,
            )
        )
        self.validation_summary = {
            "validated": False,
            "validation_skipped_by_protocol": True,
            "unvalidated_candidate": True,
            "diagnostic_only": False,
            "not_official_method": False,
            "runtime_profile": self.manifest.runtime_profile,
            "runtime_thresholds_sha256": self.runtime_thresholds_sha256,
        }


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
        if program_prompt_mode not in {"completion", "mbpp_chat"}:
            raise ValueError(
                "program_prompt_mode must be completion or mbpp_chat"
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

        mbpp_chat = (
            _is_mbpp_sample(sample_id)
            and _is_mbpp_interface_prefix(prompt)
            and getattr(self, "program_prompt_mode", "completion")
            == "mbpp_chat"
        )
        lm_prompt = (
            _mbpp_chat_program_prompt(self.tokenizer, prompt)
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
        if _is_mbpp_sample(sample_id):
            combined = (
                prompt + completion
                if _is_mbpp_interface_prefix(prompt) and not mbpp_chat
                else completion
            )
            code = _extract_python_code_completion(combined)
            return (
                _normalize_mbpp_chat_interface(prompt, code)
                if mbpp_chat
                else code
            )
        return prompt + completion


def _is_mbpp_sample(sample_id: str) -> bool:
    return sample_id.lower().startswith("mbpp/")


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


class KeyedTextRegionWindowScorer:
    """One keyed text region for structurally valid statement rewrites."""

    def __init__(self, deployment_key: bytes) -> None:
        if not isinstance(deployment_key, bytes) or not deployment_key:
            raise ValueError("deployment_key must be non-empty bytes")
        self._key = deployment_key

    def score(self, *, window_text: str, parent_descriptor: str):
        from wfcllm.semantic.window_lsh import SemanticWindowEvidence

        if not isinstance(window_text, str) or not window_text.strip():
            raise ValueError("window_text must be non-empty")
        if not isinstance(parent_descriptor, str) or not parent_descriptor:
            raise ValueError("parent_descriptor must be non-empty")
        normalized = normalize_unit_text(window_text)
        payload = (
            b"wfcllm-keyed-text-region/v1\0"
            + parent_descriptor.encode("utf-8")
            + b"\0"
            + normalized.encode("utf-8")
        )
        digest = hmac.new(self._key, payload, hashlib.sha256).digest()
        bit = int(digest[0] & 1)
        region_digest = hmac.new(
            self._key,
            b"wfcllm-keyed-text-region-id/v1\0" + parent_descriptor.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return SemanticWindowEvidence(
            signature=(bit,),
            allowed_region_id=(
                "semantic-window-region/v1:hmac-sha256:" + region_digest
            ),
            hit=bit == 0,
            margin=1.0,
            stable=True,
        )


def build_local_semantic_window_scorer(
    options: LocalHFGateRuntimeOptions, deployment_key: bytes
):
    if options.semantic_evidence_rule == "keyed_text_region":
        return KeyedTextRegionWindowScorer(deployment_key)
    from wfcllm.semantic.window_lsh import SemanticWindowScorer

    components = _load_public_semantic_components(options)
    return SemanticWindowScorer(
        verifier=components.verifier,
        keying=WatermarkKeying(deployment_key.hex(), options.lsh_dimension),
        contract_version=options.window_contract_version,
        k=max(1, round(options.lsh_gamma * (2 ** options.lsh_dimension))),
        margin=0.0,
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
        + options.semantic_evidence_rule.encode("utf-8")
        + b"\0"
        + options.window_contract_version.encode("utf-8")
        + repr(float(options.semantic_preservation_threshold)).encode("ascii")
        + b"\0"
    )
    for path in (
        options.semantic_encoder_model_path,
        options.semantic_encoder_checkpoint_path,
        options.semantic_whitening_path,
    ):
        if path is None:
            continue
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
    counts = {group_id: 0 for group_id in indexed}
    for row in cache_rows:
        group_id = row["group_id"]
        variant = (group_id, row.get("context_length"), row.get("budget"))
        if variant in variants:
            raise ValueError("local gate cache contains a duplicate variant")
        variants.add(variant)
        counts[group_id] += 1
        index = indexed[group_id]
        if row.get("split") != index.get("split"):
            raise ValueError("local gate cache split contradicts formal gate data")
        if row.get("context_length") == 3 and (
            row.get("close_target") != index.get("close_target")
            or row.get("suitable_target") != index.get("suitable_target")
        ):
            raise ValueError("local gate cache label contradicts formal gate data")
    if any(count != 3 for count in counts.values()):
        raise ValueError("local gate cache must contain three variants per group")


def _partition_training_examples(
    examples: tuple[Any, ...],
    cache_rows: list[dict[str, Any]],
    *,
    fast_experimental: bool,
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
    if fast_experimental and not validation:
        validation = tuple(
            example for example, row in paired if row.get("split") == "test"
        )
        validation_role = "test_fallback"
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
            _statement_family(unit_by_id[item.start_unit_id].text)
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


def _statement_family(text: str) -> str:
    """Classify a public Python statement without consulting labels or keys."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("statement text must be a non-empty string")
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


def _load_causal_rewriter(options: LocalHFGateRuntimeOptions):
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
    language = language_for_window_contract(options.window_contract_version)
    return CausalWindowRewriter(
        HFCausalRewriteBackend(
            model=model,
            tokenizer=tokenizer,
            device=options.model_device,
            max_new_tokens=options.rewrite_max_new_tokens,
            temperature=options.rewrite_temperature,
            top_p=options.rewrite_top_p,
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
            encoder_checkpoint_path=(
                None
                if options.semantic_encoder_checkpoint_path is None
                else str(options.semantic_encoder_checkpoint_path)
            ),
            embed_dim=options.semantic_embed_dim,
            device=options.model_device,
            use_lora=False,
            use_bf16=False,
            secret_key="wfcllm-public-window-plane/v1",
            lsh_d=options.lsh_dimension,
            whitening_path=(
                None
                if options.semantic_whitening_path is None
                else str(options.semantic_whitening_path)
            ),
        )


__all__ = [
    "LOCAL_HF_ADAPTER_NAME",
    "GateSourceCatalogRecord",
    "LocalHFGateRuntimeOptions",
    "LocalHFProductionAdapter",
    "HFCausalRewriteBackend",
    "LocalHFProgramGenerator",
    "LocalRuntimeGateBundle",
    "LocalExperimentalRuntimeGateBundle",
    "LocalUnvalidatedRuntimeGateBundle",
    "LocalSemanticRuntime",
    "KeyedTextRegionWindowScorer",
    "build_local_semantic_window_scorer",
    "experiment_contract_hash",
    "load_source_catalog",
    "load_local_causal_rewriter",
    "local_semantic_runtime_hash",
    "phase_config_hash",
]
