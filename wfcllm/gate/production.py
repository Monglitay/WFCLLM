"""Single-machine, local-only production runtime for gated experiments."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata
from typing import Any

from wfcllm.gate.data import (
    GateBuildContext,
    GateDataBuilder,
    LshProbeResult,
)
from wfcllm.generation.window_rewriter import CausalWindowRewriter, RewriteGeneration
from wfcllm.gate.labels import build_gate_labels
from wfcllm.gate.pipeline import GatePipelineGroup, ValidationOutcome
from wfcllm.gate.schema import CandidateObservation, GateTrainingGroup
from wfcllm.gate.sources import GateSourceRecord
from wfcllm.semantic.keying import WatermarkKeying
from wfcllm.windowing import PythonStatementUnitExtractor
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
    rewrite_max_new_tokens: int = 128
    gate_batch_size: int = 9
    gate_epochs: int = 20
    gate_learning_rate: float = 2e-5
    gate_early_stopping_patience: int = 3
    gate_resume_checkpoint: Path | None = None

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
        for name in (
            "semantic_embed_dim",
            "lsh_dimension",
            "rewrite_max_new_tokens",
            "gate_batch_size",
            "gate_epochs",
            "gate_early_stopping_patience",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gate_batch_size < 3:
            raise ValueError("gate_batch_size must be at least 3")
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
        self._rewriter: object | None = None
        self._semantic_runtime: object | None = None
        self._parent_by_group: dict[str, str] = {}
        self._training_group_by_group: dict[str, GateTrainingGroup] = {}
        self._unit_metadata_by_group: dict[str, dict[str, Any]] = {}
        self._gate_tokenizer: object | None = None
        self._cache_config_hash: str | None = None

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
        extractor = PythonStatementUnitExtractor()
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
        for parsed in parsed_units:
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
                    parser_contract_version=config.parser_contract,
                ),
                source_text=record.code,
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
                split = _split_for(built.split_group_id)
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
                    statement_family=start.node_type,
                    r1_success_rate=0.0,
                    r3_success_rate=0.0,
                    r6_success_rate=0.0,
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
                    candidate_indices_by_window_length={length: tuple(range(7)) for length in (1, 2, 3)},
                    observed_training_key_ids=(),
                    observed_holdout_key_ids=(),
                    candidate_observations_by_length=candidates,
                    probe_results_by_length={str(length): tuple({} for _ in range(7)) for length in (1, 2, 3)},
                    row={
                        "schema_version": "wfcllm-gate-data/v1",
                        "group_id": built.group_id,
                        "split": split,
                    },
                )

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
            observations_by_length: dict[str, tuple[CandidateObservation, ...]] = {}
            results_by_length: dict[str, tuple[Mapping[str, LshProbeResult], ...]] = {}
            for length in (1, 2, 3):
                observations: list[CandidateObservation] = []
                candidate_results: list[Mapping[str, LshProbeResult]] = []
                for observation in group.candidate_observations_by_length[str(length)]:
                    signature, margin, precision_stable, batch_stable = runtime.signature_and_margin(
                        observation.code
                    )
                    stable = bool(precision_stable and batch_stable)
                    results: dict[str, LshProbeResult] = {}
                    for key_id, material in materials.items():
                        allowed = WatermarkKeying(material, len(signature)).derive_descriptor(
                            contract_version=config.parser_contract,
                            parent_descriptor=parent,
                            k=max(1, round(0.25 * (2 ** len(signature)))),
                        )
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
                        )
                    )
                    candidate_results.append(results)
                observations_by_length[str(length)] = tuple(observations)
                results_by_length[str(length)] = tuple(candidate_results)

            selected = build_gate_labels(observations_by_length["3"], training_key_count=32)
            r1, r3, r6 = (selected.budgets[budget].success_rate for budget in (1, 3, 6))
            holdout_hits = sum(
                any(
                    results_by_length["3"][candidate][key_id].is_reliable_hit(configured_margin=0.0)
                    for candidate in range(4)
                )
                for key_id in holdout_keys.key_ids
            )
            first_hits = [
                value for value in selected.budgets[6].first_hit_by_key_id.values() if value is not None
            ]
            result = replace(
                group,
                suitable_target=selected.suitable_target,
                close_target=selected.close_target,
                r1_success_rate=r1,
                r3_success_rate=r3,
                r6_success_rate=r6,
                holdout_success_rate=holdout_hits / len(holdout_keys.key_ids),
                structural_invalid_rate=1.0 - selected.budgets[6].structural_valid_rate,
                numeric_instability_rate=selected.budgets[6].unstable_rate,
                first_hit_candidate_position=min(first_hits) if first_hits else None,
                observed_training_key_ids=tuple(training_keys.key_ids),
                observed_holdout_key_ids=tuple(holdout_keys.key_ids),
                candidate_observations_by_length=observations_by_length,
                probe_results_by_length=results_by_length,
            )
            self._append_training_examples(result, observations_by_length)
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
        from wfcllm.gate.model import GateModel
        from wfcllm.gate.trainer import GateTrainer, GateTrainerConfig, seed_gate_training

        del learning_curve_plan
        cache_rows = self._load_training_cache(config.config_hash)
        _validate_cache_against_data(cache_rows, data_jsonl.parent)
        examples = tuple(_gate_example_from_cache(row) for row in cache_rows)
        training = tuple(
            example for example, row in zip(examples, cache_rows, strict=True)
            if row["split"] == "train"
        )
        validation = tuple(
            example for example, row in zip(examples, cache_rows, strict=True)
            if row["split"] == "validation"
        )
        if not training or not validation:
            raise ValueError("gate training requires non-empty train and validation examples")
        seed_gate_training(7)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.options.gate_base_model_path), local_files_only=True
        )
        model = GateModel.from_local_pretrained(
            GateTrainConfig(base_model_path=self.options.gate_base_model_path)
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
            ),
            device=self.options.gate_device,
        )
        summary = trainer.fit(
            training,
            validation,
            resume_from=self.options.gate_resume_checkpoint,
        )
        checkpoint = torch.load(
            work / "checkpoints" / "best.pt", map_location="cpu", weights_only=True
        )
        state = checkpoint.get("model_state")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("best gate checkpoint does not contain model state")
        output_dir.mkdir(parents=True, exist_ok=False)
        torch.save(state, output_dir / "gate_float.pt")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        shutil.rmtree(work)
        return {
            "backend": LOCAL_HF_ADAPTER_NAME,
            "best_epoch": summary["best_epoch"],
            "epochs_completed": summary["epochs_completed"],
            "training_example_count": len(training),
            "validation_example_count": len(validation),
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

        from wfcllm.gate.bundle import GateBundle, quantize_gate_model_dynamic
        from wfcllm.gate.config import GateTrainConfig, GateValidateConfig
        from wfcllm.gate.model import GateModel
        from wfcllm.gate.validation import (
            GateValidationArtifacts,
            GateValidator,
            sha256_file,
        )

        rows = self._load_training_cache(config.config_hash)
        _validate_cache_against_data(rows, config.data_dir)
        fit_ids = set(threshold_fit_group_ids)
        agreement_ids = set(agreement_group_ids)
        fit = tuple(
            _validation_example_from_cache(row, "threshold_fit")
            for row in rows
            if row["group_id"] in fit_ids and row["context_length"] == 3 and row["budget"] == 3
        )
        agreement = tuple(
            _validation_example_from_cache(row, "agreement")
            for row in rows
            if row["group_id"] in agreement_ids and row["context_length"] == 3 and row["budget"] == 3
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
        )
        shutil.rmtree(inputs)
        return ValidationOutcome(validated=True, summary=summary, bundle=bundle)

    def _training_cache_path(self, config_hash: str) -> Path:
        return self.options.cache_dir / f"gate-examples-{config_hash}.jsonl"

    def _append_training_examples(
        self,
        group: GatePipelineGroup,
        observations_by_length: Mapping[str, tuple[CandidateObservation, ...]],
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
            labels = build_gate_labels(observations_by_length[str(length)], training_key_count=32)
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
            for budget in (1, 3, 6):
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

    def __init__(self, *, model: object, tokenizer: object, device: str, max_new_tokens: int) -> None:
        if not callable(getattr(model, "generate", None)):
            raise ValueError("rewrite model must expose generate")
        if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
            raise ValueError("rewrite tokenizer must be callable and expose decode")
        if not isinstance(device, str) or not device:
            raise ValueError("rewrite device must be non-empty")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("rewrite max_new_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

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

        instruction = (
            "Rewrite the Python window below using at most "
            f"{max_units} complete statements. Return Python code only.\n"
            f"Task prompt:\n{prompt}\nCompleted prefix:\n{completed_prefix}\n"
            f"Original window:\n{original_window}\nRewritten window:\n"
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
        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        generated = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=self.max_new_tokens,
            generator=generator,
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
            text=text,
            generation_seed_id=f"local-hf-v1:{seed:016x}",
            rewrite_config_id=f"local-hf-v1:max-new-tokens={self.max_new_tokens}",
        )


class LocalSemanticRuntime:
    """Key-independent semantic signature runtime with a repeatability check."""

    def __init__(self, verifier: object) -> None:
        if not callable(getattr(verifier, "verify", None)):
            raise ValueError("semantic verifier must expose verify")
        self.verifier = verifier

    def signature_and_margin(self, window_text: str):
        sentinel = frozenset({(0, 0, 0, 0)})
        first = self.verifier.verify(window_text, sentinel, 0.0)
        second = self.verifier.verify(window_text, sentinel, 0.0)
        first_signature = tuple(first.lsh_signature)
        second_signature = tuple(second.lsh_signature)
        first_margin = float(first.min_margin)
        second_margin = float(second.min_margin)
        stable = first_signature == second_signature and first_margin == second_margin
        return first_signature, min(first_margin, second_margin), stable, stable


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

        model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
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
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.is_encoder_decoder = bool(getattr(model_config, "is_encoder_decoder", False))
        self.rewriter = CausalWindowRewriter(
            HFCausalRewriteBackend(
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                max_new_tokens=rewrite_max_new_tokens,
            )
        )

    def generate_program(self, *, prompt: str, sample_id: str) -> str:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in dict(encoded).items()
        }
        digest = hashlib.sha256(f"{self.seed}\0{sample_id}\0{prompt}".encode("utf-8")).digest()
        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(
            int.from_bytes(digest[:8], "big")
        )
        generated = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-6),
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            generator=generator,
        )
        token_ids = generated[0]
        if not self.is_encoder_decoder:
            token_ids = token_ids[inputs["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not isinstance(completion, str):
            raise ValueError("generation tokenizer decode must return text")
        return completion if self.is_encoder_decoder else prompt + completion


def build_local_semantic_window_scorer(
    options: LocalHFGateRuntimeOptions, deployment_key: bytes
):
    from wfcllm.semantic.lsh import load_semantic_lsh_components
    from wfcllm.semantic.window_lsh import SemanticWindowScorer

    components = load_semantic_lsh_components(
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
    return SemanticWindowScorer(
        verifier=components.verifier,
        keying=WatermarkKeying(deployment_key.hex(), options.lsh_dimension),
        contract_version="python-statement-window/v1",
        k=max(1, round(0.25 * (2 ** options.lsh_dimension))),
        margin=0.0,
    )


def local_semantic_runtime_hash(options: LocalHFGateRuntimeOptions) -> str:
    digest = hashlib.sha256(b"wfcllm-local-semantic-runtime/v1\0")
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
    if any(count != 9 for count in counts.values()):
        raise ValueError("local gate cache must contain nine variants per group")


@dataclass(frozen=True)
class _ParsedCatalogSource:
    record: GateSourceCatalogRecord
    units: tuple[Any, ...]


class _StructuralOnlyProbe:
    """Key-independent placeholder replaced before labels are computed."""

    def probe(self, *, window_text: str, parent_descriptor: str, key_ids: tuple[str, ...]):
        return {
            key_id: LshProbeResult((0, 0, 0, 0), 0.0, False, False, False, False)
            for key_id in key_ids
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
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    path = options.effective_rewrite_model_path
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    model_class = AutoModelForSeq2SeqLM if bool(getattr(config, "is_encoder_decoder", False)) else AutoModelForCausalLM
    model = model_class.from_pretrained(str(path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model.to(options.model_device)
    model.eval()
    return CausalWindowRewriter(
        HFCausalRewriteBackend(
            model=model,
            tokenizer=tokenizer,
            device=options.model_device,
            max_new_tokens=options.rewrite_max_new_tokens,
        )
    )


def _load_semantic_runtime(options: LocalHFGateRuntimeOptions):
    from wfcllm.semantic.lsh import load_semantic_lsh_components

    components = load_semantic_lsh_components(
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
            None if options.semantic_whitening_path is None else str(options.semantic_whitening_path)
        ),
    )
    return LocalSemanticRuntime(components.verifier)


__all__ = [
    "LOCAL_HF_ADAPTER_NAME",
    "GateSourceCatalogRecord",
    "LocalHFGateRuntimeOptions",
    "LocalHFProductionAdapter",
    "HFCausalRewriteBackend",
    "LocalHFProgramGenerator",
    "LocalRuntimeGateBundle",
    "LocalSemanticRuntime",
    "build_local_semantic_window_scorer",
    "experiment_contract_hash",
    "load_source_catalog",
    "load_local_causal_rewriter",
    "local_semantic_runtime_hash",
    "phase_config_hash",
]
