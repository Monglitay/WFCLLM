"""Fail-fast validation for the public multi-language experiment matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path


SUPPORTED_EXPERIMENT_PAIRS = frozenset(
    {
        ("python", "humaneval"),
        ("python", "mbpp"),
        ("cpp", "humanevalpack"),
        ("java", "humanevalpack"),
        ("js", "humanevalpack"),
    }
)
SUPPORTED_EXPERIMENT_PROFILES = frozenset({"full"})
_EXPECTED_DATASET_COUNTS = {
    ("python", "humaneval"): 164,
    ("python", "mbpp"): 974,
    ("cpp", "humanevalpack"): 164,
    ("java", "humanevalpack"): 164,
    ("js", "humanevalpack"): 164,
}
_MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
    "vocab.json",
)


def validate_public_full_config_overlay(
    config: Mapping[str, object],
    language: str,
    dataset: str,
    profile: str,
) -> None:
    """Require the exact canonical overlay for one public Full profile."""

    validate_experiment_config(config, language, dataset, profile)
    strategy = (
        "python_ast_equivalent"
        if language == "python"
        else "model_semantic_window"
    )
    finalizer = (
        "humaneval_target_function_v1"
        if (language, dataset) == ("python", "humaneval")
        else "none"
    )
    contract = f"{language}-statement-window/v1"
    expected: dict[str, object] = {
        "method": {
            "name": "gated_semantic_window_v1",
            "windowing": {"contract_version": contract},
            "rewrite": {"strategy": strategy},
            "semantic": {"parent_descriptor_version": contract},
        },
        "generation": {
            "language": language,
            "dataset": dataset,
            "program_finalizer": finalizer,
        },
        "semantic_lsh": {
            "rule_name": "semantic_lsh",
            "lsh_d": 12,
            "lsh_gamma": 0.45,
        },
        "experiment": {"profile": "full"},
    }
    if dict(config) != expected:
        raise ValueError(
            "public Full config must equal the canonical overlay; "
            "fast, relaxed, or one-off parameter overrides are forbidden"
        )


def validate_experiment_config(
    config: Mapping[str, object],
    language: str,
    dataset: str,
    profile: str,
) -> None:
    """Validate one Full Reproduction Profile identity and Gate contract."""
    if (language, dataset) not in SUPPORTED_EXPERIMENT_PAIRS:
        raise ValueError(
            "unsupported language/dataset pair: "
            f"language={language!r}, dataset={dataset!r}"
        )
    if profile not in SUPPORTED_EXPERIMENT_PROFILES:
        raise ValueError(f"unsupported experiment profile: {profile!r}")

    generation = _mapping(config, "generation")
    experiment = _mapping(config, "experiment")
    semantic_lsh = _mapping(config, "semantic_lsh")
    method = _mapping(config, "method")
    rewrite = _mapping(method, "rewrite", prefix="method")
    windowing = _mapping(method, "windowing", prefix="method")
    semantic = _mapping(method, "semantic", prefix="method")

    _require_equal(method, "name", "gated_semantic_window_v1", "method")
    _require_equal(generation, "language", language, "generation")
    _require_equal(generation, "dataset", dataset, "generation")
    _require_equal(experiment, "profile", profile, "experiment")
    if semantic_lsh.get("rule_name") != "semantic_lsh":
        raise ValueError(
            "experiment configs must use semantic_lsh"
        )
    expected_strategy = (
        "python_ast_equivalent" if language == "python" else "model_semantic_window"
    )
    _require_equal(rewrite, "strategy", expected_strategy, "method.rewrite")
    expected_contract = f"{language}-statement-window/v1"
    _require_equal(
        windowing, "contract_version", expected_contract, "method.windowing"
    )
    _require_equal(
        semantic,
        "parent_descriptor_version",
        expected_contract,
        "method.semantic",
    )
    gate = method.get("gate")
    if gate is not None:
        if not isinstance(gate, Mapping) or set(gate) != {
            "input_contract_version",
            "candidate_contract_version",
            "uncertain_boundary_policy",
            "max_input_tokens",
        }:
            raise ValueError("method.gate must use the current fresh-run contract")
    if semantic_lsh.get("lsh_d") != 12 or semantic_lsh.get("lsh_gamma") != 0.45:
        raise ValueError("semantic_lsh must use d=12 and gamma=0.45")


def load_and_validate_experiment_config(
    path: Path,
    language: str,
    dataset: str,
    profile: str,
) -> dict[str, object]:
    """Load one explicit experiment config and validate its public identity."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment config JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("experiment config must contain a JSON object")
    validate_experiment_config(value, language, dataset, profile)
    return value


def validate_runtime_capabilities(config: Mapping[str, object]) -> None:
    """Reject language paths that would otherwise fall into Python runtime code."""
    generation = _mapping(config, "generation")
    method = _mapping(config, "method")
    rewrite = _mapping(method, "rewrite", prefix="method")
    language = generation.get("language")
    strategy = rewrite.get("strategy")
    if language == "python":
        if strategy != "python_ast_equivalent":
            raise ValueError(
                "Python experiment requires the certified python_ast_equivalent "
                "rewrite strategy"
            )
        return
    if language in {"cpp", "java", "js"}:
        if strategy != "model_semantic_window":
            raise ValueError(
                f"{language} experiments require model_semantic_window; "
                "Python AST and carrier rewrites are forbidden"
            )
        from wfcllm.windowing import (
            get_statement_unit_extractor,
            window_contract_for_language,
        )

        get_statement_unit_extractor(str(language))
        window_contract_for_language(str(language))
        return
    raise ValueError(f"unsupported runtime language: {language!r}")


def validate_runtime_resources(
    config: Mapping[str, object],
    *,
    generation_model_path: Path,
    rewrite_model_path: Path | None,
    semantic_encoder_model_path: Path,
    gate_base_model_path: Path,
    dataset_path: Path,
    pilot_source_catalog: Path,
    full_source_catalog: Path,
    negative_input: Path,
) -> None:
    """Fail before a Fresh Reproduction Run creates any artifact directories."""

    _mapping(config, "method")
    required_directories = {
        "generation model": generation_model_path,
        "semantic encoder base": semantic_encoder_model_path,
        "Gate base model": gate_base_model_path,
        "dataset root": dataset_path,
    }
    method = _mapping(config, "method")
    rewrite = _mapping(method, "rewrite", prefix="method")
    if rewrite.get("strategy") == "model_semantic_window":
        if rewrite_model_path is None:
            raise ValueError("rewrite model path is required for model rewriting")
        required_directories["rewrite model"] = rewrite_model_path
    for label, path in required_directories.items():
        _require_local_resource(path, label, directory=True)
        if label != "dataset root":
            _validate_local_hf_model(path, label)

    required_files = {
        "pilot Gate source catalog": pilot_source_catalog,
        "full Gate source catalog": full_source_catalog,
        "negative detector input": negative_input,
    }
    for label, path in required_files.items():
        _require_local_resource(path, label, directory=False)
    if pilot_source_catalog.resolve() == full_source_catalog.resolve():
        raise ValueError("pilot and full Gate source catalogs must be different")
    if _sha256_file(pilot_source_catalog) == _sha256_file(full_source_catalog):
        raise ValueError(
            "pilot and full Gate source catalogs must not contain identical input"
        )

    generation = _mapping(config, "generation")
    language = generation.get("language")
    dataset = generation.get("dataset")
    if not isinstance(language, str) or not isinstance(dataset, str):
        raise ValueError("generation language and dataset must be strings")
    _validate_dataset_runtime(dataset_path, dataset, language)
    pilot_sources, pilot_groups = _validate_gate_source_catalog(
        pilot_source_catalog, language, "pilot"
    )
    full_sources, full_groups = _validate_gate_source_catalog(
        full_source_catalog, language, "full"
    )
    if pilot_sources & full_sources:
        raise ValueError(
            "pilot and full Gate source catalogs must have disjoint source identities"
        )
    if pilot_groups & full_groups:
        raise ValueError(
            "pilot and full Gate source catalogs must have disjoint split groups"
        )
    from wfcllm.detection.gated_pipeline import load_jsonl_records

    negative_rows = load_jsonl_records(negative_input)
    if any(row["dataset"] != dataset for row in negative_rows):
        raise ValueError(
            "negative detector input dataset must match the Full profile dataset"
        )


def _validate_local_hf_model(path: Path, label: str) -> None:
    config_path = path / "config.json"
    _require_local_resource(config_path, f"{label} config", directory=False)
    config = _load_json_object(config_path, f"{label} config")
    if not config:
        raise ValueError(f"{label} config.json must not be empty")
    weights = [path / name for name in _MODEL_WEIGHT_FILES if (path / name).is_file()]
    if len(weights) != 1:
        raise ValueError(
            f"{label} must contain exactly one supported model weight or index file"
        )
    if weights[0].stat().st_size <= 0:
        raise ValueError(f"{label} model weight or index file is empty")
    if weights[0].suffix == ".json":
        _load_json_object(weights[0], f"{label} weight index")
    tokenizers = [path / name for name in _TOKENIZER_FILES if (path / name).is_file()]
    if not tokenizers:
        raise ValueError(f"{label} is missing a local tokenizer descriptor")
    for tokenizer in tokenizers:
        if tokenizer.stat().st_size <= 0:
            raise ValueError(f"{label} tokenizer descriptor is empty")
        if tokenizer.suffix == ".json":
            _load_json_object(tokenizer, f"{label} tokenizer descriptor")


def _validate_dataset_runtime(
    dataset_path: Path,
    dataset: str,
    language: str,
) -> None:
    from wfcllm import datasets

    adapter = datasets.get(dataset)
    try:
        adapter = type(adapter)(dataset_path=str(dataset_path))
    except TypeError as exc:
        raise ValueError(
            f"dataset adapter {dataset!r} cannot bind the requested local root"
        ) from exc
    if not adapter.supports(language):
        raise ValueError(
            f"dataset {dataset!r} does not support language={language!r}"
        )
    try:
        samples = tuple(adapter.iter_samples(language=language))
    except Exception as exc:
        raise ValueError(
            f"dataset {dataset!r}/{language!r} is not loadable from {dataset_path}"
        ) from exc
    expected = _EXPECTED_DATASET_COUNTS[(language, dataset)]
    if len(samples) != expected:
        raise ValueError(
            f"dataset {dataset!r}/{language!r} must contain exactly "
            f"{expected} tasks, got {len(samples)}"
        )
    task_ids: set[str] = set()
    for sample in samples:
        if (
            not isinstance(sample.task_id, str)
            or not sample.task_id
            or not isinstance(sample.prompt, str)
            or sample.language != language
        ):
            raise ValueError("dataset adapter returned a malformed Full-profile sample")
        if sample.task_id in task_ids:
            raise ValueError("dataset adapter returned duplicate task IDs")
        task_ids.add(sample.task_id)


def _validate_gate_source_catalog(
    path: Path,
    language: str,
    label: str,
) -> tuple[set[str], set[str]]:
    from wfcllm.gate.production import load_source_catalog
    from wfcllm.gate.sources import (
        GateSourceManifest,
        GateSourceRecord,
        canonical_gate_source_identity,
    )

    catalog_records = tuple(load_source_catalog(path))
    if not catalog_records:
        raise ValueError(f"{label} Gate source catalog must not be empty")
    records = tuple(
        GateSourceRecord(
            source_family=record.source_family,
            source_id=record.source_id,
            code=record.code,
            repository_id=record.repository_id,
            task_id=record.task_id,
            function_id=record.function_id,
            source_model_id=record.source_model_id,
            license_id=record.license_id,
            contract_or_hard_set=record.contract_or_hard_set,
        )
        for record in catalog_records
    )
    GateSourceManifest(records)
    oss_family = {
        "python": "oss_python",
        "cpp": "oss_cpp",
        "java": "oss_java",
        "js": "oss_js",
    }[language]
    present = {record.source_family for record in records}
    required = {"main_generation", oss_family, "parser_boundary"}
    if not required.issubset(present):
        raise ValueError(
            f"{label} Gate source catalog must contain source families "
            f"{sorted(required)}"
        )
    return (
        {
            canonical_gate_source_identity(record.source_id)
            for record in records
        },
        {record.split_group_id for record in records},
    )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid local JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_local_resource(path: Path, label: str, *, directory: bool) -> None:
    if not isinstance(path, Path):
        raise ValueError(f"{label} path must be a pathlib.Path")
    absolute = path if path.is_absolute() else Path.cwd() / path
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{label} path must not traverse symlinks: {path}")
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} {kind} is missing: {path}")


def _mapping(
    parent: Mapping[str, object],
    key: str,
    *,
    prefix: str = "",
) -> Mapping[str, object]:
    value = parent.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_equal(
    section: Mapping[str, object],
    key: str,
    expected: str,
    prefix: str,
) -> None:
    if section.get(key) != expected:
        raise ValueError(
            f"{prefix}.{key} must equal {expected!r}, got {section.get(key)!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate an experiment config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--check-runtime-capabilities", action="store_true")
    parser.add_argument("--check-runtime-resources", action="store_true")
    parser.add_argument("--generation-model-path", type=Path)
    parser.add_argument("--rewrite-model-path", type=Path)
    parser.add_argument("--semantic-encoder-model-path", type=Path)
    parser.add_argument("--gate-base-model-path", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--pilot-source-catalog", type=Path)
    parser.add_argument("--full-source-catalog", type=Path)
    parser.add_argument("--negative-input", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_and_validate_experiment_config(
            args.config,
            args.language,
            args.dataset,
            args.profile,
        )
        if args.check_runtime_capabilities:
            validate_runtime_capabilities(config)
        if args.check_runtime_resources:
            required = {
                "generation_model_path": args.generation_model_path,
                "semantic_encoder_model_path": args.semantic_encoder_model_path,
                "gate_base_model_path": args.gate_base_model_path,
                "dataset_path": args.dataset_path,
                "pilot_source_catalog": args.pilot_source_catalog,
                "full_source_catalog": args.full_source_catalog,
                "negative_input": args.negative_input,
            }
            if (
                _mapping(
                    _mapping(config, "method"),
                    "rewrite",
                    prefix="method",
                ).get("strategy")
                == "model_semantic_window"
            ):
                required["rewrite_model_path"] = args.rewrite_model_path
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "runtime resource validation is missing arguments: "
                    + ", ".join(missing)
                )
            validate_runtime_resources(
                config,
                generation_model_path=args.generation_model_path,
                rewrite_model_path=args.rewrite_model_path,
                semantic_encoder_model_path=args.semantic_encoder_model_path,
                gate_base_model_path=args.gate_base_model_path,
                dataset_path=args.dataset_path,
                pilot_source_catalog=args.pilot_source_catalog,
                full_source_catalog=args.full_source_catalog,
                negative_input=args.negative_input,
            )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
