"""Current Gate-only phase runners invoked by :class:`PhaseOrchestrator`."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from wfcllm.cli.config_resolver import load_config
from wfcllm.orchestration.state import RunStateManager

_GATED_METHOD = "gated_semantic_window_v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_PUBLIC_AUDIT_ARTIFACT_BYTES = 32 * 1024 * 1024

def get_config(args: argparse.Namespace) -> dict:
    cfg = getattr(args, "_config_cache", None)
    if cfg is None:
        config_path = getattr(args, "config", None)
        cfg = load_config(config_path) if isinstance(config_path, Path) else {}
        setattr(args, "_config_cache", cfg)
    return cfg


def run_generate(args: argparse.Namespace, state: RunStateManager) -> int:
    config = _require_gated_config(args)
    run_dir = _gate_run_dir(args, config)
    _runtime_secret(args, "deployment")
    _path, bundle_hash = resolve_gate_bundle(args)
    pipeline = _require_gated_generation_pipeline(args)
    output_path = _canonical_local_path(Path(pipeline.run()))
    final_code = _canonical_local_path(run_dir / "inputs" / "final_code.jsonl")
    if output_path != final_code:
        raise ValueError(
            "generate must produce --run-dir/inputs/final_code.jsonl"
        )
    from wfcllm.detection.gated_pipeline import load_jsonl_records

    final_rows = load_jsonl_records(final_code)
    final_code_hash = _safe_file_hash(final_code)
    manifest_path = run_dir / "generation" / "manifest.json"
    manifest = _load_json_object(manifest_path)
    if (
        manifest.get("final_code_sha256") != final_code_hash
        or manifest.get("final_code_row_count") != len(final_rows)
    ):
        raise ValueError("generation manifest does not bind final_code.jsonl")
    generation_model_identifier = manifest.get("generation_model_identifier")
    if (
        not isinstance(generation_model_identifier, str)
        or not generation_model_identifier
    ):
        raise ValueError("generation manifest is missing model identity")
    state.mark_done(
        "generate",
        method=_GATED_METHOD,
        gate_bundle_sha256=bundle_hash,
        output_path=str(final_code),
        final_code_sha256=final_code_hash,
        final_code_row_count=len(final_rows),
        generation_manifest_sha256=_safe_file_hash(manifest_path),
        generation_model_identifier=generation_model_identifier,
    )
    print("=== WFCLLM generate ===")
    return 0


def _require_gated_generation_pipeline(args: argparse.Namespace):
    pipeline = getattr(args, "_gated_generation_pipeline", None)
    if pipeline is None:
        factory = getattr(args, "_gated_generation_pipeline_factory", None)
        if callable(factory):
            pipeline = factory(args)
    if pipeline is None:
        pipeline = _build_local_gated_generation_pipeline(args)
    if not callable(getattr(pipeline, "run", None)):
        raise ValueError("gated generation runtime pipeline is not configured")
    return pipeline


def run_calibrate(args: argparse.Namespace, state: RunStateManager) -> int:
    config = _require_gated_config(args)
    _runtime_secret(args, "deployment")
    _path, bundle_hash = resolve_gate_bundle(args)
    if state.get("generate", "gate_bundle_sha256") != bundle_hash:
        raise ValueError("calibrate requires the same gate bundle as generate")
    _verify_frozen_generation_artifacts(state, _gate_run_dir(args, config))
    corpus_path, negative_manifest_hash = _resolve_gated_negative_corpus(
        args, config
    )
    negative_manifest = _load_json_object(
        _gate_run_dir(args, config)
        / "calibration"
        / "negative_corpus_manifest.json"
    )
    supplement_model = negative_manifest.get("generation_model_identifier")
    if (
        supplement_model is not None
        and supplement_model
        != state.get("generate", "generation_model_identifier")
    ):
        raise ValueError(
            "calibration supplement must use the same generation model as generate"
        )
    setattr(args, "_gated_negative_manifest_hash", negative_manifest_hash)
    pipeline = _require_gated_detection_pipeline(args)
    calibration_path = _gated_calibration_output(args, config)
    pipeline.calibrate_jsonl(str(corpus_path), output_path=calibration_path)
    if not calibration_path.is_file():
        raise ValueError("calibrate did not produce reference_calibration.json")
    if _validate_negative_corpus_manifest(
        _gate_run_dir(args, config)
    ) != negative_manifest_hash:
        raise ValueError("negative corpus manifest changed during calibration")
    state.mark_done(
        "calibrate",
        method=_GATED_METHOD,
        detector_mode="wfcllm-gated-semantic-window/v1",
        gate_bundle_sha256=bundle_hash,
        calibration_path=str(calibration_path),
        calibration_sha256=_safe_file_hash(calibration_path),
        negative_corpus_manifest_sha256=negative_manifest_hash,
    )
    print("=== WFCLLM calibrate ===")
    return 0


_GATED_NEGATIVE_CORPUS_MANIFEST_SCHEMA = "wfcllm-gated-negative-corpus-manifest/v1"


def _resolve_gated_negative_corpus(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> tuple[Path, str]:
    """Supplement held-out negatives to the configured calibration target."""

    calibration = config.get("calibration")
    target = (
        calibration.get("target_negative_count")
        if isinstance(calibration, Mapping)
        else None
    )
    negative_input = getattr(args, "negative_input", None)
    if not isinstance(negative_input, (str, Path)) or not str(negative_input):
        raise ValueError("gated calibrate requires --negative-input")
    external_records = _load_external_negative_records(negative_input)
    positive_ids, positive_prompts = _gated_positive_population(args, config)
    _assert_negative_population_is_held_out(
        external_records,
        positive_ids=positive_ids,
        positive_prompts=positive_prompts,
        label="external calibration negative corpus",
    )
    if target is None:
        path = Path(negative_input)
        return path, _safe_file_hash(path)
    if type(target) is not int or target < 1:
        raise ValueError(
            "calibration.target_negative_count must be a positive integer"
        )
    needed = target - len(external_records)
    generated_records: list[dict[str, str]] = []
    generation_model_identifier: str | None = None
    decoding_config: dict[str, object] | None = None
    if needed > 0:
        generated_records, generation_model_identifier, decoding_config = (
            _generate_unwatermarked_negative_supplement(
                args,
                config,
                needed,
                positive_ids=positive_ids,
                positive_prompts=positive_prompts,
                external_records=external_records,
            )
        )
        _assert_negative_population_is_held_out(
            generated_records,
            positive_ids=positive_ids,
            positive_prompts=positive_prompts,
            label="generated calibration negative supplement",
        )
    corpus_records = [*external_records, *generated_records]
    corpus_path = (
        _gate_run_dir(args, config) / "calibration" / "negative_corpus.jsonl"
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in corpus_records
        ),
        encoding="utf-8",
    )
    corpus_hash = _safe_file_hash(corpus_path)
    manifest_path = corpus_path.parent / "negative_corpus_manifest.json"
    _write_public_json(
        manifest_path,
        {
            "schema_version": _GATED_NEGATIVE_CORPUS_MANIFEST_SCHEMA,
            "target_negative_count": target,
            "external_count": len(external_records),
            "generated_count": len(generated_records),
            "generation_model_identifier": generation_model_identifier,
            "decoding_config": decoding_config,
            "corpus_sha256": corpus_hash,
        },
    )
    return corpus_path, _safe_file_hash(manifest_path)


def _load_external_negative_records(
    negative_input: object,
) -> list[dict[str, object]]:
    """Load the required external held-out negative corpus."""

    if not isinstance(negative_input, (str, Path)) or not str(negative_input):
        raise ValueError("gated calibrate requires --negative-input")
    path = Path(negative_input)
    if not path.is_file():
        raise ValueError(f"negative input does not exist: {path}")
    from wfcllm.detection.gated_pipeline import load_jsonl_records

    records = load_jsonl_records(path)
    return [dict(record) for record in records]


def _validate_negative_corpus_manifest(run_dir: Path) -> str:
    """Validate and hash the current calibration-corpus provenance manifest."""

    corpus_path = run_dir / "calibration" / "negative_corpus.jsonl"
    manifest_path = run_dir / "calibration" / "negative_corpus_manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != _GATED_NEGATIVE_CORPUS_MANIFEST_SCHEMA:
        raise ValueError("negative corpus manifest schema mismatch")
    target = manifest.get("target_negative_count")
    external = manifest.get("external_count")
    generated = manifest.get("generated_count")
    if (
        type(target) is not int
        or target < 1
        or type(external) is not int
        or external < 1
        or type(generated) is not int
        or generated < 0
        or external + generated < target
    ):
        raise ValueError("negative corpus manifest counts are invalid")
    from wfcllm.detection.gated_pipeline import load_jsonl_records

    records = load_jsonl_records(corpus_path)
    if len(records) != external + generated:
        raise ValueError("negative corpus row count does not match its manifest")
    if manifest.get("corpus_sha256") != _safe_file_hash(corpus_path):
        raise ValueError("negative corpus hash does not match its manifest")
    model_identifier = manifest.get("generation_model_identifier")
    decoding = manifest.get("decoding_config")
    if generated == 0:
        if model_identifier is not None or decoding is not None:
            raise ValueError(
                "negative corpus manifest has generation provenance without "
                "generated rows"
            )
    elif (
        not isinstance(model_identifier, str)
        or not model_identifier
        or not isinstance(decoding, Mapping)
    ):
        raise ValueError(
            "generated negative supplement requires model and decoding provenance"
        )
    return _safe_file_hash(manifest_path)


def _verify_frozen_generation_artifacts(
    state: RunStateManager,
    run_dir: Path,
) -> None:
    """Verify that downstream phases consume exactly generate's frozen output."""

    final_code = _canonical_local_path(
        run_dir / "inputs" / "final_code.jsonl"
    )
    recorded_path = state.get("generate", "output_path")
    if (
        not isinstance(recorded_path, str)
        or _canonical_local_path(Path(recorded_path)) != final_code
    ):
        raise ValueError("current generate output path is not bound to this run")
    from wfcllm.detection.gated_pipeline import load_jsonl_records

    rows = load_jsonl_records(final_code)
    current_hash = _safe_file_hash(final_code)
    if (
        state.get("generate", "final_code_sha256") != current_hash
        or state.get("generate", "final_code_row_count") != len(rows)
    ):
        raise ValueError("frozen final_code.jsonl changed after generate")
    manifest_path = run_dir / "generation" / "manifest.json"
    manifest = _load_json_object(manifest_path)
    if (
        state.get("generate", "generation_manifest_sha256")
        != _safe_file_hash(manifest_path)
        or manifest.get("final_code_sha256") != current_hash
        or manifest.get("final_code_row_count") != len(rows)
    ):
        raise ValueError("generation manifest binding mismatch")


def _generate_unwatermarked_negative_supplement(
    args: argparse.Namespace,
    config: Mapping[str, object],
    needed: int,
    *,
    positive_ids: set[str],
    positive_prompts: set[str],
    external_records: list[dict[str, object]],
) -> tuple[list[dict[str, str]], str, dict[str, object]]:
    """Generate plain unwatermarked completions on held-out prompts.

    No gated embedding, no window rewriting, and no deployment key are
    involved, and every completion is adopted as-is: the supplement path
    never reads pass/test/correctness signals.  The generation model is
    loaded only here, i.e. only when supplementation actually triggers.
    """

    generation = config.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("gated generation config is missing")
    decoding = _gated_negative_decoding_config(config, generation)
    samples = _load_gated_dataset_samples(args, generation)
    held_out_dataset = [
        {
            "id": sample["id"],
            "prompt": sample["prompt"],
            "source": "dataset",
        }
        for sample in samples
        if _normalise_population_text(sample["id"]) not in positive_ids
        and _normalise_prompt(sample["prompt"]) not in positive_prompts
    ]
    held_out_external = [
        {
            "id": str(record["id"]),
            "prompt": str(record["prompt"]),
            "source": "external-negative",
        }
        for record in external_records
    ]
    prompt_pool = [*held_out_dataset, *held_out_external]
    if not prompt_pool:
        raise ValueError(
            "calibration negative supplement requires at least one held-out "
            "prompt from the dataset remainder or external negative input"
        )
    selected = [prompt_pool[index % len(prompt_pool)] for index in range(needed)]
    options = _local_hf_runtime_options(args, config, "calibrate")
    from wfcllm.gate.production import LocalHFProgramGenerator

    program = LocalHFProgramGenerator(
        model_path=options.generation_model_path,
        device=options.model_device,
        max_new_tokens=int(decoding["max_new_tokens"]),
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        seed=int(decoding["seed"]),
        program_prompt_mode=str(decoding["prompt_mode"]),
        rewrite_max_new_tokens=options.rewrite_max_new_tokens,
        rewrite_generation_attempts=options.rewrite_generation_attempts,
        rewrite_temperature=options.rewrite_temperature,
        rewrite_top_p=options.rewrite_top_p,
        load_in_4bit=bool(decoding["load_in_4bit"]),
        torch_dtype=str(decoding["torch_dtype"]),
    )
    dataset = str(generation.get("dataset", "humaneval"))
    records: list[dict[str, str]] = []
    used_ids = {str(record["id"]) for record in external_records}
    for index, sample in enumerate(selected):
        base_id = str(sample["id"])
        sample_id = (
            base_id
            if base_id not in used_ids and index < len(held_out_dataset)
            else (
                "calibration-supplement/"
                f"{index:06d}/"
                + hashlib.sha256(
                    f"{sample['source']}\0{base_id}\0{index}".encode("utf-8")
                ).hexdigest()[:16]
            )
        )
        used_ids.add(sample_id)
        prompt = str(sample["prompt"])
        records.append(
            {
                "id": sample_id,
                "dataset": dataset,
                "prompt": prompt,
                "final_code": program.generate_program(
                    prompt=prompt,
                    sample_id=sample_id,
                ),
            }
        )
    return (
        records,
        _generation_model_identifier(options.generation_model_path),
        decoding,
    )


def _gated_negative_decoding_config(
    config: Mapping[str, object],
    generation: Mapping[str, object],
) -> dict[str, object]:
    """Generation-phase decoding defaults with optional calibration overrides."""

    decoding: dict[str, object] = {
        "max_new_tokens": int(generation.get("max_new_tokens", 256)),
        "temperature": float(generation.get("temperature", 0.25)),
        "top_p": float(generation.get("top_p", 0.95)),
        "seed": int(generation.get("seed", 7)),
        "prompt_mode": str(generation.get("prompt_mode", "completion")),
        "load_in_4bit": bool(generation.get("load_in_4bit", False)),
        "torch_dtype": str(generation.get("torch_dtype", "bf16")),
    }
    calibration = config.get("calibration")
    supplement = (
        calibration.get("supplement") if isinstance(calibration, Mapping) else None
    )
    if supplement is None:
        return decoding
    if not isinstance(supplement, Mapping):
        raise ValueError("calibration.supplement must be a mapping")
    allowed = {"max_new_tokens", "temperature", "top_p", "seed"}
    unknown = sorted(set(supplement) - allowed)
    if unknown:
        raise ValueError(
            f"calibration.supplement has unknown fields: {unknown}"
        )
    overridden = dict(decoding)
    for name in sorted(allowed & set(supplement)):
        overridden[name] = supplement[name]
    return overridden


def _gated_positive_population(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> tuple[set[str], set[str]]:
    """Read this run's public positive identity without using quality signals.

    inputs/final_code.jsonl is a generate artifact of the same run; reading
    its IDs and prompts here enforces the held-out guarantee for every negative
    row.  Detection, correctness, and quality signals are never read.
    """

    final_code_path = (
        _gate_run_dir(args, config) / "inputs" / "final_code.jsonl"
    )
    if not final_code_path.exists():
        raise ValueError(
            "calibration negative supplement requires the generate output "
            "inputs/final_code.jsonl to hold out positive prompts"
        )
    from wfcllm.detection.gated_pipeline import load_jsonl_records

    records = load_jsonl_records(final_code_path)
    ids = {_normalise_population_text(record["id"]) for record in records}
    prompts = {_normalise_prompt(record["prompt"]) for record in records}
    return ids, prompts


def _assert_negative_population_is_held_out(
    records: list[dict[str, object]],
    *,
    positive_ids: set[str],
    positive_prompts: set[str],
    label: str,
) -> None:
    for index, record in enumerate(records, start=1):
        record_id = _normalise_population_text(record["id"])
        prompt = _normalise_prompt(record["prompt"])
        if record_id in positive_ids:
            raise ValueError(
                f"{label} row {index} overlaps the positive population by id"
            )
        if prompt in positive_prompts:
            raise ValueError(
                f"{label} row {index} overlaps the positive population by prompt"
            )


def _normalise_population_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("population identity must be a non-empty string")
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _normalise_prompt(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("population prompt must be a non-empty string")
    return unicodedata.normalize("NFKC", value).replace("\r\n", "\n").strip()


def run_detect(args: argparse.Namespace, state: RunStateManager) -> int:
    config = _require_gated_config(args)
    run_dir = _gate_run_dir(args, config)
    _runtime_secret(args, "deployment")
    _path, bundle_hash = resolve_gate_bundle(args)
    if state.get("generate", "gate_bundle_sha256") != bundle_hash:
        raise ValueError("detect requires the same gate bundle as generate")
    if state.get("calibrate", "gate_bundle_sha256") != bundle_hash:
        raise ValueError("detect requires the same gate bundle as calibrate")
    _verify_frozen_generation_artifacts(state, run_dir)
    detector_input = run_dir / "inputs" / "final_code.jsonl"
    from wfcllm.detection.gated_pipeline import load_gated_calibration_artifact

    calibration_path = run_dir / "calibration" / "reference_calibration.json"
    if state.get("calibrate", "calibration_sha256") != _safe_file_hash(
        calibration_path
    ):
        raise ValueError("current calibration artifact hash mismatch")
    artifact = load_gated_calibration_artifact(calibration_path)
    current_negative_manifest_hash = _validate_negative_corpus_manifest(run_dir)
    if (
        artifact.negative_corpus_manifest_sha256
        != current_negative_manifest_hash
    ):
        raise ValueError("calibration negative corpus manifest hash mismatch")
    setattr(
        args,
        "_gated_negative_manifest_hash",
        artifact.negative_corpus_manifest_sha256,
    )
    pipeline = _require_gated_detection_pipeline(args)
    details_path = _gated_detection_output(args, config)
    pipeline.detect_jsonl(
        detector_input,
        artifact=artifact,
        output_path=details_path,
    )
    if not details_path.is_file():
        raise ValueError("detect did not produce positive_details.jsonl")
    detail_rows = _read_public_json_artifacts(details_path)
    if not detail_rows:
        raise ValueError("positive detection details must not be empty")
    state.mark_done(
        "detect",
        method=_GATED_METHOD,
        detector_mode="wfcllm-gated-semantic-window/v1",
        gate_bundle_sha256=bundle_hash,
        details_path=str(details_path),
        details_sha256=_safe_file_hash(details_path),
        details_row_count=len(detail_rows),
    )
    print("=== WFCLLM detect ===")
    return 0


def run_report(args: argparse.Namespace, state: RunStateManager) -> int:
    config = _require_gated_config(args)
    metrics = getattr(args, "_gated_report_metrics", None)
    run_dir_value = getattr(args, "run_dir", None)
    if metrics is None:
        if run_dir_value is None:
            raise ValueError("report requires --run-dir")
        metrics = _gated_report_metrics_from_artifacts(
            Path(run_dir_value), config
        )
    if not isinstance(metrics, Mapping):
        raise ValueError("gated report metrics must be a mapping")
    allowed_metrics = {
        "gate_coverage",
        "hit_count",
        "miss_count",
        "abstain_count",
        "rewrite_cost",
        "detection_curve",
        "calibration",
    }
    report_fields = {name: metrics.get(name) for name in sorted(allowed_metrics)}

    generation_section = config.get("generation")
    if not isinstance(generation_section, Mapping):
        raise ValueError("generation config is missing")
    report_fields["dataset"] = generation_section.get("dataset")
    report_fields["language"] = str(generation_section.get("language", "python"))

    posthoc = getattr(args, "_posthoc_pass_report", None)
    if posthoc is None and run_dir_value is not None:
        posthoc_path = Path(run_dir_value) / "reports" / "pass_report_posthoc.json"
        if posthoc_path.exists():
            posthoc = _load_json_object(posthoc_path)
    if posthoc is not None:
        if not isinstance(posthoc, dict):
            raise ValueError("posthoc pass report must be an object")
        from wfcllm.audit.artifact_integrity import assert_posthoc_pass_report_marker

        assert_posthoc_pass_report_marker(posthoc)
        report_fields["posthoc_pass_report"] = dict(posthoc)
    if run_dir_value is None:
        raise ValueError("report requires --run-dir")
    run_dir = Path(run_dir_value)
    _verify_frozen_generation_artifacts(state, run_dir)
    calibration_path = run_dir / "calibration" / "reference_calibration.json"
    if state.get("calibrate", "calibration_sha256") != _safe_file_hash(
        calibration_path
    ):
        raise ValueError("current calibration artifact hash mismatch")
    details_path = run_dir / "detection" / "positive_details.jsonl"
    if state.get("detect", "details_sha256") != _safe_file_hash(details_path):
        raise ValueError("current positive detection details hash mismatch")
    generation_model_identifier = state.get(
        "generate", "generation_model_identifier"
    )
    if (
        not isinstance(generation_model_identifier, str)
        or not generation_model_identifier
    ):
        raise ValueError("current generation model identity is missing")
    report_fields["generation_model"] = generation_model_identifier
    report_path = Path(run_dir_value) / "reports" / "reference_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_public_json(
        report_path,
        {
            "method": _GATED_METHOD,
            "detector_mode": "wfcllm-gated-semantic-window/v1",
            **report_fields,
        },
    )
    state.mark_done(
        "report",
        method=_GATED_METHOD,
        detector_mode="wfcllm-gated-semantic-window/v1",
        report_path=str(report_path),
        report_sha256=_safe_file_hash(report_path),
        **report_fields,
    )
    print("=== WFCLLM report ===")
    return 0


def run_audit(args: argparse.Namespace, state: RunStateManager) -> int:
    config = _require_gated_config(args)
    run_dir_value = getattr(args, "run_dir", None)
    if run_dir_value is None:
        raise ValueError("audit requires --run-dir")

    run_dir = Path(run_dir_value)
    final_code = run_dir / "inputs" / "final_code.jsonl"
    from wfcllm.audit import (
        audit_detector_input_file,
        audit_gate_artifact,
        audit_no_quality_gate_payload,
    )

    detector_summary = audit_detector_input_file(final_code)
    if detector_summary.get("ok") is not True:
        raise ValueError("official detector input integrity audit failed")

    negative_corpus = run_dir / "calibration" / "negative_corpus.jsonl"
    if not negative_corpus.is_file():
        raise ValueError(f"required audit artifact is missing: {negative_corpus}")
    negative_summary = audit_detector_input_file(negative_corpus)
    if negative_summary.get("ok") is not True:
        raise ValueError("calibration detector input integrity audit failed")

    artifact_paths = [
        run_dir / "gate-data" / "manifest.json",
        run_dir / "gate-data" / "window_groups.jsonl",
        run_dir / "gate-data" / "candidate_attempts.jsonl",
        run_dir / "gate-data" / "labels.jsonl",
        run_dir / "gate-data" / "split_manifest.json",
        run_dir / "gate-data" / "training_key_bank_manifest.json",
        run_dir / "gate-data" / "group_index.jsonl",
        run_dir / "gate-data" / "feasibility_summary.json",
        run_dir / "gate-train" / "development_summary.json",
        run_dir / "gate-train" / "candidate_bundle_manifest.json",
        run_dir / "generation" / "manifest.json",
        run_dir / "generation" / "audit.jsonl",
        run_dir / "generation" / "candidate_sidecar.jsonl",
        run_dir / "generation" / "progress.json",
        negative_corpus,
        run_dir / "calibration" / "negative_corpus_manifest.json",
        run_dir / "calibration" / "reference_calibration.json",
        run_dir / "detection" / "positive_details.jsonl",
        run_dir / "reports" / "reference_report.json",
    ]
    generation = config.get("generation")
    if (
        isinstance(generation, Mapping)
        and generation.get("program_finalizer") != "none"
    ):
        artifact_paths.append(run_dir / "generation" / "finalizer.jsonl")
    audited = 0
    for path in artifact_paths:
        if not path.is_file():
            raise ValueError(f"required audit artifact is missing: {path}")
        for payload in _read_public_json_artifacts(path):
            audit_gate_artifact(payload)
            no_quality = audit_no_quality_gate_payload(payload)
            if no_quality.get("ok") is not True:
                raise ValueError("formal artifact no-quality-gate audit failed")
            audited += 1
    _verify_frozen_generation_artifacts(state, run_dir)
    if _validate_negative_corpus_manifest(run_dir) != state.get(
        "calibrate", "negative_corpus_manifest_sha256"
    ):
        raise ValueError("current negative corpus manifest hash mismatch")
    if state.get("calibrate", "calibration_sha256") != _safe_file_hash(
        run_dir / "calibration" / "reference_calibration.json"
    ):
        raise ValueError("current calibration artifact hash mismatch")
    if state.get("detect", "details_sha256") != _safe_file_hash(
        run_dir / "detection" / "positive_details.jsonl"
    ):
        raise ValueError("current positive detection details hash mismatch")
    if state.get("report", "report_sha256") != _safe_file_hash(
        run_dir / "reports" / "reference_report.json"
    ):
        raise ValueError("current reference report hash mismatch")
    resolve_gate_bundle(args)

    from wfcllm.audit import reject_secret_key_leak

    posthoc_path = run_dir / "reports" / "pass_report_posthoc.json"
    if posthoc_path.exists():
        posthoc = _load_json_object(posthoc_path)
        from wfcllm.audit import assert_posthoc_pass_report_marker

        assert_posthoc_pass_report_marker(posthoc)
        reject_secret_key_leak(posthoc)
        audited += 1

    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_public_json(audit_dir / "detector_input_integrity.json", detector_summary)
    no_quality_summary = {"ok": True, "artifacts_checked": audited}
    artifact_summary = {
        "ok": True,
        "artifacts_checked": audited,
        "final_code_sha256": _safe_file_hash(final_code),
        "calibration_sha256": _safe_file_hash(
            run_dir / "calibration" / "reference_calibration.json"
        ),
        "positive_details_sha256": _safe_file_hash(
            run_dir / "detection" / "positive_details.jsonl"
        ),
        "reference_report_sha256": _safe_file_hash(
            run_dir / "reports" / "reference_report.json"
        ),
    }
    _write_public_json(audit_dir / "no_quality_gate_integrity.json", no_quality_summary)
    _write_public_json(audit_dir / "artifact_integrity.json", artifact_summary)
    state.mark_done(
        "audit",
        detector_input_integrity="pass",
        no_quality_gate_integrity="pass",
        gate_artifact_integrity="pass",
        artifacts_checked=audited,
    )
    print("=== WFCLLM audit ===")
    return 0


def _read_public_json_artifacts(path: Path) -> list[object]:
    """Read a bounded JSON/JSONL public artifact for the audit runner."""

    if path.suffix == ".jsonl":
        payloads: list[object] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    if len(line.encode("utf-8")) > _MAX_PUBLIC_AUDIT_ARTIFACT_BYTES:
                        raise ValueError(
                            f"public audit artifact line exceeds size limit: "
                            f"{path.name}:{line_number}"
                        )
                    payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL artifact: {path.name}") from exc
        return payloads

    raw = path.read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > _MAX_PUBLIC_AUDIT_ARTIFACT_BYTES:
        raise ValueError("public audit artifact exceeds size limit")
    try:
        return [json.loads(raw)]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc


def _write_public_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return value


def _gated_report_metrics_from_artifacts(
    run_dir: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate public gated metrics without adding detector evidence."""

    details_path = run_dir / "detection" / "positive_details.jsonl"
    if not details_path.is_file():
        raise ValueError("current positive detection details are missing")
    details = _read_public_json_artifacts(details_path)
    if not details:
        raise ValueError("current positive detection details are empty")
    hit_count = sum(int(row.get("hit_count", 0)) for row in details if isinstance(row, dict))
    miss_count = sum(int(row.get("miss_count", 0)) for row in details if isinstance(row, dict))
    abstain_count = sum(
        int(row.get("abstain_count", 0)) for row in details if isinstance(row, dict)
    )
    scoreable = hit_count + miss_count
    all_windows = scoreable + abstain_count
    gate_coverage = scoreable / all_windows if all_windows else 0.0

    detection_curve: list[dict[str, object]] = []
    for row in details:
        if not isinstance(row, dict):
            continue
        detection_curve.append(
            {
                "id": row.get("id"),
                "hit_rate": row.get("hit_rate"),
                "p_value": row.get("p_value"),
                "decision": row.get("decision"),
            }
        )

    rewrite_attempts = 0
    explicit_rewrite_attempts = False
    rewrite_attempts_lower_bound = 0
    selected_rewrite_count = 0
    selected_candidate_index_counts: dict[str, int] = {}
    generation_audit = run_dir / "generation" / "audit.jsonl"
    if generation_audit.exists():
        for row in _read_public_json_artifacts(generation_audit):
            if isinstance(row, dict):
                value = row.get("rewrite_attempts", row.get("attempt_count"))
                if type(value) is int and value >= 0:
                    explicit_rewrite_attempts = True
                    rewrite_attempts += value
                selected_index = row.get("selected_candidate_index")
                if type(selected_index) is int and selected_index >= 0:
                    label = str(selected_index)
                    selected_candidate_index_counts[label] = (
                        selected_candidate_index_counts.get(label, 0) + 1
                    )
                    rewrite_attempts_lower_bound += selected_index
                    selected_rewrite_count += int(selected_index > 0)

    calibration_path = run_dir / "calibration" / "reference_calibration.json"
    if not calibration_path.is_file():
        raise ValueError("current calibration artifact is missing")
    calibration: object = _load_json_object(calibration_path)

    return {
        "gate_coverage": gate_coverage,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "abstain_count": abstain_count,
        "rewrite_cost": {
            "attempts": rewrite_attempts if explicit_rewrite_attempts else None,
            "attempts_lower_bound": rewrite_attempts_lower_bound,
            "selected_rewrite_count": selected_rewrite_count,
            "selected_candidate_index_counts": dict(
                sorted(
                    selected_candidate_index_counts.items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "detection_curve": detection_curve,
        "calibration": calibration,
    }


def _resolve_current_pilot_feasibility(
    args: argparse.Namespace,
    config: Mapping[str, object],
    *,
    scale: str,
) -> Path | None:
    """Bind Full gate phases to the fresh canonical sibling pilot run."""

    supplied = _optional_runtime_path(args, "pilot_feasibility")
    if scale == "pilot":
        if supplied is not None:
            raise ValueError("pilot gate-data must not consume another pilot artifact")
        return None
    if scale != "full":
        raise ValueError("gate-data scale must be pilot or full")
    if supplied is None:
        raise ValueError("full Gate phases require --pilot-feasibility")
    run_dir = _canonical_local_path(_gate_run_dir(args, config))
    expected = _canonical_local_path(
        run_dir.parent / "pilot" / "gate-data" / "feasibility_summary.json"
    )
    if _canonical_local_path(supplied) != expected:
        raise ValueError(
            "pilot feasibility must come from the fresh canonical sibling "
            "pilot/gate-data/feasibility_summary.json"
        )
    pilot_state_path = run_dir.parent / "pilot_state.json"
    if not pilot_state_path.is_file():
        raise ValueError("fresh pilot run state is missing")
    pilot_state = RunStateManager(pilot_state_path)
    if not pilot_state.is_done("gate-data"):
        raise ValueError("fresh pilot gate-data phase is incomplete")
    from wfcllm.gate.production import experiment_contract_hash

    if pilot_state.get("gate-data", "config_hash") != experiment_contract_hash(
        config
    ):
        raise ValueError("pilot feasibility config hash mismatch")
    pilot_output = expected.parent
    if (
        pilot_state.get("gate-data", "manifest_path")
        != str(pilot_output / "manifest.json")
        or pilot_state.get("gate-data", "output_artifact_path")
        != str(pilot_output)
        or pilot_state.get("gate-data", "output_artifact_hash")
        != _stable_tree_hash(pilot_output)
    ):
        raise ValueError("pilot feasibility is not bound to the fresh pilot state")
    return expected


def run_gate_data(args: argparse.Namespace, state: RunStateManager) -> int:
    from wfcllm.gate.feasibility import FEASIBILITY_THRESHOLD_ITEMS
    from wfcllm.gate.pipeline import GateDataPipelineConfig, run_gate_data as pipeline

    config = _require_gated_config(args)
    dependencies = _formal_gate_dependencies(args, "gate-data")
    run_dir = _gate_run_dir(args, config)
    gate_data = config["gate_data"]
    method = config["method"]
    from wfcllm.gate.production import experiment_contract_hash

    resolved_hash = experiment_contract_hash(config)
    pilot_feasibility = _resolve_current_pilot_feasibility(
        args,
        config,
        scale=str(gate_data["scale"]),
    )
    thresholds = tuple(
        (name, gate_data["feasibility_thresholds"][name])
        for name, _value in FEASIBILITY_THRESHOLD_ITEMS
    )
    pipeline_config = GateDataPipelineConfig(
        output_root=run_dir,
        scale=gate_data["scale"],
        config_hash=resolved_hash,
        parser_contract=method["windowing"]["contract_version"],
        rewriter_config_hash=_canonical_hash(method["rewrite"]),
        semantic_encoder_hash=_canonical_hash(method["semantic"]),
        lsh_config_hash=_canonical_hash(config["semantic_lsh"]),
        feasibility_contract=gate_data["feasibility_contract_version"],
        feasibility_thresholds=thresholds,
        pilot_feasibility_path=pilot_feasibility,
        max_groups=int(gate_data["full_independent_group_max"]),
    )
    input_hash = compute_phase_input_hash(args, "gate-data")
    result = pipeline(pipeline_config, dependencies)
    expected_output, expected_manifest = _require_expected_gate_result_paths(
        args, "gate-data", result.output_dir, result.manifest_path
    )
    _require_formal_manifest(result.manifest, "gate-data")
    if result.manifest.get("config_hash") != resolved_hash:
        raise ValueError("gate-data output manifest config hash mismatch")
    _require_same_input_hash(args, "gate-data", input_hash)
    state.mark_done(
        "gate-data",
        config_hash=resolved_hash,
        input_hash=input_hash,
        output_manifest_hash=_safe_file_hash(expected_manifest),
        manifest_path=str(expected_manifest),
        output_artifact_path=str(expected_output),
        output_artifact_hash=_stable_tree_hash(expected_output),
    )
    print("=== WFCLLM gate-data ===")
    return 0


def run_gate_train(args: argparse.Namespace, state: RunStateManager) -> int:
    from wfcllm.gate.pipeline import GateTrainPipelineConfig, run_gate_train as pipeline

    config = _require_gated_config(args)
    dependencies = _formal_gate_dependencies(args, "gate-train")
    run_dir = _gate_run_dir(args, config)
    data_dir = run_dir / "gate-data"
    data_manifest = _load_json_object(data_dir / "manifest.json")
    _require_formal_manifest(data_manifest, "gate-data")
    from wfcllm.gate.production import experiment_contract_hash

    resolved_hash = experiment_contract_hash(config)
    pilot = _resolve_current_pilot_feasibility(
        args, config, scale="full"
    )
    input_hash = compute_phase_input_hash(args, "gate-train")
    result = pipeline(
        GateTrainPipelineConfig(
            output_root=run_dir,
            data_dir=data_dir,
            config_hash=resolved_hash,
            pilot_feasibility_path=pilot,
        ),
        dependencies,
    )
    expected_output, expected_manifest = _require_expected_gate_result_paths(
        args, "gate-train", result.output_dir, result.manifest_path
    )
    _require_formal_manifest(result.manifest, "gate-train")
    if result.manifest.get("config_hash") != resolved_hash:
        raise ValueError("gate-train output manifest config hash mismatch")
    _require_same_input_hash(args, "gate-train", input_hash)
    state.mark_done(
        "gate-train",
        config_hash=resolved_hash,
        input_hash=input_hash,
        output_manifest_hash=_safe_file_hash(expected_manifest),
        manifest_path=str(expected_manifest),
        output_artifact_path=str(expected_output),
        output_artifact_hash=_stable_tree_hash(expected_output),
    )
    print("=== WFCLLM gate-train ===")
    return 0


def compute_phase_input_hash(args: argparse.Namespace, phase: str) -> str:
    """Hash the resolved config plus the immutable inputs consumed by a gate phase."""

    injected = getattr(args, "_phase_input_hashes", None)
    if isinstance(injected, Mapping) and isinstance(injected.get(phase), str):
        value = injected[phase]
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("injected phase input hash must be lowercase SHA-256")
        return value
    config = _require_gated_config(args)
    parts = [phase.encode("utf-8"), bytes.fromhex(_canonical_hash(config))]
    run_dir = _gate_run_dir(args, config)
    if phase == "gate-data":
        source = _required_runtime_path(args, "gate_source_manifest")
        parts.append(bytes.fromhex(_safe_file_hash(source)))
        catalog = _required_runtime_path(args, "gate_source_catalog")
        parts.append(bytes.fromhex(_safe_file_hash(catalog)))
        parts.append(hashlib.sha256(_runtime_secret(args, "training_key_bank", refresh=True)).digest())
        parts.append(hashlib.sha256(_runtime_secret(args, "holdout_key_bank", refresh=True)).digest())
        pilot = _optional_runtime_path(args, "pilot_feasibility")
        if pilot is not None:
            parts.append(bytes.fromhex(_safe_file_hash(pilot)))
    elif phase == "gate-train":
        parts.append(bytes.fromhex(_safe_tree_hash(run_dir / "gate-data")))
        pilot = _optional_runtime_path(args, "pilot_feasibility") or run_dir / "gate-data" / "feasibility_summary.json"
        parts.append(bytes.fromhex(_safe_file_hash(pilot)))
        parts.append(
            bytes.fromhex(
                _canonical_hash(
                    {
                        "gate_epochs": int(config["gate_train"]["max_epochs"]),
                        "gate_early_stopping_patience": int(
                            config["gate_train"]["early_stopping_patience"]
                        ),
                    }
                )
            )
        )
    else:
        raise ValueError(f"input hashing is unsupported for phase {phase!r}")
    return hashlib.sha256(b"wfcllm-phase-input/v1\0" + b"".join(parts)).hexdigest()


def _candidate_config_hash_matches(
    expected_hash: object,
    config: Mapping[str, object],
) -> bool:
    """Bind the gate-train candidate manifest to the resolved experiment config."""

    from wfcllm.gate.production import experiment_contract_hash

    if not isinstance(expected_hash, str):
        return False
    return expected_hash == experiment_contract_hash(config)


def resolve_gate_bundle(args: argparse.Namespace) -> tuple[Path, str]:
    """Resolve the hash-bound gate-train candidate from this current run."""

    config = _require_gated_config(args)
    run_dir = _gate_run_dir(args, config)
    candidate = run_dir / "gate-train" / "candidate_bundle"
    candidate_manifest = _load_json_object(
        run_dir / "gate-train" / "candidate_bundle_manifest.json"
    )
    _require_formal_manifest(candidate_manifest, "gate-train")
    if not _candidate_config_hash_matches(
        candidate_manifest.get("config_hash"), config
    ):
        raise ValueError("gate candidate bundle config hash mismatch")
    expected = candidate_manifest.get("candidate_bundle_sha256")
    actual = _safe_tree_hash(candidate)
    if expected != actual:
        raise ValueError("gate candidate bundle hash mismatch")
    return candidate, actual


def _require_candidate_bundle_layout(path: Path) -> None:
    """Require the minimal gate bundle layout before any model load."""

    _reject_symlink_path(path)
    if not (path / "gate_float.pt").is_file() or not (path / "tokenizer").is_dir():
        raise ValueError("gate bundle must contain gate_float.pt and tokenizer/")


def _require_gated_config(args: argparse.Namespace) -> dict:
    config = get_config(args)
    if not _is_gated(config):
        raise ValueError("gate phases require gated_semantic_window_v1 resolved config")
    return config


def _is_gated(config: object) -> bool:
    return (
        isinstance(config, Mapping)
        and isinstance(config.get("method"), Mapping)
        and config["method"].get("name") == _GATED_METHOD
    )


def _require_gated_detection_pipeline(args: argparse.Namespace):
    """Return the runtime-wired pipeline supplied by the orchestration layer.

    Gate model/tokenizer and semantic encoder construction are deliberately
    outside the phase runner.  This keeps the runner from silently loading a
    different encoder or deployment key than the validated runtime selected.
    """

    pipeline = getattr(args, "_gated_detection_pipeline", None)
    if pipeline is None:
        factory = getattr(args, "_gated_detection_pipeline_factory", None)
        if callable(factory):
            pipeline = factory(args)
    if pipeline is None and getattr(args, "semantic_encoder_model_path", None):
        pipeline = _build_local_gated_detection_pipeline(args)
        setattr(args, "_gated_detection_pipeline", pipeline)
    if not callable(getattr(pipeline, "calibrate_jsonl", None)) or not callable(
        getattr(pipeline, "detect_jsonl", None)
    ):
        raise ValueError("gated detection runtime pipeline is not configured")
    return pipeline


def _resolve_program_finalizer(
    generation: Mapping[str, object],
):
    name = str(generation.get("program_finalizer", "none"))
    if name == "none":
        return None, name
    if name == "humaneval_target_function_v1":
        from wfcllm.generation.completion_finalizer import (
            finalize_humaneval_program,
        )

        return finalize_humaneval_program, name
    if name == "mbpp_target_function_v1":
        from wfcllm.generation.completion_finalizer import finalize_mbpp_program

        return finalize_mbpp_program, name
    if name == "mbpp_target_interface_wrapper_v1":
        from wfcllm.generation.completion_finalizer import (
            finalize_mbpp_program_with_interface_wrapper,
        )

        return finalize_mbpp_program_with_interface_wrapper, name
    raise ValueError(f"unsupported generation program_finalizer: {name}")


def _load_gated_dataset_samples(
    args: argparse.Namespace,
    generation: Mapping[str, object],
) -> list[dict[str, str]]:
    """Load every normalized prompt of the configured dataset in adapter order."""
    from wfcllm import datasets

    dataset = str(generation.get("dataset", "humaneval"))
    language = str(generation.get("language", "python"))
    cli_dataset = getattr(args, "dataset", None)
    cli_language = getattr(args, "language", None)
    if cli_dataset is not None and cli_dataset != dataset:
        raise ValueError(
            f"--dataset must match generation.dataset={dataset!r}"
        )
    if cli_language is not None and cli_language != language:
        raise ValueError(
            f"--language must match generation.language={language!r}"
        )

    adapter = datasets.get(dataset)
    dataset_path = getattr(args, "dataset_path", None)
    if isinstance(dataset_path, str) and dataset_path:
        try:
            adapter = type(adapter)(dataset_path=dataset_path)
        except TypeError:
            pass
    if not adapter.supports(language):
        raise ValueError(
            f"dataset {dataset!r} does not support language={language!r}"
        )
    return [
        {"id": sample.task_id, "prompt": sample.prompt}
        for sample in adapter.iter_samples(language=language)
    ]


def _load_gated_generation_samples(
    args: argparse.Namespace,
    generation: Mapping[str, object],
) -> list[dict[str, str]]:
    """Load normalized prompts through the registered dataset adapter."""
    samples = _load_gated_dataset_samples(args, generation)
    offset = getattr(args, "sample_offset", None)
    limit = getattr(args, "sample_limit", None)
    if offset is not None:
        if offset < 0:
            raise ValueError("sample_offset must be non-negative")
        samples = samples[offset:]
    if limit is not None:
        if limit < 0:
            raise ValueError("sample_limit must be non-negative")
        samples = samples[:limit]
    return samples


def _build_local_gated_generation_pipeline(args: argparse.Namespace):
    from wfcllm.detection.gated_windows import GatedWindowExtractor
    from wfcllm.gate.production import (
        LocalCandidateRuntimeGateBundle,
        LocalHFProgramGenerator,
        build_local_semantic_window_scorer,
        load_local_causal_rewriter,
        local_semantic_runtime_hash,
    )
    from wfcllm.generation.gated_generator import GatedGenerator
    from wfcllm.generation.gated_pipeline import (
        GatedGenerationPipeline,
        GatedGenerationPipelineConfig,
    )
    from wfcllm.generation.window_rewriter import (
        KeyBlindAstEquivalentWindowRewriter,
    )

    config = _require_gated_config(args)
    options = _local_hf_runtime_options(args, config, "generate")
    bundle_path, bundle_hash = resolve_gate_bundle(args)
    deployment_key = _runtime_secret(args, "deployment")
    runtime_bundle = LocalCandidateRuntimeGateBundle(
        root=bundle_path,
        base_model_path=options.gate_base_model_path,
        bundle_sha256=bundle_hash,
        max_tokens=int(config["method"]["gate"]["max_input_tokens"]),
        window_contract_version=options.window_contract_version,
    )
    generation = config.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("gated generation config is missing")
    program_finalizer, program_finalizer_name = _resolve_program_finalizer(
        generation
    )
    program = LocalHFProgramGenerator(
        model_path=options.generation_model_path,
        device=options.model_device,
        max_new_tokens=int(generation.get("max_new_tokens", 256)),
        temperature=float(generation.get("temperature", 0.25)),
        top_p=float(generation.get("top_p", 0.95)),
        seed=int(generation.get("seed", 7)),
        program_prompt_mode=str(generation.get("prompt_mode", "completion")),
        rewrite_max_new_tokens=options.rewrite_max_new_tokens,
        rewrite_generation_attempts=options.rewrite_generation_attempts,
        rewrite_temperature=options.rewrite_temperature,
        rewrite_top_p=options.rewrite_top_p,
        load_in_4bit=bool(generation.get("load_in_4bit", False)),
        torch_dtype=str(generation.get("torch_dtype", "bf16")),
    )
    semantic_scorer = build_local_semantic_window_scorer(options, deployment_key)
    windowing = config["method"]["windowing"]
    if not isinstance(windowing, Mapping):
        raise ValueError("method.windowing config is missing")
    max_units_value = windowing.get("max_units")
    extractor = GatedWindowExtractor(
        runtime_bundle,
        defer_unreliable_until_max_units=windowing.get(
            "defer_unreliable_until_max_units",
            False,
        ),
        max_units_override=(
            int(max_units_value) if max_units_value is not None else None
        ),
    )
    if runtime_bundle.manifest.window_contract_version != options.window_contract_version:
        raise ValueError("gate bundle and experiment window contracts differ")
    language = str(generation.get("language", "python"))
    rewrite_strategy = str(config["method"]["rewrite"].get("strategy", ""))
    if rewrite_strategy == "python_ast_equivalent":
        if language != "python":
            raise ValueError("Python AST rewrite strategy cannot serve another language")
        rewrite_config = config["method"]["rewrite"]
        budget_value = rewrite_config.get("ast_variant_budget")
        rewriter = KeyBlindAstEquivalentWindowRewriter(
            ast_variant_budget=(
                int(budget_value) if budget_value is not None else 12
            ),
            comprehension_alpha=rewrite_config.get("comprehension_alpha", True),
        )
    elif rewrite_strategy == "model_semantic_window":
        rewriter = load_local_causal_rewriter(options)
    else:
        raise ValueError(f"unsupported gated rewrite strategy: {rewrite_strategy!r}")
    generator = GatedGenerator(
        partitioner=extractor.partitioner,
        scorer=semantic_scorer,
        rewriter=rewriter,
        max_rewrites=int(config["method"]["rewrite"]["max_attempts"]),
        evidence_channels=int(
            config["semantic_lsh"].get("evidence_channels", 1)
        ),
        extractor=extractor.unit_extractor,
    )
    dataset = str(generation.get("dataset", "humaneval"))
    samples = _load_gated_generation_samples(args, generation)
    run_dir = _gate_run_dir(args, config)
    semantic_hash = local_semantic_runtime_hash(options)
    generation_model_identifier = _generation_model_identifier(
        options.generation_model_path
    )
    return GatedGenerationPipeline(
        config=GatedGenerationPipelineConfig(
            output_dir=run_dir,
            dataset=dataset,
            bundle_path=bundle_path,
            bundle_sha256=bundle_hash,
            parser_contract=runtime_bundle.manifest.window_contract_version,
            gate_input_contract=runtime_bundle.manifest.gate_input_contract_version,
            tokenizer_sha256=runtime_bundle.manifest.tokenizer_sha256,
            semantic_encoder_sha256=semantic_hash,
            lsh_config_sha256=_canonical_hash(config["semantic_lsh"]),
            generation_config_sha256=_canonical_hash(generation),
            generation_model_identifier=generation_model_identifier,
            secret_source_type=(
                "file" if getattr(args, "secret_key_file", None) else "environment"
            ),
            embedding_passes=int(generation.get("embedding_passes", 1)),
        ),
        base_model=program,
        generator=generator,
        data_adapter=samples,
        deployment_key=deployment_key,
        program_finalizer=program_finalizer,
        program_finalizer_name=program_finalizer_name,
        bundle_loader=lambda _path: runtime_bundle,
    )


def _build_local_gated_detection_pipeline(args: argparse.Namespace):
    from wfcllm.detection.gated_pipeline import (
        GatedDetectionBindings,
        GatedDetectionPipeline,
    )
    from wfcllm.detection.gated_windows import GatedWindowExtractor
    from wfcllm.detection.scoring import GatedWindowScorer
    from wfcllm.gate.production import (
        LocalCandidateRuntimeGateBundle,
        build_local_semantic_window_scorer,
        local_semantic_runtime_hash,
    )

    config = _require_gated_config(args)
    options = _local_hf_runtime_options(args, config, "detect")
    bundle_path, bundle_hash = resolve_gate_bundle(args)
    deployment_key = _runtime_secret(args, "deployment")
    runtime_bundle = LocalCandidateRuntimeGateBundle(
        root=bundle_path,
        base_model_path=options.gate_base_model_path,
        bundle_sha256=bundle_hash,
        max_tokens=int(config["method"]["gate"]["max_input_tokens"]),
        window_contract_version=options.window_contract_version,
    )
    negative_hash = getattr(args, "_gated_negative_manifest_hash", None)
    if not isinstance(negative_hash, str) or _DIGEST.fullmatch(negative_hash) is None:
        raise ValueError(
            "gated detector runtime requires the current negative corpus "
            "manifest hash"
        )
    detector = config.get("detector")
    if not isinstance(detector, Mapping):
        raise ValueError("gated detector config is missing")
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("gated calibration config is missing")
    semantic_hash = local_semantic_runtime_hash(options)
    if runtime_bundle.manifest.window_contract_version != options.window_contract_version:
        raise ValueError("gate bundle and experiment window contracts differ")
    windowing = config["method"]["windowing"]
    if not isinstance(windowing, Mapping):
        raise ValueError("method.windowing config is missing")
    return GatedDetectionPipeline(
        extractor=GatedWindowExtractor(
            runtime_bundle,
            defer_unreliable_until_max_units=windowing.get(
                "defer_unreliable_until_max_units",
                False,
            ),
            max_units_override=(
                int(windowing["max_units"])
                if windowing.get("max_units") is not None
                else None
            ),
        ),
        scorer=GatedWindowScorer(
            semantic_scorer=build_local_semantic_window_scorer(options, deployment_key),
            minimum_reliable_windows=int(detector.get("minimum_reliable_windows", 2)),
            evidence_channels=int(
                config["semantic_lsh"].get("evidence_channels", 1)
            ),
        ),
        bindings=GatedDetectionBindings(
            gate_bundle_sha256=bundle_hash,
            semantic_encoder_sha256=semantic_hash,
            lsh_config_sha256=_canonical_hash(config["semantic_lsh"]),
            key_identifier_sha256=hashlib.sha256(deployment_key).hexdigest(),
            negative_corpus_manifest_sha256=negative_hash,
            window_contract_version=options.window_contract_version,
        ),
        target_fpr=float(detector.get("target_fpr", 0.05)),
        calibration_group_by=str(
            calibration.get("group_by", "reliable_window_count")
        ),
    )


def _gated_calibration_output(
    args: argparse.Namespace, config: Mapping[str, object]
) -> Path:
    return _gate_run_dir(args, config) / "calibration" / "reference_calibration.json"


def _gated_detection_output(
    args: argparse.Namespace, config: Mapping[str, object]
) -> Path:
    return _gate_run_dir(args, config) / "detection" / "positive_details.jsonl"


def _gate_run_dir(args: argparse.Namespace, config: Mapping[str, object]) -> Path:
    explicit = getattr(args, "run_dir", None)
    if not isinstance(explicit, (str, Path)) or not str(explicit):
        raise ValueError("gated phases require a non-empty --run-dir")
    path = Path(explicit)
    _reject_symlink_path(path)
    return path


_GATE_MANIFEST_NAMES = {
    "gate-data": "manifest.json",
    "gate-train": "candidate_bundle_manifest.json",
}


def _canonical_local_path(path: Path) -> Path:
    """Return an absolute lexical path after rejecting every symlink component."""

    _reject_symlink_path(path)
    return Path(os.path.abspath(os.fspath(path)))


def expected_gate_state_paths(args: argparse.Namespace, phase: str) -> tuple[Path, Path]:
    """Return the only output/manifest paths valid for ``phase`` in this run."""

    if phase not in _GATE_MANIFEST_NAMES:
        raise ValueError(f"expected paths are unsupported for phase {phase!r}")
    config = _require_gated_config(args)
    run_dir = _canonical_local_path(_gate_run_dir(args, config))
    output = _canonical_local_path(run_dir / phase)
    manifest = _canonical_local_path(output / _GATE_MANIFEST_NAMES[phase])
    if output != run_dir / phase or manifest != output / _GATE_MANIFEST_NAMES[phase]:
        raise ValueError("gate artifact path escaped the current run directory")
    return output, manifest


def _require_expected_gate_result_paths(
    args: argparse.Namespace,
    phase: str,
    output_dir: Path,
    manifest_path: Path,
) -> tuple[Path, Path]:
    expected_output, expected_manifest = expected_gate_state_paths(args, phase)
    if _canonical_local_path(Path(output_dir)) != expected_output:
        raise ValueError(f"{phase} output path does not match the current run")
    if _canonical_local_path(Path(manifest_path)) != expected_manifest:
        raise ValueError(f"{phase} manifest path does not match the current run")
    return expected_output, expected_manifest


def _formal_gate_dependencies(args: argparse.Namespace, phase: str) -> object:
    config = _require_gated_config(args)
    source_manifest = _optional_runtime_path(args, "gate_source_manifest")
    if source_manifest is None:
        raise ValueError(f"--gate-source-manifest is required for {phase}")
    _validate_current_source_manifest(args, source_manifest)
    adapter_options = _local_hf_runtime_options(args, config, phase)
    base_model_path = adapter_options.gate_base_model_path
    from wfcllm.gate.dependencies import build_local_gate_dependencies
    from wfcllm.gate.production import LOCAL_HF_ADAPTER_NAME

    dependencies = build_local_gate_dependencies(
        source_manifest=source_manifest,
        training_key_file=getattr(args, "training_key_bank_file", None),
        training_key_env=getattr(args, "training_key_bank_env", None),
        holdout_key_file=getattr(args, "holdout_key_bank_file", None),
        holdout_key_env=getattr(args, "holdout_key_bank_env", None),
        base_model_path=base_model_path,
        adapter_name=LOCAL_HF_ADAPTER_NAME,
        adapter_options=adapter_options,
    )
    if getattr(dependencies, "diagnostic_test_backend", None) is not False:
        raise ValueError("main orchestration rejects diagnostic test backend artifacts")
    return dependencies


def _validate_current_source_manifest(
    args: argparse.Namespace,
    source_manifest: Path,
) -> None:
    """Require prepare's complete formal manifest for the current catalog."""

    catalog = _required_runtime_path(args, "gate_source_catalog")
    from wfcllm.gate.production import load_source_catalog
    from wfcllm.gate.sources import GateSourceManifest, GateSourceRecord

    records = tuple(load_source_catalog(catalog))
    expected = GateSourceManifest(
        tuple(
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
            for record in records
        )
    ).to_dict()
    expected["catalog_sha256"] = _safe_file_hash(catalog)
    if _load_json_object(source_manifest) != expected:
        raise ValueError(
            "Gate source manifest does not bind the complete current formal catalog"
        )


def _resolve_semantic_encoder_checkpoint(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> Path:
    """Resolve only the semantic encoder trained by this Fresh Reproduction Run."""

    from wfcllm.orchestration.state import DEFAULT_STATE_FILE

    state_file = getattr(args, "state_file", None)
    state_path = (
        Path(state_file)
        if isinstance(state_file, (str, Path)) and str(state_file)
        else DEFAULT_STATE_FILE
    )
    if not state_path.exists():
        raise ValueError("current run semantic encoder state is missing")
    try:
        encoder_state = RunStateManager(path=state_path)
        recorded = encoder_state.get("encoder", "best_model_path")
        recorded_hash = encoder_state.get("encoder", "best_model_sha256")
        recorded_catalog_hash = encoder_state.get(
            "encoder", "source_catalog_sha256"
        )
        recorded_config_hash = encoder_state.get("encoder", "config_sha256")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("current run semantic encoder state is invalid") from exc
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("current run semantic encoder checkpoint is missing")
    checkpoint = _canonical_local_path(Path(recorded))
    expected = _canonical_local_path(
        _gate_run_dir(args, config) / "encoder" / "best_model.pt"
    )
    if checkpoint != expected:
        raise ValueError(
            "current run semantic encoder checkpoint path does not match "
            "--run-dir/encoder/best_model.pt"
        )
    if not checkpoint.is_file():
        raise ValueError("current run semantic encoder checkpoint does not exist")
    if (
        not isinstance(recorded_hash, str)
        or _DIGEST.fullmatch(recorded_hash) is None
        or _safe_file_hash(checkpoint) != recorded_hash
    ):
        raise ValueError("current run semantic encoder checkpoint hash mismatch")
    catalog = _required_runtime_path(args, "gate_source_catalog")
    if (
        not isinstance(recorded_catalog_hash, str)
        or _DIGEST.fullmatch(recorded_catalog_hash) is None
        or _safe_file_hash(catalog) != recorded_catalog_hash
    ):
        raise ValueError("current run semantic encoder source catalog hash mismatch")
    from wfcllm.gate.production import experiment_contract_hash

    if (
        not isinstance(recorded_config_hash, str)
        or _DIGEST.fullmatch(recorded_config_hash) is None
        or experiment_contract_hash(config) != recorded_config_hash
    ):
        raise ValueError("current run semantic encoder config hash mismatch")
    return checkpoint


def _local_hf_runtime_options(
    args: argparse.Namespace, config: Mapping[str, object], phase: str
):
    gate_train = config.get("gate_train")
    method = config.get("method")
    rewrite = method.get("rewrite") if isinstance(method, Mapping) else None
    semantic_lsh = config.get("semantic_lsh")
    gate = method.get("gate") if isinstance(method, Mapping) else None
    semantic = method.get("semantic") if isinstance(method, Mapping) else None
    method_semantic_lsh = (
        semantic.get("lsh") if isinstance(semantic, Mapping) else None
    )
    preservation = (
        semantic.get("preservation") if isinstance(semantic, Mapping) else None
    )
    semantic_rule = (
        str(semantic_lsh.get("rule_name", "semantic_lsh"))
        if isinstance(semantic_lsh, Mapping)
        else "semantic_lsh"
    )
    top_level_gamma = (
        semantic_lsh.get("lsh_gamma", 0.25)
        if isinstance(semantic_lsh, Mapping)
        else 0.25
    )
    method_gamma = (
        method_semantic_lsh.get("gamma", top_level_gamma)
        if isinstance(method_semantic_lsh, Mapping)
        else top_level_gamma
    )
    for gamma_name, gamma_value in (
        ("method.semantic.lsh.gamma", method_gamma),
        ("semantic_lsh.lsh_gamma", top_level_gamma),
    ):
        if (
            isinstance(gamma_value, bool)
            or not isinstance(gamma_value, (int, float))
            or not 0.0 < gamma_value < 1.0
        ):
            raise ValueError(f"{gamma_name} must be in (0, 1)")
    if (
        isinstance(method_semantic_lsh, Mapping)
        and "gamma" in method_semantic_lsh
        and isinstance(semantic_lsh, Mapping)
        and "lsh_gamma" in semantic_lsh
        and float(method_gamma) != float(top_level_gamma)
    ):
        raise ValueError("method and top-level LSH gamma must match")
    base_model_id = gate_train.get("base_encoder_id") if isinstance(gate_train, Mapping) else None
    base_model_override = getattr(args, "gate_base_model_path", None)
    base_model_path = (
        Path(base_model_override)
        if isinstance(base_model_override, str) and base_model_override
        else Path(base_model_id) if isinstance(base_model_id, str) else None
    )
    from wfcllm.gate.production import LocalHFGateRuntimeOptions

    def required_runtime_path(name: str) -> Path:
        value = getattr(args, name, None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"--{name.replace('_', '-')} is required for {phase}")
        return Path(value)

    if base_model_path is None:
        raise ValueError(f"--gate-base-model-path is required for {phase}")
    return LocalHFGateRuntimeOptions(
        source_catalog=required_runtime_path("gate_source_catalog"),
        generation_model_path=required_runtime_path("generation_model_path"),
        rewrite_model_path=(
            required_runtime_path("rewrite_model_path")
            if rewrite.get("strategy") == "model_semantic_window"
            else None
        ),
        semantic_encoder_model_path=required_runtime_path("semantic_encoder_model_path"),
        semantic_encoder_checkpoint_path=_resolve_semantic_encoder_checkpoint(
            args, config
        ),
        gate_base_model_path=base_model_path,
        model_device=getattr(args, "model_device", "cuda"),
        gate_device=getattr(args, "gate_device", "cuda"),
        cache_dir=Path(getattr(args, "gate_cache_dir", "data/gate-cache")),
        lsh_dimension=(
            int(method_semantic_lsh.get("d", semantic_lsh.get("lsh_d", 4)))
            if isinstance(method_semantic_lsh, Mapping)
            and isinstance(semantic_lsh, Mapping)
            else int(semantic_lsh.get("lsh_d", 4))
            if isinstance(semantic_lsh, Mapping)
            else 4
        ),
        lsh_gamma=float(method_gamma),
        semantic_evidence_rule=semantic_rule,
        semantic_preservation_threshold=(
            float(preservation.get("threshold", 0.9))
            if isinstance(preservation, Mapping)
            else 0.9
        ),
        rewrite_max_new_tokens=(
            int(rewrite.get("max_new_tokens", 32))
            if isinstance(rewrite, Mapping)
            else 32
        ),
        rewrite_generation_attempts=(
            int(rewrite.get("generation_attempts", 3))
            if isinstance(rewrite, Mapping)
            else 3
        ),
        rewrite_temperature=(
            float(rewrite.get("temperature", 0.8))
            if isinstance(rewrite, Mapping)
            else 0.8
        ),
        rewrite_top_p=(
            float(rewrite.get("top_p", 0.95))
            if isinstance(rewrite, Mapping)
            else 0.95
        ),
        gate_batch_size=getattr(args, "gate_batch_size", 9),
        gate_epochs=int(gate_train.get("max_epochs", 4)),
        gate_max_tokens=(
            int(gate_train.get("max_tokens", 256))
            if isinstance(gate_train, Mapping)
            else 256
        ),
        gate_learning_rate=(
            float(gate_train.get("learning_rate", 2e-5))
            if isinstance(gate_train, Mapping)
            else 2e-5
        ),
        gate_early_stopping_patience=int(
            gate_train.get("early_stopping_patience", 1)
        ),
        window_contract_version=str(
            method.get("windowing", {}).get(
                "contract_version", "python-statement-window/v1"
            )
        ),
    )


def _runtime_secret(
    args: argparse.Namespace,
    prefix: str,
    *,
    refresh: bool = False,
) -> bytes:
    private_key_bank = prefix in {"training_key_bank", "holdout_key_bank"}
    cache = getattr(args, "_gate_runtime_secrets", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(args, "_gate_runtime_secrets", cache)
    if prefix in cache and not refresh and not private_key_bank:
        value = cache[prefix]
        if not isinstance(value, bytes):
            raise ValueError("runtime secret cache is invalid")
        return value
    if prefix == "deployment":
        file_value = getattr(args, "secret_key_file", None)
        env_value = getattr(args, "secret_key_env", None)
    else:
        file_value = getattr(args, f"{prefix}_file", None)
        env_value = getattr(args, f"{prefix}_env", None)
    from wfcllm.common.secrets import load_secret

    value = load_secret(secret_file=file_value, env_name=env_value)
    if not refresh and not private_key_bank:
        cache[prefix] = value
    return value


def _require_same_input_hash(
    args: argparse.Namespace,
    phase: str,
    expected: str,
) -> None:
    if compute_phase_input_hash(args, phase) != expected:
        raise ValueError(f"{phase} input changed while the pipeline was running")


def _required_runtime_path(args: argparse.Namespace, name: str) -> Path:
    path = _optional_runtime_path(args, name)
    if path is None:
        raise ValueError(f"--{name.replace('_', '-')} is required")
    return path


def _optional_runtime_path(args: argparse.Namespace, name: str) -> Path | None:
    value = getattr(args, name, None)
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value) or "\x00" in str(value):
        raise ValueError(f"{name} must be a non-empty local path")
    path = Path(value)
    _reject_symlink_path(path)
    return path


def _canonical_hash(value: object) -> str:
    try:
        payload = json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("resolved config must be canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _reject_symlink_path(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    for candidate in (absolute, *absolute.parents):
        try:
            if candidate.is_symlink():
                raise ValueError("artifact path cannot traverse symlinks")
        except OSError as exc:
            raise ValueError("artifact path cannot be safely inspected") from exc


def _safe_file_hash(path: Path) -> str:
    _reject_symlink_path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("hash input must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("hash input changed while reading")
        return digest.hexdigest()
    except OSError as exc:
        raise ValueError("hash input is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _safe_tree_hash(root: Path) -> str:
    _reject_symlink_path(root)
    if not root.is_dir():
        raise ValueError("artifact tree is missing or unsafe")
    digest = hashlib.sha256(b"wfcllm-artifact-tree/v1\0")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("artifact tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("artifact tree contains an unsupported entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        before = path.stat()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(before.st_size.to_bytes(8, "big"))
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev, before.st_ino, before.st_size
            ):
                raise ValueError("artifact tree changed while hashing")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("artifact tree changed while hashing")
        except OSError as exc:
            raise ValueError("artifact tree is missing or unsafe") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return digest.hexdigest()


def _generation_model_identifier(model_path: Path) -> str:
    """Return a stable non-secret identity for one validated local model tree."""

    canonical = _canonical_local_path(model_path)
    return f"{canonical.name}:sha256:{_safe_tree_hash(canonical)}"


def _stable_tree_hash(root: Path) -> str:
    first = _safe_tree_hash(root)
    if _safe_tree_hash(root) != first:
        raise ValueError("artifact tree changed while hashing")
    return first


def _load_formal_json(path: Path, name: str) -> Mapping[str, object]:
    raw = _safe_read_file(path, max_bytes=1024 * 1024)

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in {name} manifest")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} manifest must be an object")
    _require_formal_manifest(value, name)
    return value


def _require_formal_manifest(manifest: Mapping[str, object], name: str) -> None:
    formal_identity = {
        "diagnostic_test_backend": False,
        "formal_eligible": True,
        "diagnostic_only": False,
        "not_official_method": False,
    }
    if any(
        manifest.get(field) is not value
        for field, value in formal_identity.items()
    ):
        raise ValueError(f"{name} artifact does not have complete formal identity")
    formal_schemas = {
        "gate-data": "wfcllm-gate-data-manifest/v1",
        "gate-train": "wfcllm-gate-train-candidate/v1",
    }
    if name in formal_schemas:
        if manifest.get("schema_version") != formal_schemas[name]:
            raise ValueError(f"{name} manifest schema is incompatible")

def _safe_read_file(path: Path, *, max_bytes: int) -> bytes:
    _reject_symlink_path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("public manifest is not a bounded regular file")
        data = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(data))):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError("public manifest exceeds size limit")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("public manifest changed while reading")
        return bytes(data)
    except OSError as exc:
        raise ValueError("public manifest is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def run_encoder(args: argparse.Namespace, state: RunStateManager) -> int:
    """Train the mandatory per-dataset semantic projection for a gated run."""

    from wfcllm.encoder import projection_training

    config = _require_gated_config(args)
    catalog = _required_runtime_path(args, "gate_source_catalog")
    model_path = _required_runtime_path(args, "semantic_encoder_model_path")
    generation = config.get("generation")
    language = (
        str(generation.get("language", "python"))
        if isinstance(generation, Mapping)
        else "python"
    )
    cli_language = getattr(args, "language", None)
    if cli_language is not None and cli_language != language:
        raise ValueError(
            f"--language must match generation.language={language!r}"
        )
    run_dir = _gate_run_dir(args, config)
    result = projection_training.train_semantic_projection(
        projection_training.ProjectionTrainingSettings(
            source_catalog=catalog,
            model_path=model_path,
            output_dir=run_dir / "encoder",
            language=language,
        )
    )
    best_model_path = _canonical_local_path(Path(str(result["best_model_path"])))
    expected_best_model = _canonical_local_path(
        run_dir / "encoder" / "best_model.pt"
    )
    if best_model_path != expected_best_model or not best_model_path.is_file():
        raise ValueError(
            "encoder trainer must produce the current run checkpoint at "
            "--run-dir/encoder/best_model.pt"
        )
    built_group_counts = dict(result["built_group_counts"])
    from wfcllm.gate.production import experiment_contract_hash

    state.mark_done(
        "encoder",
        run_dir=str(_canonical_local_path(run_dir)),
        checkpoint=str(best_model_path),
        best_model_path=str(best_model_path),
        best_model_sha256=_safe_file_hash(best_model_path),
        source_catalog_sha256=_safe_file_hash(catalog),
        config_sha256=experiment_contract_hash(config),
        language=language,
        built_group_counts=built_group_counts,
    )
    print("=== WFCLLM encoder ===")
    return 0
