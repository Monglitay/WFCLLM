"""Per-phase runner functions invoked by PhaseOrchestrator.

Lifted from run.py (Phase 1 refactor). Functions are CLI-bound:
they consume argparse.Namespace, build phase configs, invoke pipelines,
and update RunStateManager.

Future refactor phases (Phase 5: pretrain, Phase 8: generator split) will
further decompose these into their phase packages. For Phase 1 they stay
together to avoid scope creep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

from wfcllm.cli.config_resolver import load_config
from wfcllm.orchestration.state import RunStateManager

_GATED_METHOD = "gated_semantic_window_v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# ---------------------------------------------------------------------------
# Compare-only mode flag names
# ---------------------------------------------------------------------------

COMPARE_ONLY_REQUIRED_FLAGS = (
    "compare_summary_left",
    "compare_details_left",
    "compare_summary_right",
    "compare_details_right",
    "compare_output",
)
COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS = (
    "compare_watermarked_left",
    "compare_watermarked_right",
)

# ---------------------------------------------------------------------------
# Config helper utilities (used by multiple runners)
# ---------------------------------------------------------------------------


def get_config(args: argparse.Namespace) -> dict:
    cfg = getattr(args, "_config_cache", None)
    if cfg is None:
        config_path = getattr(args, "config", None)
        cfg = load_config(config_path) if isinstance(config_path, Path) else {}
        setattr(args, "_config_cache", cfg)
    return cfg


def configured_extract_input(args: argparse.Namespace) -> str | None:
    return (get_config(args).get("extract") or {}).get("input_file")


def is_compare_only_mode(args: argparse.Namespace) -> bool:
    required_present = all(getattr(args, flag, None) for flag in COMPARE_ONLY_REQUIRED_FLAGS)
    if not required_present:
        return False

    optional_watermarked_flags = tuple(
        getattr(args, flag, None) for flag in COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    )
    return not any(optional_watermarked_flags) or all(optional_watermarked_flags)


def has_explicit_extract_input(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "input_file", None) or configured_extract_input(args))


def validate_compare_only_mode(args: argparse.Namespace) -> str | None:
    """Return an error string if compare-only flags are used incorrectly, else None."""
    compare_flags = COMPARE_ONLY_REQUIRED_FLAGS + COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    if not any(getattr(args, flag, None) for flag in compare_flags):
        return None
    if args.phase not in {"extract", "legacy-extract"}:
        return "[错误] compare-only 模式仅支持 --phase legacy-extract"
    if not is_compare_only_mode(args):
        return (
            "[错误] compare-only 模式要求提供左右 summary/details 和 compare-output；"
            "watermarked 必须两侧同时提供或同时省略"
        )
    return None


def _phase_state_key(args: argparse.Namespace, default: str) -> str:
    override = getattr(args, "_state_phase_override", None)
    return override if isinstance(override, str) else default


def _run_with_state_phase(
    args: argparse.Namespace,
    state: RunStateManager,
    phase: str,
    runner,
) -> int:
    sentinel = object()
    previous = getattr(args, "_state_phase_override", sentinel)
    setattr(args, "_state_phase_override", phase)
    try:
        return runner(args, state)
    finally:
        if previous is sentinel:
            delattr(args, "_state_phase_override")
        else:
            setattr(args, "_state_phase_override", previous)


def _watermark_output_from_state(args: argparse.Namespace, state: RunStateManager) -> str | None:
    phase = _phase_state_key(args, "extract")
    if phase == "legacy-extract":
        return state.get("legacy-watermark", "output_file") or state.get(
            "watermark",
            "output_file",
        )
    return state.get("watermark", "output_file") or state.get(
        "legacy-watermark",
        "output_file",
    )


def _legacy_phase_archived(phase: str, guidance: str) -> int:
    print(
        f"[错误] {phase} implementation has been archived; {guidance}",
        file=sys.stderr,
    )
    return 1


# ---------------------------------------------------------------------------
# Per-phase runner functions
# ---------------------------------------------------------------------------


def run_generate(args: argparse.Namespace, state: RunStateManager) -> int:
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME

    config = get_config(args)
    if _is_gated(config):
        _runtime_secret(args, "deployment")
        _path, bundle_hash = resolve_validated_gate_bundle(args)
        pipeline = _optional_gated_generation_pipeline(args)
        output_path = pipeline.run() if pipeline is not None else None
        state.mark_done(
            "generate",
            method=_GATED_METHOD,
            gate_bundle_sha256=bundle_hash,
            output_path=(str(output_path) if output_path is not None else None),
            **_unvalidated_artifact_markers(config),
        )
    else:
        state.mark_done("generate", method=EVIDENCE_RETRY_SEED7X3_NAME)
    print("=== WFCLLM generate ===")
    return 0


def _optional_gated_generation_pipeline(args: argparse.Namespace):
    """Resolve the runtime-wired gated pipeline without affecting old presets."""

    pipeline = getattr(args, "_gated_generation_pipeline", None)
    if pipeline is None:
        factory = getattr(args, "_gated_generation_pipeline_factory", None)
        if callable(factory):
            pipeline = factory(args)
    if pipeline is None and getattr(args, "generation_model_path", None):
        pipeline = _build_local_gated_generation_pipeline(args)
    if pipeline is None:
        if hasattr(args, "generation_model_path"):
            raise ValueError(
                "--generation-model-path is required for gated generation"
            )
        # Narrow dependency-injected orchestration tests use Namespaces that do
        # not expose production runtime flags.
        return None
    if not callable(getattr(pipeline, "run", None)):
        raise ValueError("gated generation runtime pipeline is not configured")
    return pipeline


def run_calibrate(args: argparse.Namespace, state: RunStateManager) -> int:
    config = get_config(args)
    if _is_gated(config):
        _runtime_secret(args, "deployment")
        _path, bundle_hash = resolve_validated_gate_bundle(args)
        generated_hash = state.get("generate", "gate_bundle_sha256")
        if generated_hash != bundle_hash:
            raise ValueError("calibrate requires the same validated gate bundle as generate")
        calibration_path = None
        negative_input = getattr(args, "negative_input", None)
        if negative_input is not None:
            setattr(args, "_gated_negative_corpus_hash", _safe_file_hash(Path(negative_input)))
            pipeline = _require_gated_detection_pipeline(args)
            calibration_path = _gated_calibration_output(args, config)
            pipeline.calibrate_jsonl(negative_input, output_path=calibration_path)
        state.mark_done(
            "calibrate",
            method=_GATED_METHOD,
            detector_mode="wfcllm-gated-semantic-window/v1",
            gate_bundle_sha256=bundle_hash,
            calibration_path=(str(calibration_path) if calibration_path else None),
            **_unvalidated_artifact_markers(config),
        )
    else:
        state.mark_done("calibrate")
    print("=== WFCLLM calibrate ===")
    return 0


def run_detect(args: argparse.Namespace, state: RunStateManager) -> int:
    config = get_config(args)
    if _is_gated(config):
        _runtime_secret(args, "deployment")
        _path, bundle_hash = resolve_validated_gate_bundle(args)
        if state.get("generate", "gate_bundle_sha256") != bundle_hash:
            raise ValueError("detect requires the same validated gate bundle as generate")
        if state.get("calibrate", "gate_bundle_sha256") != bundle_hash:
            raise ValueError("detect requires the same validated gate bundle as calibrate")
        details_path = None
        detector_input = getattr(args, "input", None)
        if detector_input is not None:
            from wfcllm.detection.gated_pipeline import load_gated_calibration_artifact

            calibration_path = getattr(args, "calibration", None) or state.get(
                "calibrate", "calibration_path"
            )
            if not calibration_path:
                raise ValueError("gated detect requires a calibration artifact")
            artifact = load_gated_calibration_artifact(calibration_path)
            artifact_negative_hash = getattr(
                artifact, "negative_corpus_manifest_sha256", None
            )
            if isinstance(artifact_negative_hash, str):
                setattr(args, "_gated_negative_corpus_hash", artifact_negative_hash)
            pipeline = _require_gated_detection_pipeline(args)
            details_path = _gated_detection_output(args, config)
            pipeline.detect_jsonl(
                detector_input,
                artifact=artifact,
                output_path=details_path,
            )
        state.mark_done(
            "detect",
            method=_GATED_METHOD,
            detector_mode="wfcllm-gated-semantic-window/v1",
            gate_bundle_sha256=bundle_hash,
            details_path=(str(details_path) if details_path else None),
            **_unvalidated_artifact_markers(config),
        )
    else:
        state.mark_done("detect")
    print("=== WFCLLM detect ===")
    return 0


def run_report(args: argparse.Namespace, state: RunStateManager) -> int:
    config = get_config(args)
    if _is_gated(config):
        metrics = getattr(args, "_gated_report_metrics", None)
        if metrics is None:
            run_dir_value = getattr(args, "run_dir", None)
            metrics = (
                _gated_report_metrics_from_artifacts(Path(run_dir_value), config)
                if run_dir_value is not None
                else {}
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

        posthoc = getattr(args, "_posthoc_pass_report", None)
        run_dir_value = getattr(args, "run_dir", None)
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
        if run_dir_value is not None:
            report_path = Path(run_dir_value) / "reports" / "reference_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _write_public_json(
                report_path,
                {
                    "method": _GATED_METHOD,
                    "detector_mode": "wfcllm-gated-semantic-window/v1",
                    **_unvalidated_artifact_markers(config),
                    **report_fields,
                },
            )
        state.mark_done(
            "report",
            method=_GATED_METHOD,
            detector_mode="wfcllm-gated-semantic-window/v1",
            **_unvalidated_artifact_markers(config),
            **report_fields,
        )
    else:
        method = config.get("method") if isinstance(config, Mapping) else None
        method_name = method.get("name") if isinstance(method, Mapping) else None
        from wfcllm.detection.config import DETECTOR_MODE

        state.mark_done(
            "report",
            method=method_name,
            detector_mode=DETECTOR_MODE,
        )
    print("=== WFCLLM report ===")
    return 0


def run_audit(args: argparse.Namespace, state: RunStateManager) -> int:
    run_dir_value = getattr(args, "run_dir", None)
    if run_dir_value is None:
        state.mark_done("audit")
        print("=== WFCLLM audit ===")
        return 0

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

    artifact_paths = (
        run_dir / "gate-data" / "manifest.json",
        run_dir / "gate-data" / "window_groups.jsonl",
        run_dir / "gate-data" / "candidate_attempts.jsonl",
        run_dir / "gate-data" / "labels.jsonl",
        run_dir / "gate-data" / "split_manifest.json",
        run_dir / "gate-data" / "training_key_bank_manifest.json",
        run_dir / "gate-data" / "feasibility_summary.json",
        run_dir / "gate-train" / "resolved_training_config.json",
        run_dir / "gate-train" / "training_metrics.jsonl",
        run_dir / "gate-train" / "development_summary.json",
        run_dir / "gate-train" / "candidate_bundle_manifest.json",
        run_dir / "gate-validate" / "validation_summary.json",
        run_dir / "gate-validate" / "agreement_details.jsonl",
        run_dir / "gate-validate" / "gate_bundle_manifest.json",
        run_dir / "gate-validate" / "bundle" / "manifest.json",
        run_dir / "gate-validate" / "bundle" / "validation_summary.json",
        run_dir / "generation" / "audit.jsonl",
        run_dir / "generation" / "candidate_sidecar.jsonl",
        run_dir / "calibration" / "reference_calibration.json",
        run_dir / "detection" / "positive_details.jsonl",
        run_dir / "reports" / "reference_report.json",
        run_dir / "reports" / "model_negative_report.json",
    )
    audited = 0
    for path in artifact_paths:
        if not path.exists():
            continue
        for payload in _read_public_json_artifacts(path):
            audit_gate_artifact(payload)
            no_quality = audit_no_quality_gate_payload(payload)
            if no_quality.get("ok") is not True:
                raise ValueError("formal artifact no-quality-gate audit failed")
            audited += 1

    from wfcllm.audit import reject_secret_key_leak

    for config_path in (
        run_dir / "config" / "resolved_config.json",
        run_dir / "config" / "method_preset.json",
    ):
        if config_path.exists():
            reject_secret_key_leak(_load_json_object(config_path))
            audited += 1
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
    artifact_summary = {"ok": True, "artifacts_checked": audited}
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

    raw = path.read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("public audit artifact exceeds size limit")
    if path.suffix == ".jsonl":
        try:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL artifact: {path.name}") from exc
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
    details = _read_public_json_artifacts(details_path) if details_path.exists() else []
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
    calibration: object = None
    if calibration_path.exists():
        calibration = _load_json_object(calibration_path)
    elif isinstance(config.get("calibration"), Mapping):
        calibration = {
            name: config["calibration"].get(name)
            for name in ("method", "group_by", "target_fpr")
        }

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
        pilot_feasibility_path=_optional_runtime_path(args, "pilot_feasibility"),
        max_groups=int(gate_data["full_independent_group_max"]),
        fast_experimental=False,
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
    _load_formal_json(data_dir / "manifest.json", "gate-data")
    pilot = _optional_runtime_path(args, "pilot_feasibility")
    from wfcllm.gate.production import experiment_contract_hash

    resolved_hash = experiment_contract_hash(config)
    input_hash = compute_phase_input_hash(args, "gate-train")
    result = pipeline(
        GateTrainPipelineConfig(
            output_root=run_dir,
            data_dir=data_dir,
            config_hash=resolved_hash,
            pilot_feasibility_path=pilot,
            fast_experimental=False,
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


def run_gate_validate(args: argparse.Namespace, state: RunStateManager) -> int:
    from wfcllm.gate.pipeline import GateValidatePipelineConfig, run_gate_validate as pipeline

    config = _require_gated_config(args)
    dependencies = _formal_gate_dependencies(args, "gate-validate")
    run_dir = _gate_run_dir(args, config)
    data_dir = run_dir / "gate-data"
    candidate = run_dir / "gate-train" / "candidate_bundle"
    _load_formal_json(data_dir / "manifest.json", "gate-data")
    _load_formal_json(run_dir / "gate-train" / "candidate_bundle_manifest.json", "gate-train")
    from wfcllm.gate.production import experiment_contract_hash

    resolved_hash = experiment_contract_hash(config)
    input_hash = compute_phase_input_hash(args, "gate-validate")
    result = pipeline(
        GateValidatePipelineConfig(run_dir, candidate, data_dir, resolved_hash),
        dependencies,
    )
    if not result.validated or result.manifest_path is None or result.bundle is None:
        raise ValueError("gate-validate did not publish a validated formal gate bundle")
    expected_output, expected_manifest = _require_expected_gate_result_paths(
        args, "gate-validate", result.output_dir, result.manifest_path
    )
    publication = _load_formal_json(expected_manifest, "gate-validate")
    if publication.get("config_hash") != resolved_hash:
        raise ValueError("gate-validate publication config hash mismatch")
    _require_same_input_hash(args, "gate-validate", input_hash)
    expected_bundle = expected_output / "bundle"
    actual_bundle = Path(result.bundle.root) if hasattr(result.bundle, "root") else expected_bundle
    if _canonical_local_path(actual_bundle) != expected_bundle:
        raise ValueError("gate-validate result bundle path does not match the current run")
    bundle_path, bundle_hash = _validate_bundle_publication(expected_bundle, publication)
    state.mark_done(
        "gate-validate",
        config_hash=resolved_hash,
        input_hash=input_hash,
        output_manifest_hash=_safe_file_hash(expected_manifest),
        manifest_path=str(expected_manifest),
        output_artifact_path=str(expected_output),
        output_artifact_hash=_stable_tree_hash(expected_output),
        bundle_path=str(bundle_path),
        bundle_sha256=bundle_hash,
    )
    print("=== WFCLLM gate-validate ===")
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
                        "gate_epochs": _gate_training_runtime_int(
                            args, config, "gate_epochs", "max_epochs", 4
                        ),
                        "gate_early_stopping_patience": _gate_training_runtime_int(
                            args,
                            config,
                            "gate_early_stopping_patience",
                            "early_stopping_patience",
                            1,
                        ),
                    }
                )
            )
        )
    elif phase == "gate-validate":
        parts.append(bytes.fromhex(_safe_tree_hash(run_dir / "gate-data")))
        parts.append(bytes.fromhex(_safe_tree_hash(run_dir / "gate-train" / "candidate_bundle")))
        parts.append(bytes.fromhex(_safe_file_hash(run_dir / "gate-train" / "candidate_bundle_manifest.json")))
        parts.append(
            hashlib.sha256(
                _runtime_secret(args, "holdout_key_bank", refresh=True)
            ).digest()
        )
    else:
        raise ValueError(f"input hashing is unsupported for phase {phase!r}")
    return hashlib.sha256(b"wfcllm-phase-input/v1\0" + b"".join(parts)).hexdigest()


def _unvalidated_candidate_config_hash_matches(
    expected_hash: object,
    config: Mapping[str, object],
) -> bool:
    """Accept the frozen generation-only upgrade without weakening gate checks."""

    from wfcllm.gate.production import experiment_contract_hash

    if not isinstance(expected_hash, str):
        return False
    if expected_hash == experiment_contract_hash(config):
        return True

    generation_value = config.get("generation")
    if not isinstance(generation_value, Mapping):
        return False
    if (
        generation_value.get("max_new_tokens") != 512
        or generation_value.get("temperature") != 0.0
        or generation_value.get("program_finalizer")
        != "humaneval_target_function_v1"
    ):
        return False

    legacy_config = dict(config)
    legacy_generation = dict(generation_value)
    legacy_generation["max_new_tokens"] = 256
    legacy_generation["temperature"] = 0.25
    legacy_generation.pop("program_finalizer", None)
    legacy_config["generation"] = legacy_generation
    return expected_hash == experiment_contract_hash(legacy_config)


def resolve_validated_gate_bundle(args: argparse.Namespace) -> tuple[Path, str]:
    """Resolve a formal bundle or the explicitly enabled experimental candidate."""

    config = _require_gated_config(args)
    gate = config["method"]["gate"]
    configured_path = gate.get("bundle_path")
    configured_hash = gate.get("bundle_sha256")
    if configured_path is not None:
        if not isinstance(configured_path, str) or not isinstance(configured_hash, str):
            raise ValueError("external validated gate bundle path/hash are required together")
        path = Path(configured_path)
        actual = _safe_tree_hash(path)
        if actual != configured_hash:
            raise ValueError("validated gate bundle hash mismatch")
        from wfcllm.gate.bundle import GateBundle

        GateBundle.load(path)
        if _safe_tree_hash(path) != actual:
            raise ValueError("validated gate bundle changed while loading")
        return path, actual

    if gate.get("require_validated") is False:
        run_dir = _gate_run_dir(args, config)
        candidate = run_dir / "gate-train" / "candidate_bundle"
        publication = _load_json_object(
            run_dir / "gate-train" / "candidate_bundle_manifest.json"
        )
        _require_formal_manifest(publication, "gate-train")
        if not _unvalidated_candidate_config_hash_matches(
            publication.get("config_hash"), config
        ):
            raise ValueError("unvalidated gate candidate config hash mismatch")
        expected = publication.get("candidate_bundle_sha256")
        actual = _safe_tree_hash(candidate)
        if expected != actual:
            raise ValueError("unvalidated gate candidate hash mismatch")
        return candidate, actual

    run_dir = _gate_run_dir(args, config)
    publication = _load_formal_json(
        run_dir / "gate-validate" / "gate_bundle_manifest.json", "gate-validate"
    )
    return _validate_bundle_publication(run_dir / "gate-validate" / "bundle", publication)


def _validate_bundle_publication(
    bundle_path: Path,
    publication: Mapping[str, object],
) -> tuple[Path, str]:
    if (
        publication.get("validated") is not True
        or publication.get("diagnostic_test_backend") is not False
        or publication.get("schema_version") != "wfcllm-gate-validate-publication/v1"
    ):
        raise ValueError("validated gate bundle publication is not formal")
    expected = publication.get("bundle_sha256")
    if not isinstance(expected, str) or _DIGEST.fullmatch(expected) is None:
        raise ValueError("validated gate bundle publication hash is invalid")
    actual = _safe_tree_hash(bundle_path)
    if actual != expected:
        raise ValueError("validated gate bundle hash mismatch")
    from wfcllm.gate.bundle import GateBundle

    GateBundle.load(bundle_path)
    if _safe_tree_hash(bundle_path) != actual:
        raise ValueError("validated gate bundle changed while loading")
    return bundle_path, actual


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


def _uses_unvalidated_gate_candidate(config: Mapping[str, object]) -> bool:
    method = config.get("method")
    gate = method.get("gate") if isinstance(method, Mapping) else None
    return (
        isinstance(gate, Mapping)
        and gate.get("require_validated") is False
        and gate.get("bundle_path") is None
    )


def _unvalidated_artifact_markers(
    config: Mapping[str, object],
) -> dict[str, bool]:
    if not _uses_unvalidated_gate_candidate(config):
        return {}
    return {
        "gate_validation_skipped": True,
        "unvalidated_gate_candidate": True,
    }


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
    raise ValueError(f"unsupported generation program_finalizer: {name}")


def _build_local_gated_generation_pipeline(args: argparse.Namespace):
    from wfcllm.datasets.loaders.local import load_prompts
    from wfcllm.detection.gated_windows import GatedWindowExtractor
    from wfcllm.gate.production import (
        LocalHFProgramGenerator,
        LocalRuntimeGateBundle,
        LocalUnvalidatedRuntimeGateBundle,
        build_local_semantic_window_scorer,
        load_local_causal_rewriter,
        local_semantic_runtime_hash,
    )
    from wfcllm.generation.gated_generator import GatedGenerator
    from wfcllm.generation.gated_pipeline import (
        GatedGenerationPipeline,
        GatedGenerationPipelineConfig,
    )
    from wfcllm.generation.window_rewriter import KeyBlindWhitespaceWindowRewriter

    config = _require_gated_config(args)
    options = _local_hf_runtime_options(args, config, "generate")
    bundle_path, bundle_hash = resolve_validated_gate_bundle(args)
    deployment_key = _runtime_secret(args, "deployment")
    unvalidated = _uses_unvalidated_gate_candidate(config)
    runtime_type = (
        LocalUnvalidatedRuntimeGateBundle if unvalidated else LocalRuntimeGateBundle
    )
    runtime_bundle = runtime_type(
        root=bundle_path,
        base_model_path=options.gate_base_model_path,
        bundle_sha256=bundle_hash,
        **({"max_tokens": int(config["method"]["gate"]["max_input_tokens"])} if unvalidated else {}),
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
        rewrite_max_new_tokens=options.rewrite_max_new_tokens,
        rewrite_generation_attempts=options.rewrite_generation_attempts,
        rewrite_temperature=options.rewrite_temperature,
        rewrite_top_p=options.rewrite_top_p,
        load_in_4bit=bool(generation.get("load_in_4bit", False)),
        torch_dtype=str(generation.get("torch_dtype", "bf16")),
    )
    semantic_scorer = build_local_semantic_window_scorer(options, deployment_key)
    extractor = GatedWindowExtractor(
        runtime_bundle,
        allow_unvalidated=unvalidated,
    )
    generator = GatedGenerator(
        partitioner=extractor.partitioner,
        scorer=semantic_scorer,
        rewriter=(
            KeyBlindWhitespaceWindowRewriter()
            if options.semantic_evidence_rule == "keyed_text_region"
            else (
                program.rewriter
                if options.effective_rewrite_model_path == options.generation_model_path
                else load_local_causal_rewriter(options)
            )
        ),
        max_rewrites=int(config["method"]["rewrite"]["max_attempts"]),
    )
    dataset = str(generation.get("dataset", "humaneval"))
    dataset_path = getattr(args, "dataset_path", None) or "data/datasets"
    samples = load_prompts(
        dataset,
        dataset_path,
        sample_limit=getattr(args, "sample_limit", None),
        sample_offset=getattr(args, "sample_offset", None),
    )
    run_dir = _gate_run_dir(args, config)
    semantic_hash = local_semantic_runtime_hash(options)
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
            secret_source_type=(
                "file" if getattr(args, "secret_key_file", None) else "environment"
            ),
            unvalidated_gate_candidate=unvalidated,
        ),
        base_model=program,
        generator=generator,
        data_adapter=samples,
        deployment_key=deployment_key,
        program_finalizer=program_finalizer,
        program_finalizer_name=program_finalizer_name,
        bundle_loader=(lambda _path: runtime_bundle) if unvalidated else None,
    )


def _build_local_gated_detection_pipeline(args: argparse.Namespace):
    from wfcllm.detection.gated_pipeline import (
        GatedDetectionBindings,
        GatedDetectionPipeline,
    )
    from wfcllm.detection.gated_windows import GatedWindowExtractor
    from wfcllm.detection.scoring import GatedWindowScorer
    from wfcllm.gate.production import (
        LocalRuntimeGateBundle,
        LocalUnvalidatedRuntimeGateBundle,
        build_local_semantic_window_scorer,
        local_semantic_runtime_hash,
    )

    config = _require_gated_config(args)
    options = _local_hf_runtime_options(args, config, "detect")
    bundle_path, bundle_hash = resolve_validated_gate_bundle(args)
    deployment_key = _runtime_secret(args, "deployment")
    unvalidated = _uses_unvalidated_gate_candidate(config)
    runtime_type = (
        LocalUnvalidatedRuntimeGateBundle if unvalidated else LocalRuntimeGateBundle
    )
    runtime_bundle = runtime_type(
        root=bundle_path,
        base_model_path=options.gate_base_model_path,
        bundle_sha256=bundle_hash,
        **({"max_tokens": int(config["method"]["gate"]["max_input_tokens"])} if unvalidated else {}),
    )
    negative_hash = getattr(args, "_gated_negative_corpus_hash", None)
    if not isinstance(negative_hash, str) or _DIGEST.fullmatch(negative_hash) is None:
        negative_input = getattr(args, "negative_input", None)
        if not negative_input:
            raise ValueError("gated detector runtime requires the calibration corpus hash")
        negative_hash = _safe_file_hash(Path(negative_input))
    detector = config.get("detector")
    if not isinstance(detector, Mapping):
        raise ValueError("gated detector config is missing")
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("gated calibration config is missing")
    semantic_hash = local_semantic_runtime_hash(options)
    return GatedDetectionPipeline(
        extractor=GatedWindowExtractor(
            runtime_bundle,
            allow_unvalidated=unvalidated,
        ),
        scorer=GatedWindowScorer(
            semantic_scorer=build_local_semantic_window_scorer(options, deployment_key),
            minimum_reliable_windows=int(detector.get("minimum_reliable_windows", 2)),
        ),
        bindings=GatedDetectionBindings(
            gate_bundle_sha256=bundle_hash,
            semantic_encoder_sha256=semantic_hash,
            lsh_config_sha256=_canonical_hash(config["semantic_lsh"]),
            key_identifier_sha256=hashlib.sha256(deployment_key).hexdigest(),
            negative_corpus_manifest_sha256=negative_hash,
        ),
        target_fpr=float(detector.get("target_fpr", 0.05)),
        calibration_group_by=str(
            calibration.get("group_by", "reliable_window_count")
        ),
    )


def _gated_calibration_output(
    args: argparse.Namespace, config: Mapping[str, object]
) -> Path:
    explicit = getattr(args, "calibration", None)
    if explicit:
        return Path(explicit)
    return _gate_run_dir(args, config) / "calibration" / "reference_calibration.json"


def _gated_detection_output(
    args: argparse.Namespace, config: Mapping[str, object]
) -> Path:
    explicit = getattr(args, "positive_details", None)
    if explicit:
        return Path(explicit)
    return _gate_run_dir(args, config) / "detection" / "positive_details.jsonl"


def _gate_run_dir(args: argparse.Namespace, config: Mapping[str, object]) -> Path:
    explicit = getattr(args, "run_dir", None)
    if explicit is not None:
        if not isinstance(explicit, (str, Path)) or not str(explicit):
            raise ValueError("run_dir must be a non-empty local path")
        path = Path(explicit)
        _reject_symlink_path(path)
        return path
    run_id = getattr(args, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("gated phases require --run-id or --run-dir")
    artifacts = config.get("artifacts")
    run_root = artifacts.get("run_root") if isinstance(artifacts, Mapping) else None
    if not isinstance(run_root, str) or not run_root:
        raise ValueError("artifacts.run_root must be a non-empty local path")
    from wfcllm.method.artifacts import build_run_paths

    path = build_run_paths(run_root, run_id).run_dir
    _reject_symlink_path(path)
    return path


_GATE_MANIFEST_NAMES = {
    "gate-data": "manifest.json",
    "gate-train": "candidate_bundle_manifest.json",
    "gate-validate": "gate_bundle_manifest.json",
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


def _local_hf_runtime_options(
    args: argparse.Namespace, config: Mapping[str, object], phase: str
):
    gate_train = config.get("gate_train")
    method = config.get("method")
    rewrite = method.get("rewrite") if isinstance(method, Mapping) else None
    semantic_lsh = config.get("semantic_lsh")
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
        rewrite_model_path=_optional_runtime_path(args, "rewrite_model_path"),
        semantic_encoder_model_path=required_runtime_path("semantic_encoder_model_path"),
        semantic_encoder_checkpoint_path=_optional_runtime_path(
            args, "semantic_encoder_checkpoint_path"
        ),
        semantic_whitening_path=_optional_runtime_path(args, "semantic_whitening_path"),
        gate_base_model_path=base_model_path,
        model_device=getattr(args, "model_device", "cuda"),
        gate_device=getattr(args, "gate_device", "cuda"),
        cache_dir=Path(getattr(args, "gate_cache_dir", "data/gate-cache")),
        lsh_dimension=(
            int(semantic_lsh.get("lsh_d", 4))
            if isinstance(semantic_lsh, Mapping)
            else 4
        ),
        semantic_evidence_rule=(
            str(semantic_lsh.get("rule_name", "semantic_lsh"))
            if isinstance(semantic_lsh, Mapping)
            else "semantic_lsh"
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
        gate_epochs=_gate_training_runtime_int(
            args, config, "gate_epochs", "max_epochs", 4
        ),
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
        gate_early_stopping_patience=_gate_training_runtime_int(
            args,
            config,
            "gate_early_stopping_patience",
            "early_stopping_patience",
            1,
        ),
        gate_resume_checkpoint=_optional_runtime_path(args, "gate_resume_checkpoint"),
    )


def _gate_training_runtime_int(
    args: argparse.Namespace,
    config: Mapping[str, object],
    argument_name: str,
    config_name: str,
    default: int,
) -> int:
    override = getattr(args, argument_name, None)
    gate_train = config.get("gate_train")
    value = (
        override
        if override is not None
        else gate_train.get(config_name, default)
        if isinstance(gate_train, Mapping)
        else default
    )
    if type(value) is not int or value <= 0:
        raise ValueError(f"{argument_name} must be a positive integer")
    return value


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
    if manifest.get("diagnostic_test_backend") is not False:
        raise ValueError(f"{name} diagnostic test backend is not formal")
    formal_schemas = {
        "gate-data": "wfcllm-gate-data-manifest/v1",
        "gate-train": "wfcllm-gate-train-candidate/v1",
    }
    if name in formal_schemas:
        if manifest.get("schema_version") != formal_schemas[name]:
            raise ValueError(f"{name} manifest schema is incompatible")
        if manifest.get("formal_eligible") is not True:
            raise ValueError(f"{name} manifest is not formal eligible")
    elif "formal_eligible" in manifest and manifest.get("formal_eligible") is not True:
        raise ValueError(f"{name} manifest is not formal eligible")


def _require_experimental_manifest(
    manifest: Mapping[str, object], name: str
) -> None:
    expected_schemas = {
        "gate-data": "wfcllm-gate-data-manifest/v1",
        "gate-train": "wfcllm-gate-train-candidate/v1",
    }
    if name in expected_schemas and manifest.get("schema_version") != expected_schemas[name]:
        raise ValueError(f"{name} experimental manifest schema is incompatible")
    expected = {
        "experimental_only": True,
        "diagnostic_only": True,
        "not_official_method": True,
        "formal_eligible": False,
    }
    if any(manifest.get(field) is not value for field, value in expected.items()):
        raise ValueError(f"{name} is not an explicitly experimental artifact")


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


def run_posthoc_pass_report(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done(
        "posthoc-pass-report",
        posthoc_only=True,
        not_used_for_generation=True,
        not_used_for_retry=True,
        not_used_for_selection=True,
        not_used_for_calibration=True,
        not_used_for_detection=True,
    )
    print("=== WFCLLM posthoc pass report ===")
    return 0


def run_diagnostic_selector(args: argparse.Namespace, state: RunStateManager) -> int:
    if not getattr(args, "diagnostic_only", False):
        print("[错误] diagnostic-selector requires --diagnostic-only", file=sys.stderr)
        return 1
    state.mark_done(
        "diagnostic-selector",
        diagnostic_only=True,
        not_official_method=True,
    )
    print("=== WFCLLM diagnostic selector ===")
    return 0


def run_legacy_ablation(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-ablation",
        "historical code is under archive/legacy_wfcllm_2026_07/code/ablation "
        "and scripts/legacy/run_ablation.py is retained as guidance only.",
    )


def run_legacy_watermark(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-watermark",
        "historical code is under archive/legacy_wfcllm_2026_07/code/watermark.",
    )


def run_legacy_extract(args: argparse.Namespace, state: RunStateManager) -> int:
    if is_compare_only_mode(args):
        return _run_with_state_phase(args, state, "legacy-extract", run_extract)
    return _legacy_phase_archived(
        "legacy-extract",
        "historical code is under archive/legacy_wfcllm_2026_07/code/extract.",
    )


def run_legacy_token_channel_train(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-token-channel-train",
        "historical token-channel code is under "
        "archive/legacy_wfcllm_2026_07/code/watermark/token_channel.",
    )


def run_legacy_build_entropy_profile(
    args: argparse.Namespace,
    state: RunStateManager,
) -> int:
    return _legacy_phase_archived(
        "legacy-build-entropy-profile",
        "historical adaptive-gamma code is under "
        "archive/legacy_wfcllm_2026_07/code/watermark/adaptive_gamma.",
    )


def run_legacy_pretrain(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "legacy-pretrain",
        "historical code is under archive/legacy_wfcllm_2026_07/code/pretrain.",
    )


def run_phase(phase: str, args: argparse.Namespace, state: RunStateManager) -> int:
    """Dispatch to registered phase runners."""
    runners = {
        "generate": run_generate,
        "gate-data": run_gate_data,
        "gate-train": run_gate_train,
        "gate-validate": run_gate_validate,
        "calibrate": run_calibrate,
        "detect": run_detect,
        "report": run_report,
        "audit": run_audit,
        "posthoc-pass-report": run_posthoc_pass_report,
        "diagnostic-selector": run_diagnostic_selector,
        "encoder": run_encoder,
        "legacy-watermark": run_legacy_watermark,
        "legacy-extract": run_legacy_extract,
        "legacy-token-channel-train": run_legacy_token_channel_train,
        "legacy-build-entropy-profile": run_legacy_build_entropy_profile,
        "legacy-pretrain": run_legacy_pretrain,
        "legacy-ablation": run_legacy_ablation,
    }
    return runners[phase](args, state)


def run_encoder(args: argparse.Namespace, state: RunStateManager) -> int:
    """阶段一：训练语义编码器。"""
    import glob

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.train import main as encoder_main

    print("=== 阶段一：语义编码器预训练 ===")

    if args.eval_only:
        from wfcllm.encoder.train import evaluate_only

        default_best = str(Path(EncoderConfig().output_model_dir) / "best_model.pt")
        checkpoint = (
            args.checkpoint
            or (default_best if Path(default_best).exists() else None)
            or state.get("encoder", "checkpoint")
        )
        if not checkpoint:
            print("[错误] 未找到 checkpoint，请用 --checkpoint 指定路径", file=sys.stderr)
            return 1
        if not Path(checkpoint).exists():
            print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
            return 1
        print(f"[评测] 使用模型: {checkpoint}")

        config = EncoderConfig()
        if args.model_name:
            config.model_name = args.model_name
        if args.embed_dim:
            config.embed_dim = args.embed_dim
        if args.no_lora:
            config.use_lora = False
        if args.no_bf16:
            config.use_bf16 = False

        try:
            evaluate_only(checkpoint, config)
        except Exception as e:
            print(f"[错误] 评测失败：{e}", file=sys.stderr)
            return 1
        return 0

    config = EncoderConfig()
    if args.model_name:
        config.model_name = args.model_name
    if args.embed_dim:
        config.embed_dim = args.embed_dim
    if args.lr:
        config.lr = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.margin:
        config.margin = args.margin
    if args.no_lora:
        config.use_lora = False
    if args.no_bf16:
        config.use_bf16 = False

    if args.model_name is None:
        local_codet5 = Path(config.local_model_dir) / "codet5-base"
        if local_codet5.exists() and (local_codet5 / "config.json").exists():
            config.model_name = str(local_codet5)
            print(f"[自动] 使用本地模型: {config.model_name}")
        else:
            print(f"[回退] 使用 HF Hub 模型: {config.model_name}")

    try:
        encoder_main(config)
    except Exception as e:
        print(f"[错误] 编码器训练失败：{e}", file=sys.stderr)
        return 1

    best_model_path = str(Path(config.output_model_dir) / "best_model.pt")
    ckpt_pattern = str(Path(config.checkpoint_dir) / "encoder_epoch*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    checkpoint_path = checkpoints[-1] if checkpoints else config.checkpoint_dir

    state.mark_done("encoder", checkpoint=checkpoint_path, best_model_path=best_model_path)
    print(f"[完成] 编码器训练完毕，最优模型: {best_model_path}")
    return 0


def run_watermark(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "watermark",
        "use --phase legacy-watermark --legacy for archive guidance or the new generate phase.",
    )


def run_offline_analysis(args: argparse.Namespace) -> int:
    from wfcllm.evaluation.detection_report import (
        build_offline_regression_report,
        load_detail_artifact,
        load_summary_artifact,
        load_watermarked_artifact,
        write_offline_regression_report,
    )

    left_watermarked = (
        load_watermarked_artifact(args.compare_watermarked_left)
        if args.compare_watermarked_left
        else None
    )
    right_watermarked = (
        load_watermarked_artifact(args.compare_watermarked_right)
        if args.compare_watermarked_right
        else None
    )

    report = build_offline_regression_report(
        left_summary=load_summary_artifact(args.compare_summary_left),
        left_details=load_detail_artifact(args.compare_details_left),
        left_watermarked=left_watermarked,
        right_summary=load_summary_artifact(args.compare_summary_right),
        right_details=load_detail_artifact(args.compare_details_right),
        right_watermarked=right_watermarked,
    )
    output_path = write_offline_regression_report(args.compare_output, report)
    print(f"[完成] 离线回归报告已保存至 {output_path}")
    return 0


def run_extract(args: argparse.Namespace, state: RunStateManager) -> int:
    if is_compare_only_mode(args):
        return run_offline_analysis(args)
    return _legacy_phase_archived(
        "extract",
        "use --phase legacy-extract --legacy for archive guidance or the new detect phase.",
    )


def run_generate_negative(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "generate-negative",
        "historical negative-corpus generation is archived under "
        "archive/legacy_wfcllm_2026_07/code/extract/calibration.",
    )


def resolve_token_channel_train_config(args: argparse.Namespace) -> dict[str, object]:
    """Legacy token-channel config resolution is archived."""
    raise RuntimeError(
        "token-channel-train config resolution has been archived; see "
        "archive/legacy_wfcllm_2026_07/code/watermark/token_channel."
    )


def validate_token_channel_train_config(train_cfg: dict[str, object]) -> str | None:
    """Validate required user-facing inputs before workflow construction."""

    if not train_cfg.get("dataset"):
        return "[错误] token-channel-train 需要提供 dataset（可通过配置文件或 --dataset 指定）"
    if not train_cfg.get("lm_model_path"):
        return "[错误] token-channel-train 需要提供 lm_model_path（可通过配置文件或 --lm-model-path 指定）"
    return None


def run_token_channel_train(args: argparse.Namespace, state: RunStateManager) -> int:
    return _legacy_phase_archived(
        "token-channel-train",
        "historical token-channel training is archived under "
        "archive/legacy_wfcllm_2026_07/code/watermark/token_channel.",
    )


def run_build_entropy_profile(
    args: argparse.Namespace,
    state: RunStateManager,
) -> int:
    return _legacy_phase_archived(
        "build-entropy-profile",
        "historical adaptive-gamma profile building is archived under "
        "archive/legacy_wfcllm_2026_07/code/watermark/adaptive_gamma.",
    )
