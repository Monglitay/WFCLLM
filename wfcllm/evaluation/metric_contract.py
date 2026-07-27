from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

SCHEMA_VERSION = "wfcllm-metric-contract/v2"
AUROC_DEFINITION = "pooled-negative-statistic-rank/v1"
_METHOD = "gated_semantic_window_v1"
_DETECTOR = "wfcllm-gated-semantic-window/v1"
_CALIBRATION_SCHEMA = "wfcllm-gated-calibration/v1"
_GATE_DATA_SCHEMA = "wfcllm-gate-data-manifest/v1"
_MODEL_IDENTIFIER = re.compile(r"^.+:sha256:[0-9a-f]{64}$")


def _load_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"current {name} artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"current {name} artifact is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"current {name} artifact must be an object")
    return value


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _auroc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _negative_statistics(calibration: Mapping[str, Any]) -> list[float]:
    buckets = calibration.get("reliable_window_count_buckets")
    if not isinstance(buckets, Mapping):
        return []
    values: list[float] = []
    for bucket in buckets.values():
        if not isinstance(bucket, list):
            raise ValueError("current calibration buckets must be arrays")
        values.extend(
            _finite_number(value, "calibration background statistic")
            for value in bucket
        )
    return values


def _posthoc_pass(run_dir: Path, report: Mapping[str, Any]) -> tuple[float | None, str | None, list[str]]:
    payload = report.get("posthoc_pass_report")
    if payload is None:
        path = run_dir / "reports" / "pass_report_posthoc.json"
        if path.exists():
            payload = _load_object(path, "posthoc Pass@1")
    if payload is None:
        return None, None, ["posthoc Pass@1 artifact is not present"]
    if not isinstance(payload, Mapping):
        raise ValueError("posthoc Pass@1 artifact must be an object")
    expected_markers = {
        "posthoc_only": True,
        "not_used_for_generation": True,
        "not_used_for_retry": True,
        "not_used_for_selection": True,
        "not_used_for_calibration": True,
        "not_used_for_detection": True,
    }
    if any(payload.get(name) is not value for name, value in expected_markers.items()):
        raise ValueError("posthoc Pass@1 artifact markers are incomplete")
    if payload.get("metric") != "pass@1":
        raise ValueError("only posthoc Pass@1 is supported")
    return _finite_number(payload.get("value"), "posthoc Pass@1"), "pass@1", []


def extract_metric_contract(run_dir: Path) -> dict[str, Any]:
    """Extract one unconditional row from the current run layout only."""

    root = Path(run_dir)
    report = _load_object(
        root / "reports" / "reference_report.json", "reference report"
    )
    if report.get("method") != _METHOD or report.get("detector_mode") != _DETECTOR:
        raise ValueError("reference report is not the current Gate-only schema")
    generation_model = report.get("generation_model")
    if (
        not isinstance(generation_model, str)
        or _MODEL_IDENTIFIER.fullmatch(generation_model) is None
    ):
        raise ValueError(
            "reference report lacks a content-bound generation model identity"
        )
    _verify_audited_artifact_hashes(root)

    gate_manifest = _load_object(
        root / "gate-data" / "manifest.json", "gate-data manifest"
    )
    if (
        gate_manifest.get("schema_version") != _GATE_DATA_SCHEMA
        or gate_manifest.get("diagnostic_test_backend") is not False
        or gate_manifest.get("formal_eligible") is not True
        or gate_manifest.get("diagnostic_only") is not False
        or gate_manifest.get("not_official_method") is not False
    ):
        raise ValueError("Metric Contract requires formal current gate-data artifacts")

    calibration = report.get("calibration")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("schema_version") != _CALIBRATION_SCHEMA
    ):
        raise ValueError("reference report lacks the current calibration schema")
    thresholds = calibration.get("thresholds_by_reliable_window_count")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("current calibration thresholds are missing")
    curve = report.get("detection_curve")
    if not isinstance(curve, list):
        raise ValueError("reference report detection_curve must be an array")

    positive_scores: list[float] = []
    detected_count = 0
    insufficient_count = 0
    for row in curve:
        if not isinstance(row, Mapping):
            raise ValueError("detection_curve rows must be objects")
        decision = row.get("decision")
        if decision not in {
            "watermarked",
            "not_watermarked",
            "insufficient_evidence",
        }:
            raise ValueError("detection_curve contains an unknown decision")
        detected_count += int(decision == "watermarked")
        insufficient_count += int(decision == "insufficient_evidence")
        positive_scores.append(_finite_number(row.get("hit_rate"), "hit_rate"))

    sample_count = len(curve)
    tpr = detected_count / sample_count if sample_count else None
    pass_rate, pass_metric, caveats = _posthoc_pass(root, report)
    auroc = _auroc(positive_scores, _negative_statistics(calibration))
    if auroc is None:
        caveats.append("AUROC background statistics are not present")

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": root.name,
        "method": _METHOD,
        "detector_mode": _DETECTOR,
        "dataset": report.get("dataset"),
        "language": report.get("language"),
        "generation_model": generation_model,
        "sample_count": sample_count,
        "detected_count": detected_count,
        "insufficient_evidence_count": insufficient_count,
        "tpr": tpr,
        "tpr_is_conditional": False,
        "target_fpr": _finite_number(calibration.get("target_fpr"), "target_fpr"),
        "minimum_reliable_windows": calibration.get("minimum_reliable_windows"),
        "calibration_rule": calibration.get("empirical_p_value_rule"),
        "thresholds_by_reliable_window_count": dict(thresholds),
        "auroc": auroc,
        "auroc_definition": AUROC_DEFINITION,
        "key_identifier_sha256": calibration.get("key_identifier_sha256"),
        "semantic_encoder_sha256": calibration.get("semantic_encoder_sha256"),
        "gate_bundle_sha256": calibration.get("gate_bundle_sha256"),
        "pass_rate": pass_rate,
        "pass_metric": pass_metric,
        "caveats": caveats,
    }


def _verify_audited_artifact_hashes(run_dir: Path) -> None:
    audit = _load_object(
        run_dir / "audit" / "artifact_integrity.json",
        "artifact integrity audit",
    )
    if audit.get("ok") is not True:
        raise ValueError("Metric Contract requires a passing current audit")
    bindings = {
        "final_code_sha256": run_dir / "inputs" / "final_code.jsonl",
        "calibration_sha256": (
            run_dir / "calibration" / "reference_calibration.json"
        ),
        "positive_details_sha256": (
            run_dir / "detection" / "positive_details.jsonl"
        ),
        "reference_report_sha256": (
            run_dir / "reports" / "reference_report.json"
        ),
    }
    for field, path in bindings.items():
        if not path.is_file():
            raise ValueError(f"audited current artifact is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if audit.get(field) != actual:
            raise ValueError(f"audited current artifact hash mismatch: {field}")
