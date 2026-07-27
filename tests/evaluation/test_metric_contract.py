from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from wfcllm.evaluation.metric_contract import (
    AUROC_DEFINITION,
    SCHEMA_VERSION,
    extract_metric_contract,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(tmp_path: Path, *, include_pass: bool = False) -> Path:
    run = tmp_path / "fresh-run"
    _write(
        run / "gate-data" / "manifest.json",
        {
            "schema_version": "wfcllm-gate-data-manifest/v1",
            "formal_eligible": True,
            "diagnostic_test_backend": False,
            "diagnostic_only": False,
            "not_official_method": False,
        },
    )
    report = {
        "method": "gated_semantic_window_v1",
        "detector_mode": "wfcllm-gated-semantic-window/v1",
        "dataset": "humanevalpack",
        "language": "js",
        "generation_model": "local-model:sha256:" + "d" * 64,
        "calibration": {
            "schema_version": "wfcllm-gated-calibration/v1",
            "target_fpr": 0.05,
            "minimum_reliable_windows": 2,
            "key_identifier_sha256": "a" * 64,
            "semantic_encoder_sha256": "b" * 64,
            "gate_bundle_sha256": "c" * 64,
            "reliable_window_count_buckets": {"2": [0.1, 0.2, 0.5]},
            "thresholds_by_reliable_window_count": {"2": 0.5},
            "empirical_p_value_rule": "right_tail_plus_one/v1",
        },
        "detection_curve": [
            {"id": "JS/0", "decision": "watermarked", "hit_rate": 0.9},
            {"id": "JS/1", "decision": "not_watermarked", "hit_rate": 0.2},
            {"id": "JS/2", "decision": "insufficient_evidence", "hit_rate": 0.0},
            {"id": "JS/3", "decision": "watermarked", "hit_rate": 0.8},
        ],
    }
    if include_pass:
        report["posthoc_pass_report"] = {
            "metric": "pass@1",
            "value": 0.75,
            "posthoc_only": True,
            "not_used_for_generation": True,
            "not_used_for_retry": True,
            "not_used_for_selection": True,
            "not_used_for_calibration": True,
            "not_used_for_detection": True,
        }
    _write(run / "reports" / "reference_report.json", report)
    _write(
        run / "inputs" / "final_code.jsonl",
        {
            "id": "JS/0",
            "dataset": "humanevalpack",
            "prompt": "prompt",
            "final_code": "code",
        },
    )
    _write(
        run / "calibration" / "reference_calibration.json",
        report["calibration"],
    )
    _write(
        run / "detection" / "positive_details.jsonl",
        {"id": "JS/0", "decision": "watermarked"},
    )
    bindings = {
        "final_code_sha256": run / "inputs" / "final_code.jsonl",
        "calibration_sha256": (
            run / "calibration" / "reference_calibration.json"
        ),
        "positive_details_sha256": (
            run / "detection" / "positive_details.jsonl"
        ),
        "reference_report_sha256": (
            run / "reports" / "reference_report.json"
        ),
    }
    _write(
        run / "audit" / "artifact_integrity.json",
        {
            "ok": True,
            **{
                field: hashlib.sha256(path.read_bytes()).hexdigest()
                for field, path in bindings.items()
            },
        },
    )
    return run


def test_current_metric_contract_is_unconditional_and_full_sample(tmp_path) -> None:
    row = extract_metric_contract(_run(tmp_path))

    assert row["schema_version"] == SCHEMA_VERSION
    assert row["tpr"] == pytest.approx(0.5)
    assert row["tpr_is_conditional"] is False
    assert row["insufficient_evidence_count"] == 1
    assert row["auroc"] == pytest.approx(0.625)
    assert row["auroc_definition"] == AUROC_DEFINITION
    assert row["thresholds_by_reliable_window_count"] == {"2": 0.5}
    assert row["calibration_rule"] == "right_tail_plus_one/v1"
    assert row["pass_rate"] is None
    assert row["caveats"] == ["posthoc Pass@1 artifact is not present"]


def test_current_metric_contract_reports_only_posthoc_pass_at_1(tmp_path) -> None:
    row = extract_metric_contract(_run(tmp_path, include_pass=True))
    assert row["pass_metric"] == "pass@1"
    assert row["pass_rate"] == pytest.approx(0.75)
    assert row["caveats"] == []


def test_metric_contract_rejects_nonformal_current_layout(tmp_path) -> None:
    run = _run(tmp_path)
    manifest = run / "gate-data" / "manifest.json"
    _write(
        manifest,
        {
            "schema_version": "wfcllm-gate-data-manifest/v1",
            "formal_eligible": False,
            "diagnostic_test_backend": True,
            "diagnostic_only": True,
            "not_official_method": True,
        },
    )
    with pytest.raises(ValueError, match="formal current"):
        extract_metric_contract(run)


def test_metric_contract_rejects_post_audit_report_tampering(tmp_path) -> None:
    run = _run(tmp_path)
    report = run / "reports" / "reference_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["generation_model"] = "tampered:sha256:" + "e" * 64
    _write(report, payload)

    with pytest.raises(ValueError, match="hash mismatch"):
        extract_metric_contract(run)
