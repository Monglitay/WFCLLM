from __future__ import annotations

import pytest

from wfcllm.audit.artifact_integrity import (
    assert_audit_only_marker,
    assert_posthoc_pass_report_marker,
    audit_gate_artifact,
    reject_secret_key_leak,
)
from wfcllm.audit.no_quality_gate import audit_no_quality_gate_payload


def test_assert_audit_only_marker_requires_detector_input_false() -> None:
    assert_audit_only_marker({"audit_only": True, "detector_input_allowed": False})

    with pytest.raises(ValueError, match="detector_input_allowed"):
        assert_audit_only_marker({"audit_only": True, "detector_input_allowed": True})


def test_assert_audit_only_marker_requires_audit_only_true() -> None:
    with pytest.raises(ValueError, match="audit_only"):
        assert_audit_only_marker({"audit_only": False, "detector_input_allowed": False})


def test_assert_posthoc_pass_report_marker_requires_all_flags() -> None:
    payload = {
        "posthoc_only": True,
        "not_used_for_generation": True,
        "not_used_for_retry": True,
        "not_used_for_selection": True,
        "not_used_for_calibration": True,
        "not_used_for_detection": True,
    }
    assert_posthoc_pass_report_marker(payload)

    with pytest.raises(ValueError, match="not_used_for_detection"):
        assert_posthoc_pass_report_marker(
            {
                key: value
                for key, value in payload.items()
                if key != "not_used_for_detection"
            }
        )


def test_reject_secret_key_leak_finds_nested_secret_key() -> None:
    with pytest.raises(ValueError, match="calibration.secret_key"):
        reject_secret_key_leak({"calibration": {"secret_key": "1010"}})


def test_reject_secret_key_leak_finds_list_nested_raw_secret_key() -> None:
    with pytest.raises(ValueError, match=r"attempts\[0\].raw_secret_key"):
        reject_secret_key_leak({"attempts": [{"raw_secret_key": "1010"}]})


def test_deployment_key_replicate_is_public_factor_not_key_material() -> None:
    audit_gate_artifact(
        {
            "schema_version": "wfcllm-current-study-runtime/v1",
            "supplementary_experiment": {
                "study_kind": "supplementary_experiment",
            },
            "runtime": {
                "deployment_key_replicate": 2,
            },
        }
    )

    with pytest.raises(ValueError, match="deployment_key"):
        audit_gate_artifact({"runtime": {"deployment_key": 2}})


def test_posthoc_pass_report_is_allowed_only_with_required_markers() -> None:
    payload = {
        "schema_version": "wfcllm-reference-report/v1",
        "posthoc_pass_report": {
            "metric": "pass@1",
            "value": 0.5,
            "passed_count": 1,
            "total_count": 2,
            "posthoc_only": True,
            "not_used_for_generation": True,
            "not_used_for_retry": True,
            "not_used_for_selection": True,
            "not_used_for_calibration": True,
            "not_used_for_detection": True,
        },
    }

    audit_gate_artifact(payload)
    assert audit_no_quality_gate_payload(payload) == {
        "ok": True,
        "forbidden_path": None,
    }

    tampered = {
        "posthoc_pass_report": {
            key: value
            for key, value in payload["posthoc_pass_report"].items()
            if key != "not_used_for_detection"
        }
    }
    with pytest.raises(ValueError, match="not_used_for_detection"):
        audit_gate_artifact(tampered)


def test_plain_formal_pass_field_stays_forbidden() -> None:
    with pytest.raises(ValueError, match="formal quality proxy field"):
        audit_gate_artifact({"passed": True})


def test_supplementary_mechanism_funnel_fields_are_metadata_only() -> None:
    payload = {
        "schema_version": "wfcllm-reference-report/v1",
        "supplementary_ablation": {
            "study_kind": "supplementary_ablation",
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "diagnostic_only": True,
            "not_official_method": True,
            "resolved_config_hash": "a" * 64,
        },
        "mechanism_funnel": {
            "suitable_pass": {
                "count": 2,
                "denominator": 3,
            },
        },
    }

    audit_gate_artifact(payload)

    with pytest.raises(ValueError, match="formal quality proxy field"):
        audit_gate_artifact(
            {
                "schema_version": "wfcllm-reference-report/v1",
                "mechanism_funnel": {
                    "suitable_pass": {
                        "count": 2,
                        "denominator": 3,
                    },
                },
            }
        )
