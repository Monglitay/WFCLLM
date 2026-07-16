from __future__ import annotations

import pytest

from wfcllm.audit.no_quality_gate import audit_no_quality_gate_payload


def test_audit_no_quality_gate_payload_accepts_evidence_counts() -> None:
    report = audit_no_quality_gate_payload(
        {
            "accepted_hit_count": 2,
            "closed_without_hit_count": 1,
            "fallback_count": 0,
            "candidate_count": 5,
        }
    )

    assert report == {"ok": True, "forbidden_path": None}


def test_audit_no_quality_gate_payload_rejects_nested_pass_field() -> None:
    report = audit_no_quality_gate_payload({"attempt": {"pass": True}})

    assert report["ok"] is False
    assert report["forbidden_path"] == "attempt.pass"


def test_audit_no_quality_gate_payload_rejects_list_nested_quality_field() -> None:
    report = audit_no_quality_gate_payload(
        {"attempts": [{"accepted_hit_count": 1}, {"syntax_valid": True}]}
    )

    assert report["ok"] is False
    assert report["forbidden_path"] == "attempts[1].syntax_valid"


@pytest.mark.parametrize(
    "field",
    [
        "testResults",
        "testresults",
        "syntaxValids",
        "signaturecompatible",
        "correctnessScores",
        "qualityProxies",
    ],
)
def test_no_quality_gate_rejects_plural_and_compact_proxy_aliases(
    field: str,
) -> None:
    report = audit_no_quality_gate_payload({"formal": {field: True}})

    assert report == {"ok": False, "forbidden_path": f"formal.{field}"}


def test_no_quality_gate_keeps_structural_status_fields() -> None:
    report = audit_no_quality_gate_payload(
        {"parse_status": "ok", "status": "completed", "analysis": "structural"}
    )

    assert report == {"ok": True, "forbidden_path": None}


def test_no_quality_gate_rejects_conflicting_diagnostic_identity() -> None:
    report = audit_no_quality_gate_payload(
        {
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "validated": True,
            "correctness_score": 0.5,
        }
    )

    assert report["ok"] is False
    assert "diagnostic" in str(report["forbidden_path"])


@pytest.mark.parametrize(
    ("identity_field", "identity_value"),
    [
        ("artifact_type", "gate-data-jsonl"),
        ("artifact_type", "validation-summary"),
        ("schema_version", "wfcllm-gate-data/v1"),
        ("contract_version", "wfcllm-gate-training-metrics/v1"),
    ],
)
def test_no_quality_gate_rejects_diagnostic_formal_identity_relabeling(
    identity_field: str,
    identity_value: str,
) -> None:
    report = audit_no_quality_gate_payload(
        {
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            identity_field: identity_value,
            "correctness_score": 0.5,
        }
    )

    assert report["ok"] is False
    assert "diagnostic" in str(report["forbidden_path"])
    assert "formal" in str(report["forbidden_path"])
