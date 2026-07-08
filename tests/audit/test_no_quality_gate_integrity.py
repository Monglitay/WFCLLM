from __future__ import annotations

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
