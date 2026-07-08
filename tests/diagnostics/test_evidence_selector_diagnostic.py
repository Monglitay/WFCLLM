from __future__ import annotations

from wfcllm.diagnostics.evidence_selector import mark_evidence_selector_diagnostic


def test_mark_evidence_selector_diagnostic_sets_required_fields() -> None:
    payload = mark_evidence_selector_diagnostic(
        {"name": "evidence_only_selector_diagnostic"}
    )

    assert payload["diagnostic_only"] is True
    assert payload["not_official_method"] is True
    assert payload["name"] == "evidence_only_selector_diagnostic"
