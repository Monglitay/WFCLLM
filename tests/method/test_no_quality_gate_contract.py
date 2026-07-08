from __future__ import annotations

import pytest

from wfcllm.method.contracts import (
    evidence_retry_key,
    reject_quality_proxy_fields,
    require_diagnostic_marker,
)


class Result:
    accepted_hit_count = 2
    closed_without_hit_count = 1
    fallback_count = 3
    candidate_count = 8


def test_evidence_retry_key_is_exact_spec_tuple() -> None:
    assert evidence_retry_key(Result(), attempt_index=4) == (2, -1, -3, 8, -4)


@pytest.mark.parametrize(
    "field",
    [
        "pass",
        "passed",
        "correctness_result",
        "syntax_valid",
        "signature_compatible",
        "static_correctness_score",
        "generated_vs_canonical_length_gap",
        "checkpoint_correctness_proxy",
    ],
)
def test_reject_quality_proxy_fields_rejects_nested_forbidden_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        reject_quality_proxy_fields({"sample": {"metadata": {field: True}}})


def test_require_diagnostic_marker_requires_both_marker_fields() -> None:
    require_diagnostic_marker(
        {
            "diagnostic_only": True,
            "not_official_method": True,
            "name": "evidence_only_selector_diagnostic",
        }
    )

    with pytest.raises(ValueError, match="diagnostic_only"):
        require_diagnostic_marker({"not_official_method": True})

    with pytest.raises(ValueError, match="not_official_method"):
        require_diagnostic_marker({"diagnostic_only": True})
