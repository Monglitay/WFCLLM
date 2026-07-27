from __future__ import annotations

import pytest

from wfcllm.method.contracts import (
    reject_quality_proxy_fields,
    require_diagnostic_marker,
)


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
            "name": "fixture-only-artifact",
        }
    )

    with pytest.raises(ValueError, match="diagnostic_only"):
        require_diagnostic_marker({"not_official_method": True})

    with pytest.raises(ValueError, match="not_official_method"):
        require_diagnostic_marker({"diagnostic_only": True})
