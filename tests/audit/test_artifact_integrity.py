from __future__ import annotations

import pytest

from wfcllm.audit.artifact_integrity import (
    assert_audit_only_marker,
    assert_posthoc_pass_report_marker,
    reject_secret_key_leak,
)


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
