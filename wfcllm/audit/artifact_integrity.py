from __future__ import annotations

from typing import Any

from wfcllm.audit.forbidden import FORBIDDEN_SECRET_FIELDS, POSTHOC_PASS_REQUIRED_FLAGS


def assert_audit_only_marker(payload: dict[str, Any]) -> None:
    if payload.get("audit_only") is not True:
        raise ValueError("audit_only must be true")
    if payload.get("detector_input_allowed") is not False:
        raise ValueError("detector_input_allowed must be false")


def assert_posthoc_pass_report_marker(payload: dict[str, Any]) -> None:
    for field in sorted(POSTHOC_PASS_REQUIRED_FLAGS):
        if payload.get(field) is not True:
            raise ValueError(f"{field} must be true")


def reject_secret_key_leak(payload: Any) -> None:
    path = _find_secret_path(payload, path="")
    if path is not None:
        raise ValueError(f"secret material is forbidden in public artifacts: {path}")


def _find_secret_path(value: Any, *, path: str) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}" if path else key_name
            if key_name in FORBIDDEN_SECRET_FIELDS:
                return child_path
            found = _find_secret_path(nested, path=child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found = _find_secret_path(item, path=child_path)
            if found is not None:
                return found
    return None
