from __future__ import annotations

from typing import Any

from wfcllm.audit.artifact_integrity import reject_formal_quality_proxy_fields


def audit_no_quality_gate_payload(payload: Any) -> dict[str, object]:
    try:
        reject_formal_quality_proxy_fields(payload)
    except ValueError as exc:
        message = str(exc)
        marker = "formal quality proxy field is forbidden: "
        return {
            "ok": False,
            "forbidden_path": message.split(marker, 1)[1]
            if marker in message
            else message,
        }
    return {"ok": True, "forbidden_path": None}
