from __future__ import annotations

from typing import Any

from wfcllm.method.contracts import require_diagnostic_marker


def mark_sparse_statistics_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    marked = {
        **payload,
        "diagnostic_only": True,
        "not_official_method": True,
    }
    require_diagnostic_marker(marked)
    return marked


__all__ = ["mark_sparse_statistics_diagnostic"]
