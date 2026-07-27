from __future__ import annotations

from wfcllm.method.config import WFCLLMMethodPreset
from wfcllm.method.contracts import (
    FORBIDDEN_QUALITY_GATE_FIELDS,
    reject_quality_proxy_fields,
    require_diagnostic_marker,
)
from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset

__all__ = [
    "FORBIDDEN_QUALITY_GATE_FIELDS",
    "GATED_SEMANTIC_WINDOW_V1_NAME",
    "WFCLLMMethodPreset",
    "load_method_preset",
    "reject_quality_proxy_fields",
    "require_diagnostic_marker",
]
