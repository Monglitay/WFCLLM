"""SAWR final-code detector configuration."""

from __future__ import annotations

from wfcllm.sawr.detect.config import (
    DETECTOR_MODE,
    BucketEdges,
    SawrDetectionConfig,
    bucket_label,
)
from wfcllm.sawr.detect.proxy_windows import (
    DirectStatement,
    ProxyWindow,
    StructureContext,
    extract_proxy_windows,
    extract_structure_contexts,
    select_target_function_name,
)

__all__ = [
    "DETECTOR_MODE",
    "BucketEdges",
    "DirectStatement",
    "ProxyWindow",
    "SawrDetectionConfig",
    "StructureContext",
    "bucket_label",
    "extract_proxy_windows",
    "extract_structure_contexts",
    "select_target_function_name",
]
