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
from wfcllm.sawr.detect.scoring import (
    SawrWindowScorer,
    WindowEvidence,
    load_sawr_window_scorer,
)

__all__ = [
    "DETECTOR_MODE",
    "BucketEdges",
    "DirectStatement",
    "ProxyWindow",
    "SawrDetectionConfig",
    "SawrWindowScorer",
    "StructureContext",
    "WindowEvidence",
    "bucket_label",
    "extract_proxy_windows",
    "extract_structure_contexts",
    "load_sawr_window_scorer",
    "select_target_function_name",
]
