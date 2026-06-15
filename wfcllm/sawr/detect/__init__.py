"""SAWR final-code detector configuration."""

from __future__ import annotations

from wfcllm.sawr.detect.calibration import (
    CalibrationArtifact,
    ContextCalibrationInput,
    build_bucket_key,
    build_calibration_artifact,
    context_score_from_null,
    empirical_upper_tail_p,
    load_calibration_artifact,
    null_for_context_from_artifact,
    percentile_threshold,
    write_calibration_artifact,
)
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
    "CalibrationArtifact",
    "ContextCalibrationInput",
    "DETECTOR_MODE",
    "BucketEdges",
    "DirectStatement",
    "ProxyWindow",
    "SawrDetectionConfig",
    "SawrWindowScorer",
    "StructureContext",
    "WindowEvidence",
    "bucket_label",
    "build_bucket_key",
    "build_calibration_artifact",
    "context_score_from_null",
    "empirical_upper_tail_p",
    "extract_proxy_windows",
    "extract_structure_contexts",
    "load_calibration_artifact",
    "load_sawr_window_scorer",
    "null_for_context_from_artifact",
    "percentile_threshold",
    "select_target_function_name",
    "write_calibration_artifact",
]
