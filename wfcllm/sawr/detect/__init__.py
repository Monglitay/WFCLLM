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
from wfcllm.sawr.detect.metrics import (
    build_detection_report,
    split_records_by_task,
    write_detection_report,
)
from wfcllm.sawr.detect.pipeline import (
    FORBIDDEN_DETECTOR_OUTPUT_FIELDS,
    ContextDetectionSummary,
    SawrDetectionPipeline,
    SawrDetectionResult,
    code_from_record,
    load_jsonl_records,
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
    "ContextDetectionSummary",
    "DETECTOR_MODE",
    "FORBIDDEN_DETECTOR_OUTPUT_FIELDS",
    "BucketEdges",
    "DirectStatement",
    "ProxyWindow",
    "SawrDetectionConfig",
    "SawrDetectionPipeline",
    "SawrDetectionResult",
    "SawrWindowScorer",
    "StructureContext",
    "WindowEvidence",
    "bucket_label",
    "build_bucket_key",
    "build_calibration_artifact",
    "build_detection_report",
    "code_from_record",
    "context_score_from_null",
    "empirical_upper_tail_p",
    "extract_proxy_windows",
    "extract_structure_contexts",
    "load_calibration_artifact",
    "load_jsonl_records",
    "load_sawr_window_scorer",
    "null_for_context_from_artifact",
    "percentile_threshold",
    "select_target_function_name",
    "split_records_by_task",
    "write_calibration_artifact",
    "write_detection_report",
]
