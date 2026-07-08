from __future__ import annotations

from wfcllm.detection.calibration import (
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
from wfcllm.detection.code_only import (
    OFFICIAL_DETECTOR_INPUT_FIELDS,
    SUPPORTED_OFFICIAL_DATASETS,
    code_from_record,
    validate_final_code_record_exact,
)
from wfcllm.detection.config import (
    DETECTOR_MODE,
    BucketEdges,
    WFCLLMDetectionConfig,
    bucket_label,
)
from wfcllm.detection.metrics import (
    build_detection_report,
    split_records_by_task,
    write_detection_report,
)
from wfcllm.detection.pipeline import (
    FORBIDDEN_DETECTOR_OUTPUT_FIELDS,
    ContextDetectionSummary,
    WFCLLMDetectionPipeline,
    WFCLLMDetectionResult,
    load_jsonl_records,
    validate_final_code_detector_input_record,
)
from wfcllm.detection.proxy_windows import (
    DirectStatement,
    ProxyWindow,
    StructureContext,
    extract_proxy_windows,
    extract_structure_contexts,
    select_target_function_name,
)
from wfcllm.detection.scoring import (
    WFCLLMWindowScorer,
    WindowEvidence,
    load_wfcllm_window_scorer,
)

__all__ = [
    "BucketEdges",
    "CalibrationArtifact",
    "ContextCalibrationInput",
    "ContextDetectionSummary",
    "DETECTOR_MODE",
    "DirectStatement",
    "FORBIDDEN_DETECTOR_OUTPUT_FIELDS",
    "OFFICIAL_DETECTOR_INPUT_FIELDS",
    "ProxyWindow",
    "SUPPORTED_OFFICIAL_DATASETS",
    "StructureContext",
    "WFCLLMDetectionConfig",
    "WFCLLMDetectionPipeline",
    "WFCLLMDetectionResult",
    "WFCLLMWindowScorer",
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
    "load_wfcllm_window_scorer",
    "null_for_context_from_artifact",
    "percentile_threshold",
    "select_target_function_name",
    "split_records_by_task",
    "validate_final_code_detector_input_record",
    "validate_final_code_record_exact",
    "write_calibration_artifact",
    "write_detection_report",
]
