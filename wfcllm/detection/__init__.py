from __future__ import annotations

from wfcllm.detection.code_only import (
    OFFICIAL_DETECTOR_INPUT_FIELDS,
    SUPPORTED_OFFICIAL_DATASETS,
    code_from_record,
    validate_final_code_record_exact,
)
from wfcllm.detection.config import GATED_DETECTOR_MODE, GatedDetectionConfig
from wfcllm.detection.gated_pipeline import (
    EMPIRICAL_P_VALUE_RULE,
    GATED_CALIBRATION_SCHEMA_VERSION,
    GATED_METHOD_NAME,
    QUANTILE_THRESHOLD_RULE,
    GatedCalibrationArtifact,
    GatedDetectionBindings,
    GatedDetectionPipeline,
    GatedDetectionResult,
    calibrate_gated_detector,
    empirical_right_tail_plus_one,
    hash_negative_corpus_manifest,
    load_gated_calibration_artifact,
    load_jsonl_records,
    write_gated_calibration_artifact,
)
from wfcllm.detection.gated_windows import (
    GatedWindowExtraction,
    GatedWindowExtractor,
    RecoveredGatedWindow,
    gate_bundle_tree_sha256,
)
from wfcllm.detection.metrics import (
    build_detection_report,
    split_records_by_task,
    write_detection_report,
)
from wfcllm.detection.scoring import (
    GatedSampleScore,
    GatedScoreDecision,
    GatedWindowEvidence,
    GatedWindowScorer,
    ScorableWindow,
)

__all__ = [
    "EMPIRICAL_P_VALUE_RULE",
    "GATED_CALIBRATION_SCHEMA_VERSION",
    "GATED_DETECTOR_MODE",
    "GATED_METHOD_NAME",
    "GatedCalibrationArtifact",
    "GatedDetectionBindings",
    "GatedDetectionConfig",
    "GatedDetectionPipeline",
    "GatedDetectionResult",
    "GatedSampleScore",
    "GatedScoreDecision",
    "GatedWindowEvidence",
    "GatedWindowExtraction",
    "GatedWindowExtractor",
    "GatedWindowScorer",
    "OFFICIAL_DETECTOR_INPUT_FIELDS",
    "QUANTILE_THRESHOLD_RULE",
    "RecoveredGatedWindow",
    "SUPPORTED_OFFICIAL_DATASETS",
    "ScorableWindow",
    "build_detection_report",
    "calibrate_gated_detector",
    "code_from_record",
    "empirical_right_tail_plus_one",
    "gate_bundle_tree_sha256",
    "hash_negative_corpus_manifest",
    "load_gated_calibration_artifact",
    "load_jsonl_records",
    "split_records_by_task",
    "validate_final_code_record_exact",
    "write_detection_report",
    "write_gated_calibration_artifact",
]
