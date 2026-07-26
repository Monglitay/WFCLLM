from __future__ import annotations

from collections.abc import Mapping
from typing import Any


OFFICIAL_DETECTOR_INPUT_FIELDS = {"id", "dataset", "prompt", "final_code"}
SUPPORTED_OFFICIAL_DATASETS = {"humaneval", "mbpp", "humanevalpack"}

FORBIDDEN_DETECTOR_INPUT_FIELDS = {
    "artifact_type",
    "schema_version",
    "generated_code",
    "watermark_params",
    "blocks",
    "token_channel",
    "audit",
    "audit_only",
    "detector_input_allowed",
    "logits",
    "checkpoint",
    "retry_trace",
    "rollback_trace",
    "sampling_trace",
    "candidate_sidecar",
    "p_value",
    "z_score",
    "detector_score",
    "score",
    "is_watermarked",
    "pass",
    "passed",
    "correctness_result",
}
FORBIDDEN_DETECTOR_INPUT_PREFIXES = ("generation_", "audit_")
FORBIDDEN_DETECTOR_OUTPUT_FIELDS = {
    "generation_checkpoint",
    "generation_candidate_id",
    "generation_window_id",
    "generation_layer_id",
    "audit_event_id",
    "candidate_sidecar",
    "logits",
    "retry_trace",
    "rollback_trace",
    "sampling_trace",
    "watermark_params",
    "blocks",
}


def validate_final_code_record_exact(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("detector input record must be a mapping")

    actual_fields = set(record)
    if actual_fields != OFFICIAL_DETECTOR_INPUT_FIELDS:
        extra = sorted(actual_fields - OFFICIAL_DETECTOR_INPUT_FIELDS)
        missing = sorted(OFFICIAL_DETECTOR_INPUT_FIELDS - actual_fields)
        details: list[str] = []
        if extra:
            details.append(f"forbidden extra fields: {extra}")
        if missing:
            details.append(f"missing fields: {missing}")
        raise ValueError(
            "detector input must have exactly four fields; "
            + "; ".join(details)
        )

    sample_id = record["id"]
    dataset = record["dataset"]
    prompt = record["prompt"]
    final_code = record["final_code"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("id must be a non-empty string")
    if not isinstance(dataset, str):
        raise ValueError("dataset must be a string")
    if dataset not in SUPPORTED_OFFICIAL_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(SUPPORTED_OFFICIAL_DATASETS)}"
        )
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not isinstance(final_code, str):
        raise ValueError("final_code must be a string")


def code_from_record(record: Mapping[str, Any]) -> str:
    validate_final_code_record_exact(record)
    return record["final_code"]


def reject_forbidden_detector_output_fields(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("detector output payload must be a mapping")
    forbidden_fields = FORBIDDEN_DETECTOR_OUTPUT_FIELDS & set(payload)
    if forbidden_fields:
        raise ValueError(
            "forbidden detector output fields: "
            f"{sorted(forbidden_fields)}"
        )
