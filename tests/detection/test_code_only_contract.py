from __future__ import annotations

import pytest

from wfcllm.detection.code_only import (
    OFFICIAL_DETECTOR_INPUT_FIELDS,
    code_from_record,
    validate_final_code_record_exact,
)


def test_validate_final_code_record_exact_accepts_only_four_fields() -> None:
    record = {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
    }

    validate_final_code_record_exact(record)
    assert set(record) == OFFICIAL_DETECTOR_INPUT_FIELDS
    assert code_from_record(record) == "def foo():\n    return 1\n"


@pytest.mark.parametrize(
    "extra_field",
    [
        "artifact_type",
        "schema_version",
        "generated_code",
        "watermark_params",
        "blocks",
        "audit",
        "audit_only",
        "detector_input_allowed",
        "generation_trace",
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
    ],
)
def test_validate_final_code_record_exact_rejects_extra_fields(
    extra_field: str,
) -> None:
    record = {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
        extra_field: "blocked",
    }

    with pytest.raises(ValueError, match=extra_field):
        validate_final_code_record_exact(record)


def test_code_from_record_rejects_generated_code_fallback() -> None:
    with pytest.raises(ValueError, match="generated_code"):
        code_from_record(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "generated_code": "    return 1\n",
            }
        )


@pytest.mark.parametrize(
    "record,message",
    [
        (["not", "a", "mapping"], "mapping"),
        (
            {
                "id": "",
                "dataset": "humaneval",
                "prompt": "",
                "final_code": "",
            },
            "id must be a non-empty string",
        ),
        (
            {
                "id": "HumanEval/0",
                "dataset": "unknown",
                "prompt": "",
                "final_code": "",
            },
            "dataset must be one of",
        ),
        (
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": 123,
                "final_code": "",
            },
            "prompt must be a string",
        ),
        (
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "",
                "final_code": None,
            },
            "final_code must be a string",
        ),
    ],
)
def test_validate_final_code_record_exact_rejects_malformed_values(
    record: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_final_code_record_exact(record)  # type: ignore[arg-type]
