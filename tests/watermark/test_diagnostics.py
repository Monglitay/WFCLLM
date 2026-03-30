"""Unit tests for route-one diagnostics primitives."""

import json

from wfcllm.watermark.diagnostics import (
    BlockLifecycleRecord,
    FailureReason,
    hash_block_text,
    summarize_sample_diagnostics,
)


def _example_record():
    return BlockLifecycleRecord(
        sample_id="HumanEval/1",
        block_ordinal=1,
        initial_verify={
            "passed": False,
            "failure_reason": FailureReason.signature_miss,
        },
        retry_attempts=[
            {
                "attempt_index": 0,
                "produced_block": False,
                "failure_reason": FailureReason.no_block_generated,
            },
            {
                "attempt_index": 1,
                "produced_block": True,
                "failure_reason": None,
            },
        ],
        cascade_events=[{"triggered": True}],
        final_outcome={
            "embedded": True,
            "rescued_by_retry": True,
            "rescued_by_cascade": False,
        },
    )


def test_hash_block_text_matches_sha256():
    assert (
        hash_block_text("watermark-block")
        == "43ffcb3c60ac4321b7409161fa5fa313c9e330f03c84de55214d516f0a0e2f7a"
    )


def test_block_lifecycle_record_to_dict_contains_ids_and_reasons():
    record = _example_record()
    serialized = record.to_dict()
    assert serialized["sample_id"] == "HumanEval/1"
    assert serialized["block_ordinal"] == 1
    assert serialized["initial_verify"]["failure_reason"] == "signature_miss"


def test_block_lifecycle_record_to_dict_serializes_nested_retry_data():
    record = _example_record()
    serialized = record.to_dict()
    assert serialized["retry_attempts"][0]["failure_reason"] == "no_block_generated"
    assert serialized["retry_attempts"][1]["produced_block"] is True
    assert serialized["cascade_events"][0]["triggered"] is True


def test_block_lifecycle_record_to_dict_is_json_serializable():
    record = _example_record()
    json.dumps(record.to_dict())


def test_summarize_sample_diagnostics_counts_rescued_rollups():
    retry_rescue = BlockLifecycleRecord(
        sample_id="HumanEval/2",
        block_ordinal=2,
        initial_verify={"failure_reason": FailureReason.margin_miss},
        retry_attempts=[
            {"attempt_index": 0, "produced_block": True, "failure_reason": None},
        ],
        final_outcome={"embedded": True, "rescued_by_retry": True},
    )
    cascade_rescue = BlockLifecycleRecord(
        sample_id="HumanEval/2",
        block_ordinal=3,
        initial_verify={"failure_reason": FailureReason.signature_miss},
        retry_attempts=[],
        cascade_events=[{"triggered": True}],
        final_outcome={"embedded": True, "rescued_by_cascade": True},
    )
    summary = summarize_sample_diagnostics([retry_rescue, cascade_rescue])

    assert summary["retry_summary"]["retry_rescued_blocks"] == 1
    assert summary["cascade_summary"]["cascade_rescued_blocks"] == 1
    assert summary["rescued_blocks"] == 2


def test_rescued_blocks_counts_each_record_only_once():
    dual_rescue = BlockLifecycleRecord(
        sample_id="HumanEval/4",
        block_ordinal=5,
        initial_verify={"failure_reason": FailureReason.signature_and_margin_miss},
        final_outcome={
            "embedded": True,
            "rescued_by_retry": True,
            "rescued_by_cascade": True,
        },
    )
    summary = summarize_sample_diagnostics([dual_rescue])
    assert summary["rescued_blocks"] == 1
    assert summary["retry_summary"]["retry_rescued_blocks"] == 1
    assert summary["cascade_summary"]["cascade_rescued_blocks"] == 1


def test_cascade_replaced_failure_reason_counts():
    record = BlockLifecycleRecord(
        sample_id="HumanEval/5",
        block_ordinal=6,
        initial_verify={},
        final_outcome={
            "embedded": False,
            "failure_reason": FailureReason.cascade_replaced,
        },
    )
    summary = summarize_sample_diagnostics([record])
    assert summary["failure_reason_counts"]["cascade_replaced"] == 1


def test_successful_retry_without_failure_reason_does_not_increment_unknown():
    record = BlockLifecycleRecord(
        sample_id="HumanEval/3",
        block_ordinal=4,
        initial_verify={"failure_reason": FailureReason.signature_miss},
        retry_attempts=[{"attempt_index": 0, "produced_block": True}],
        final_outcome={"embedded": True, "rescued_by_retry": True},
    )
    summary = summarize_sample_diagnostics([record])
    assert summary["failure_reason_counts"]["unknown"] == 0


def test_retry_attempt_missing_produced_block_counts_no_block():
    record = BlockLifecycleRecord(
        sample_id="HumanEval/6",
        block_ordinal=7,
        initial_verify={},
        retry_attempts=[{"attempt_index": 0}],
    )
    summary = summarize_sample_diagnostics([record])
    assert summary["retry_summary"]["attempts_no_block"] == 1


def test_summarize_sample_diagnostics_counts():
    record = _example_record()
    summary = summarize_sample_diagnostics([record])

    assert summary["diagnostics_version"] == 1
    retry = summary["retry_summary"]
    assert retry["blocks_with_retry"] == 1
    assert retry["attempts_total"] == 2
    assert retry["attempts_no_block"] == 1

    cascade = summary["cascade_summary"]
    assert cascade["cascade_triggers"] == 1

    failure_counts = summary["failure_reason_counts"]
    assert failure_counts["signature_miss"] == 1
    assert failure_counts["no_block_generated"] == 1

    assert summary["rescued_blocks"] == 1
    assert summary["unrescued_blocks"] == 0
