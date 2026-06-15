from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.sawr.detect.config import DETECTOR_MODE, BucketEdges, SawrDetectionConfig
from wfcllm.sawr.detect.pipeline import (
    FORBIDDEN_DETECTOR_OUTPUT_FIELDS,
    ContextDetectionSummary,
    SawrDetectionPipeline,
    SawrDetectionResult,
    code_from_record,
    load_jsonl_records,
    validate_final_code_detector_input_record,
)
from wfcllm.sawr.detect.proxy_windows import ProxyWindow
from wfcllm.sawr.detect.scoring import WindowEvidence


class FakeScorer:
    def score_window(self, window: ProxyWindow) -> WindowEvidence:
        raw_by_text = {
            "x = 1": 0.1,
            "return x": 0.6,
            "x = 1\nreturn x": 0.9,
            "x = 0": 0.0,
            "return 0": 0.2,
            "x = 0\nreturn 0": 0.3,
        }
        raw = raw_by_text.get(window.normalized_text, 0.0)
        return WindowEvidence(
            window_id=window.window_id,
            context_id=window.context_id,
            in_valid_set=raw > 0,
            passed_margin=raw > 0,
            min_margin=raw,
            lsh_signature=(1, 0, 1, 0),
            parent_node_type=window.parent_node_type,
            window_length=window.window_length,
            structure_type=window.structure_type,
            context_window_count=window.context_window_count,
            context_statement_count=window.context_statement_count,
            window_raw=raw,
        )


def _row(sample_id: str, code: str) -> dict[str, object]:
    return {
        "artifact_type": "sawr_final_code",
        "schema_version": "sawr-smoke/v1",
        "id": sample_id,
        "dataset": "humaneval",
        "prompt": "def target():\n",
        "final_code": code,
        "scientific_claims_enabled": False,
    }


def test_code_from_record_accepts_final_code_and_generated_code() -> None:
    assert code_from_record({"final_code": "x = 1"}) == "x = 1"
    assert code_from_record({"generated_code": "x = 2"}) == "x = 2"

    with pytest.raises(ValueError, match="record must contain final_code or generated_code"):
        code_from_record({"id": "missing"})


def test_code_from_record_combines_prompt_with_generated_body_when_needed() -> None:
    assert code_from_record(
        {
            "prompt": "def target():\n",
            "generated_code": "    x = 1\n    return x\n",
        }
    ) == "def target():\n    x = 1\n    return x\n"


def test_code_from_record_does_not_duplicate_full_generated_source() -> None:
    full_source = "def target():\n    x = 1\n    return x\n"

    assert code_from_record(
        {
            "prompt": "def target():\n",
            "generated_code": full_source,
        }
    ) == full_source


def test_code_from_record_rejects_non_string_code_fields() -> None:
    with pytest.raises(ValueError, match="final_code must be a string"):
        code_from_record({"final_code": ["x = 1"]})

    with pytest.raises(ValueError, match="generated_code must be a string"):
        code_from_record({"generated_code": ["x = 2"]})


def test_validate_final_code_detector_input_record_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="record id must be a non-empty string"):
        validate_final_code_detector_input_record({"id": "", "final_code": "pass"})


@pytest.mark.parametrize(
    "record,field",
    [
        (
            {
                "id": "audit",
                "artifact_type": "sawr_audit_event",
                "final_code": "def target():\n    return 0\n",
            },
            "artifact_type",
        ),
        (
            {
                "id": "audit-only",
                "audit_only": True,
                "final_code": "def target():\n    return 0\n",
            },
            "audit_only",
        ),
        (
            {
                "id": "blocked",
                "detector_input_allowed": False,
                "final_code": "def target():\n    return 0\n",
            },
            "detector_input_allowed",
        ),
        (
            {
                "id": "trace",
                "final_code": "def target():\n    return 0\n",
                "retry_trace": [],
            },
            "retry_trace",
        ),
        (
            {
                "id": "nested-trace",
                "final_code": "def target():\n    return 0\n",
                "metadata": {"audit_event_id": "event-1"},
            },
            "audit_event_id",
        ),
        (
            {
                "id": "generation-prefix",
                "final_code": "def target():\n    return 0\n",
                "metadata": {"generation_score": 0.2},
            },
            "generation_score",
        ),
    ],
)
def test_validate_final_code_detector_input_record_rejects_audit_and_trace_fields(
    record: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_final_code_detector_input_record(record)


def test_validate_final_code_detector_input_record_allows_generated_code_name() -> None:
    validate_final_code_detector_input_record(
        {
            "id": "ok",
            "generated_code": "def target():\n    return 0\n",
        }
    )


def test_package_exports_final_code_input_validator() -> None:
    from wfcllm.sawr.detect import (
        validate_final_code_detector_input_record as exported_validator,
    )

    assert exported_validator is validate_final_code_detector_input_record


def test_load_jsonl_records_reads_non_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")

    assert load_jsonl_records(path) == [{"id": "a"}, {"id": "b"}]


def test_load_jsonl_records_wraps_malformed_json_with_path_and_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n{"id": \n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid JSONL record .*rows\.jsonl:2"):
        load_jsonl_records(path)


@pytest.mark.parametrize("payload", ['["a"]\n', '"a"\n'])
def test_load_jsonl_records_rejects_non_object_rows_with_path_and_line(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=r"JSONL record must be an object .*rows\.jsonl:1"):
        load_jsonl_records(path)


def test_pipeline_calibrates_and_detects_without_trace_fields(tmp_path: Path) -> None:
    config = SawrDetectionConfig(secret_key="1010", target_fpr=0.05)
    pipeline = SawrDetectionPipeline(config=config, scorer=FakeScorer())
    negatives = [
        _row("neg-1", "def target():\n    x = 0\n    return 0\n"),
        _row("neg-2", "def target():\n    x = 0\n    return 0\n"),
    ]

    artifact = pipeline.calibrate(negatives)
    result = pipeline.detect_one(
        _row("pos-1", "def target():\n    x = 1\n    return x\n"),
        artifact=artifact,
    )

    payload = result.to_dict()
    assert payload["id"] == "pos-1"
    assert payload["detector_mode"] == DETECTOR_MODE
    assert payload["scoreable_contexts"] == 1
    assert payload["proxy_windows"] == 3
    assert payload["insufficient_evidence"] is False
    assert payload["score"] >= 0
    assert payload["threshold_5fpr"] == artifact.threshold_5fpr
    assert payload["threshold_at_target_fpr"] == artifact.threshold_5fpr
    assert 0 < payload["p_value"] <= 1
    assert not (FORBIDDEN_DETECTOR_OUTPUT_FIELDS & set(payload))


def test_pipeline_rejects_trace_rows_during_calibration() -> None:
    pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="1010"),
        scorer=FakeScorer(),
    )

    with pytest.raises(ValueError, match="watermark_params"):
        pipeline.calibrate(
            [
                {
                    **_row("neg", "def target():\n    x = 0\n    return 0\n"),
                    "watermark_params": {"gamma": 0.75},
                }
            ]
        )


def test_pipeline_rejects_audit_rows_during_detection() -> None:
    pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="1010"),
        scorer=FakeScorer(),
    )
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])

    with pytest.raises(ValueError, match="sawr_audit_event"):
        pipeline.detect_one(
            {
                "artifact_type": "sawr_audit_event",
                "id": "audit",
                "audit_only": True,
                "detector_input_allowed": False,
                "final_code": "def target():\n    x = 1\n    return x\n",
            },
            artifact=artifact,
        )


def test_pipeline_rejects_trace_rows_when_writing_detection_jsonl(
    tmp_path: Path,
) -> None:
    pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="1010"),
        scorer=FakeScorer(),
    )
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])

    with pytest.raises(ValueError, match="sampling_trace"):
        pipeline.detect_to_jsonl(
            [
                {
                    **_row("pos", "def target():\n    x = 1\n    return x\n"),
                    "sampling_trace": [],
                }
            ],
            artifact=artifact,
            output_path=tmp_path / "details.jsonl",
        )


def test_pipeline_classifies_by_p_value_not_zero_threshold_tie() -> None:
    config = SawrDetectionConfig(secret_key="1010", statistic="raw_context_max")
    pipeline = SawrDetectionPipeline(config=config, scorer=FakeScorer())
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    y = 99\n    return y\n")
    ])

    result = pipeline.detect_one(
        _row("held-out", "def target():\n    y = 99\n    return y\n"),
        artifact=artifact,
    )

    assert artifact.threshold_5fpr == 0.0
    assert result.score == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert result.is_watermarked is False


def test_pipeline_scores_generated_code_body_with_prompt() -> None:
    pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="1010"),
        scorer=FakeScorer(),
    )
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])

    result = pipeline.detect_one(
        {
            "artifact_type": "sawr_final_code",
            "id": "body",
            "prompt": "def target():\n",
            "generated_code": "    x = 1\n    return x\n",
        },
        artifact=artifact,
    )

    assert result.scoreable_contexts == 1
    assert result.proxy_windows == 3
    assert result.insufficient_evidence is False


def test_pipeline_rejects_artifact_with_mismatched_statistic() -> None:
    raw_config = SawrDetectionConfig(secret_key="1010", statistic="raw_context_max")
    raw_pipeline = SawrDetectionPipeline(config=raw_config, scorer=FakeScorer())
    artifact = raw_pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])
    default_pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="1010"),
        scorer=FakeScorer(),
    )

    with pytest.raises(ValueError, match="statistic"):
        default_pipeline.detect_one(
            _row("pos", "def target():\n    x = 1\n    return x\n"),
            artifact=artifact,
        )


def test_pipeline_rejects_artifact_with_mismatched_secret_key() -> None:
    pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="1010"),
        scorer=FakeScorer(),
    )
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])
    mismatched_pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="different"),
        scorer=FakeScorer(),
    )

    with pytest.raises(ValueError, match="secret_key_sha256"):
        mismatched_pipeline.detect_one(
            _row("pos", "def target():\n    x = 1\n    return x\n"),
            artifact=artifact,
        )


def test_pipeline_rejects_artifact_with_mismatched_bucket_edges() -> None:
    pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(secret_key="1010"),
        scorer=FakeScorer(),
    )
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])
    mismatched_pipeline = SawrDetectionPipeline(
        config=SawrDetectionConfig(
            secret_key="1010",
            bucket_edges=BucketEdges(window_count=(1, 3, 6, 12, 24)),
        ),
        scorer=FakeScorer(),
    )

    with pytest.raises(ValueError, match="bucket_edges"):
        mismatched_pipeline.detect_one(
            _row("pos", "def target():\n    x = 1\n    return x\n"),
            artifact=artifact,
        )


def test_pipeline_calibrates_raw_context_max_on_raw_score_scale() -> None:
    config = SawrDetectionConfig(
        secret_key="1010",
        statistic="raw_context_max",
        target_fpr=0.5,
    )
    pipeline = SawrDetectionPipeline(config=config, scorer=FakeScorer())

    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])
    result = pipeline.detect_one(
        _row("pos", "def target():\n    x = 1\n    return x\n"),
        artifact=artifact,
    )

    assert artifact.sample_scores == pytest.approx([0.3])
    assert artifact.threshold_5fpr == pytest.approx(0.3)
    assert result.score == pytest.approx(0.9)
    assert result.threshold_5fpr == pytest.approx(0.3)
    assert result.p_value == pytest.approx(0.5)
    assert result.is_watermarked is True


def test_pipeline_calibrates_context_mean_window_evidence_on_mean_score_scale() -> None:
    config = SawrDetectionConfig(
        secret_key="1010",
        statistic="context_mean_window_evidence",
        target_fpr=0.5,
    )
    pipeline = SawrDetectionPipeline(config=config, scorer=FakeScorer())

    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])
    result = pipeline.detect_one(
        _row("pos", "def target():\n    x = 1\n    return x\n"),
        artifact=artifact,
    )

    assert artifact.sample_scores == pytest.approx([(0.0 + 0.2 + 0.3) / 3])
    assert artifact.threshold_5fpr == pytest.approx((0.0 + 0.2 + 0.3) / 3)
    assert result.score == pytest.approx((0.1 + 0.6 + 0.9) / 3)
    assert result.threshold_5fpr == pytest.approx((0.0 + 0.2 + 0.3) / 3)
    assert result.p_value == pytest.approx(0.5)
    assert result.is_watermarked is True


def test_pipeline_marks_insufficient_evidence() -> None:
    config = SawrDetectionConfig(secret_key="1010", min_proxy_windows=2)
    pipeline = SawrDetectionPipeline(config=config, scorer=FakeScorer())
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])

    result = pipeline.detect_one(
        _row("short", "def target():\n    return 1\n"),
        artifact=artifact,
    )

    assert result.insufficient_evidence is True
    assert result.is_watermarked is False


def test_pipeline_writes_detection_details_jsonl(tmp_path: Path) -> None:
    config = SawrDetectionConfig(secret_key="1010")
    pipeline = SawrDetectionPipeline(config=config, scorer=FakeScorer())
    artifact = pipeline.calibrate([
        _row("neg", "def target():\n    x = 0\n    return 0\n")
    ])
    output = tmp_path / "details.jsonl"

    pipeline.detect_to_jsonl(
        [_row("pos", "def target():\n    x = 1\n    return x\n")],
        artifact=artifact,
        output_path=output,
    )

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["id"] == "pos"
    assert rows[0]["detector_mode"] == DETECTOR_MODE


def test_detection_result_to_dict_rejects_non_json_safe_floats() -> None:
    result = SawrDetectionResult(
        id="nan",
        is_watermarked=False,
        score=float("nan"),
        threshold_5fpr=0.0,
        threshold_at_target_fpr=0.0,
        p_value=1.0,
        fpr_target=0.05,
        scoreable_contexts=1,
        proxy_windows=3,
        insufficient_evidence=False,
        detector_mode=DETECTOR_MODE,
        context_summaries=(
            ContextDetectionSummary(
                context_id="ctx",
                structure_type="function_body",
                parent_node_type="function_definition",
                context_raw=float("inf"),
                context_score=0.0,
                calibration_bucket_level="global",
                proxy_windows=3,
                direct_statements=2,
            ),
        ),
        code_chars=10,
        direct_statements=2,
    )

    with pytest.raises(ValueError, match="detector output must be JSON-safe"):
        result.to_dict()
