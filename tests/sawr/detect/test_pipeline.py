from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.sawr.detect.config import DETECTOR_MODE, SawrDetectionConfig
from wfcllm.sawr.detect.pipeline import (
    FORBIDDEN_DETECTOR_OUTPUT_FIELDS,
    SawrDetectionPipeline,
    code_from_record,
    load_jsonl_records,
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


def test_load_jsonl_records_reads_non_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")

    assert load_jsonl_records(path) == [{"id": "a"}, {"id": "b"}]


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
    assert 0 < payload["p_value"] <= 1
    assert not (FORBIDDEN_DETECTOR_OUTPUT_FIELDS & set(payload))


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
