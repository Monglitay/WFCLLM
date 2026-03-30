from __future__ import annotations

import json
import math
from importlib import import_module

import pytest


def _load_module():
    return import_module("wfcllm.extract.offline_analysis")


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compare_run_parameters_prefers_saved_watermarked_metadata(tmp_path):
    offline_analysis = _load_module()

    summary_path = tmp_path / "summary.json"
    watermarked_path = tmp_path / "watermarked.jsonl"

    _write_json(
        summary_path,
        {
            "dataset": "HumanEval",
            "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
        },
    )
    _write_jsonl(
        watermarked_path,
        [
            {
                "id": "HumanEval/0",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            }
        ],
    )

    summary = offline_analysis.load_summary_artifact(summary_path)
    watermarked = offline_analysis.load_watermarked_artifact(watermarked_path)

    comparison = offline_analysis.compare_run_parameters(
        summary,
        summary,
        watermarked,
        None,
    )

    assert comparison.left_source == "watermarked"
    assert comparison.left_params == {"lsh_d": 4, "lsh_gamma": 0.75}
    assert comparison.right_source == "summary"
    assert comparison.right_params == {"lsh_d": 3, "lsh_gamma": 0.5}
    assert comparison.differing_keys == ("lsh_d", "lsh_gamma")


def test_artifact_compatibility_requires_same_id_set_and_comparable_details(tmp_path):
    offline_analysis = _load_module()

    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"

    _write_jsonl(
        left_path,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": True,
                "z_score": 2.1,
                "p_value": 0.02,
                "independent_blocks": 8,
                "hits": 6,
            },
            {
                "id": "HumanEval/1",
                "is_watermarked": False,
                "z_score": 0.5,
                "p_value": 0.31,
                "independent_blocks": 8,
                "hits": 4,
            },
        ],
    )
    _write_jsonl(
        right_path,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": True,
                "z_score": 1.9,
                "p_value": 0.03,
                "independent_blocks": 8,
            }
        ],
    )

    left = offline_analysis.load_detail_artifact(left_path)
    right = offline_analysis.load_detail_artifact(right_path)
    compatibility = offline_analysis.check_artifact_compatibility(left, right)

    assert compatibility.same_id_set is False
    assert compatibility.missing_in_right == ("HumanEval/1",)
    assert compatibility.comparable_details is False
    assert compatibility.missing_detail_fields == {"HumanEval/0": ("hits",)}
    assert compatibility.is_compatible is False


def test_build_detail_delta_report_flags_true_to_false_detection_flip(tmp_path):
    offline_analysis = _load_module()

    left_path = tmp_path / "left_details.jsonl"
    right_path = tmp_path / "right_details.jsonl"

    _write_jsonl(
        left_path,
        [
            {
                "id": "HumanEval/7",
                "is_watermarked": True,
                "z_score": 2.6,
                "p_value": 0.01,
                "independent_blocks": 8,
                "hits": 6,
            }
        ],
    )
    _write_jsonl(
        right_path,
        [
            {
                "id": "HumanEval/7",
                "is_watermarked": False,
                "z_score": 1.1,
                "p_value": 0.13,
                "independent_blocks": 8,
                "hits": 5,
            }
        ],
    )

    left = offline_analysis.load_detail_artifact(left_path)
    right = offline_analysis.load_detail_artifact(right_path)
    report = offline_analysis.build_detail_delta_report(left, right)
    delta = report.deltas["HumanEval/7"]

    assert report.total_samples == 1
    assert report.detection_loss_ids == ("HumanEval/7",)
    assert delta.detection_flipped is True
    assert delta.flip_direction == "true_to_false"
    assert "detection_lost" in delta.anomaly_flags


def test_build_detail_delta_report_calculates_z_score_delta(tmp_path):
    offline_analysis = _load_module()

    left_path = tmp_path / "left_details.jsonl"
    right_path = tmp_path / "right_details.jsonl"

    _write_jsonl(
        left_path,
        [
            {
                "id": "HumanEval/3",
                "is_watermarked": True,
                "z_score": 1.8,
                "p_value": 0.04,
                "independent_blocks": 10,
                "hits": 7,
            }
        ],
    )
    _write_jsonl(
        right_path,
        [
            {
                "id": "HumanEval/3",
                "is_watermarked": True,
                "z_score": 0.9,
                "p_value": 0.12,
                "independent_blocks": 11,
                "hits": 7,
            }
        ],
    )

    left = offline_analysis.load_detail_artifact(left_path)
    right = offline_analysis.load_detail_artifact(right_path)
    report = offline_analysis.build_detail_delta_report(left, right)
    delta = report.deltas["HumanEval/3"]

    assert math.isclose(delta.z_score_delta, -0.9)
    assert math.isclose(delta.p_value_delta, 0.08)
    assert delta.independent_blocks_delta == 1
    assert delta.hits_delta == 0
    assert "z_score_drop" in delta.anomaly_flags


def test_build_detail_delta_report_rejects_non_boolean_is_watermarked(tmp_path):
    offline_analysis = _load_module()

    left_path = tmp_path / "left_details.jsonl"
    right_path = tmp_path / "right_details.jsonl"
    _write_jsonl(
        left_path,
        [
            {
                "id": "HumanEval/9",
                "is_watermarked": "false",
                "z_score": 1.0,
                "p_value": 0.5,
                "independent_blocks": 4,
                "hits": 2,
            }
        ],
    )
    _write_jsonl(
        right_path,
        [
            {
                "id": "HumanEval/9",
                "is_watermarked": False,
                "z_score": 0.9,
                "p_value": 0.6,
                "independent_blocks": 4,
                "hits": 2,
            }
        ],
    )

    left = offline_analysis.load_detail_artifact(left_path)
    right = offline_analysis.load_detail_artifact(right_path)

    with pytest.raises(ValueError, match="is_watermarked"):
        offline_analysis.build_detail_delta_report(left, right)


def test_load_detail_artifact_rejects_duplicate_ids(tmp_path):
    offline_analysis = _load_module()

    detail_path = tmp_path / "details.jsonl"
    _write_jsonl(
        detail_path,
        [
            {
                "id": "HumanEval/1",
                "is_watermarked": True,
                "z_score": 1.0,
                "p_value": 0.2,
                "independent_blocks": 4,
                "hits": 2,
            },
            {
                "id": "HumanEval/1",
                "is_watermarked": False,
                "z_score": 0.5,
                "p_value": 0.4,
                "independent_blocks": 4,
                "hits": 1,
            },
        ],
    )

    with pytest.raises(ValueError, match="duplicate id"):
        offline_analysis.load_detail_artifact(detail_path)




def test_load_watermarked_artifact_preserves_optional_route_one_summary(tmp_path):
    offline_analysis = _load_module()

    watermarked_path = tmp_path / "watermarked.jsonl"
    _write_jsonl(
        watermarked_path,
        [
            {
                "id": "HumanEval/0",
                "total_blocks": 8,
                "embedded_blocks": 6,
                "failed_blocks": 2,
                "fallback_blocks": 0,
                "embed_rate": 0.75,
                "diagnostics_version": 1,
                "retry_summary": {"blocks_with_retry": 2, "retry_exhausted_blocks": 1},
                "cascade_summary": {"cascade_triggers": 1, "cascade_rescued_blocks": 0},
                "failure_reason_counts": {"signature_miss": 3},
                "rescued_blocks": 1,
                "unrescued_blocks": 0,
            }
        ],
    )

    artifact = offline_analysis.load_watermarked_artifact(watermarked_path)

    record = artifact.records["HumanEval/0"]
    assert record["retry_summary"]["blocks_with_retry"] == 2
    assert record["cascade_summary"]["cascade_triggers"] == 1
    assert record["failure_reason_counts"]["signature_miss"] == 3


def test_load_watermarked_artifact_keeps_older_rows_compatible(tmp_path):
    offline_analysis = _load_module()

    legacy_path = tmp_path / "legacy.jsonl"
    _write_jsonl(
        legacy_path,
        [
            {
                "id": "HumanEval/1",
                "total_blocks": 4,
                "embedded_blocks": 3,
                "failed_blocks": 1,
                "fallback_blocks": 0,
                "embed_rate": 0.75,
            }
        ],
    )

    artifact = offline_analysis.load_watermarked_artifact(legacy_path)

    assert artifact.records["HumanEval/1"]["embed_rate"] == 0.75



def test_build_offline_regression_report_includes_regression_classification_keys(tmp_path):
    offline_analysis = _load_module()

    left_summary_path = tmp_path / "left_summary.json"
    right_summary_path = tmp_path / "right_summary.json"
    left_details_path = tmp_path / "left_details.jsonl"
    right_details_path = tmp_path / "right_details.jsonl"
    left_watermarked_path = tmp_path / "left_watermarked.jsonl"
    right_watermarked_path = tmp_path / "right_watermarked.jsonl"

    _write_json(
        left_summary_path,
        {
            "dataset": "HumanEval",
            "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
            "summary": {"watermark_rate": 1.0},
        },
    )
    _write_json(
        right_summary_path,
        {
            "dataset": "HumanEval",
            "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            "summary": {"watermark_rate": 0.0},
        },
    )
    _write_jsonl(
        left_details_path,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": True,
                "z_score": 2.8,
                "p_value": 0.01,
                "independent_blocks": 8,
                "hits": 6,
            }
        ],
    )
    _write_jsonl(
        right_details_path,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": False,
                "z_score": 1.1,
                "p_value": 0.12,
                "independent_blocks": 9,
                "hits": 5,
            }
        ],
    )
    _write_jsonl(
        left_watermarked_path,
        [
            {
                "id": "HumanEval/0",
                "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
                "total_blocks": 8,
                "embedded_blocks": 6,
                "failed_blocks": 0,
                "fallback_blocks": 0,
                "embed_rate": 0.75,
            }
        ],
    )
    _write_jsonl(
        right_watermarked_path,
        [
            {
                "id": "HumanEval/0",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
                "total_blocks": 8,
                "embedded_blocks": 5,
                "failed_blocks": 1,
                "fallback_blocks": 0,
                "embed_rate": 0.625,
            }
        ],
    )

    report = offline_analysis.build_offline_regression_report(
        left_summary=offline_analysis.load_summary_artifact(left_summary_path),
        left_details=offline_analysis.load_detail_artifact(left_details_path),
        left_watermarked=offline_analysis.load_watermarked_artifact(left_watermarked_path),
        right_summary=offline_analysis.load_summary_artifact(right_summary_path),
        right_details=offline_analysis.load_detail_artifact(right_details_path),
        right_watermarked=offline_analysis.load_watermarked_artifact(right_watermarked_path),
    )

    assert set(report) == {
        "compatibility",
        "parameter_diff",
        "detail_delta",
        "embedding_delta",
        "anomalies",
        "regression_classification",
    }
    assert set(report["regression_classification"]) == {
        "parameter_drift",
        "adaptive_gamma_shift",
        "extraction_conservatism",
        "calibration_drift",
        "implementation_bug",
        "recommended_branch",
    }
    assert report["regression_classification"]["recommended_branch"] in {"A", "B", "C", "stop"}






def test_build_offline_regression_report_classifies_parameter_drift_without_implementation_bug_as_branch_a(tmp_path):
    offline_analysis = _load_module()

    left_summary_path = tmp_path / "left_summary.json"
    right_summary_path = tmp_path / "right_summary.json"
    left_details_path = tmp_path / "left_details.jsonl"
    right_details_path = tmp_path / "right_details.jsonl"
    left_watermarked_path = tmp_path / "left_watermarked.jsonl"
    right_watermarked_path = tmp_path / "right_watermarked.jsonl"

    _write_json(left_summary_path, {"dataset": "HumanEval", "summary": {"watermark_rate": 0.12}})
    _write_json(right_summary_path, {"dataset": "HumanEval", "summary": {"watermark_rate": 0.05}})
    _write_jsonl(
        left_details_path,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": True,
                "z_score": 2.2,
                "p_value": 0.02,
                "independent_blocks": 8,
                "hits": 6,
            }
        ],
    )
    _write_jsonl(
        right_details_path,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": False,
                "z_score": 1.4,
                "p_value": 0.09,
                "independent_blocks": 9,
                "hits": 6,
            }
        ],
    )
    _write_jsonl(
        left_watermarked_path,
        [
            {
                "id": "HumanEval/0",
                "watermark_params": {
                    "lsh_d": 4,
                    "lsh_gamma": 0.75,
                    "adaptive_gamma": {"anchors": {"p10": 0.75}},
                },
                "total_blocks": 8,
                "embedded_blocks": 6,
                "failed_blocks": 0,
                "fallback_blocks": 0,
                "embed_rate": 0.75,
            }
        ],
    )
    _write_jsonl(
        right_watermarked_path,
        [
            {
                "id": "HumanEval/0",
                "watermark_params": {
                    "lsh_d": 5,
                    "lsh_gamma": 0.75,
                    "adaptive_gamma": {"anchors": {"p10": 0.90}},
                },
                "total_blocks": 9,
                "embedded_blocks": 6,
                "failed_blocks": 0,
                "fallback_blocks": 0,
                "embed_rate": 0.667,
            }
        ],
    )

    report = offline_analysis.build_offline_regression_report(
        left_summary=offline_analysis.load_summary_artifact(left_summary_path),
        left_details=offline_analysis.load_detail_artifact(left_details_path),
        left_watermarked=offline_analysis.load_watermarked_artifact(left_watermarked_path),
        right_summary=offline_analysis.load_summary_artifact(right_summary_path),
        right_details=offline_analysis.load_detail_artifact(right_details_path),
        right_watermarked=offline_analysis.load_watermarked_artifact(right_watermarked_path),
    )

    assert report["regression_classification"]["parameter_drift"] is True
    assert report["regression_classification"]["adaptive_gamma_shift"] is True
    assert report["regression_classification"]["implementation_bug"] is False
    assert report["regression_classification"]["recommended_branch"] == "A"


def test_build_offline_regression_report_marks_incompatible_comparison_unresolved(tmp_path):
    offline_analysis = _load_module()

    left_summary_path = tmp_path / "left_summary.json"
    right_summary_path = tmp_path / "right_summary.json"
    left_details_path = tmp_path / "left_details.jsonl"
    right_details_path = tmp_path / "right_details.jsonl"

    _write_json(left_summary_path, {"dataset": "HumanEval", "summary": {"watermark_rate": 1.0}})
    _write_json(right_summary_path, {"dataset": "HumanEval", "summary": {"watermark_rate": 0.8}})
    _write_jsonl(
        left_details_path,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": True,
                "z_score": 2.0,
                "p_value": 0.02,
                "independent_blocks": 8,
                "hits": 6,
            }
        ],
    )
    _write_jsonl(
        right_details_path,
        [
            {
                "id": "HumanEval/1",
                "is_watermarked": True,
                "z_score": 1.7,
                "p_value": 0.04,
                "independent_blocks": 8,
                "hits": 5,
            }
        ],
    )

    report = offline_analysis.build_offline_regression_report(
        left_summary=offline_analysis.load_summary_artifact(left_summary_path),
        left_details=offline_analysis.load_detail_artifact(left_details_path),
        left_watermarked=None,
        right_summary=offline_analysis.load_summary_artifact(right_summary_path),
        right_details=offline_analysis.load_detail_artifact(right_details_path),
        right_watermarked=None,
    )

    assert report["compatibility"]["is_compatible"] is False
    assert report["regression_classification"]["recommended_branch"] == "stop"
