from __future__ import annotations

import pytest

from wfcllm.sawr.detect.metrics import (
    auroc,
    build_detection_report,
    pearson_corr,
    quantile_summary,
    split_records_by_task,
    task_id_from_sample_id,
    wilson_ci,
    write_detection_report,
)


def _detail(
    sample_id: str,
    *,
    score: float,
    watermarked: bool,
    decision: bool,
    threshold: float = 0.5,
    insufficient: bool = False,
    passed: bool | None = None,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "score": score,
        "is_watermarked": decision,
        "threshold_5fpr": threshold,
        "p_value": 0.5,
        "scoreable_contexts": 1,
        "proxy_windows": 3,
        "direct_statements": 2,
        "code_chars": 20,
        "insufficient_evidence": insufficient,
        "label_watermarked": watermarked,
        "passed": passed,
    }


def test_task_id_from_sample_id_groups_generations_by_task() -> None:
    assert task_id_from_sample_id("HumanEval/12#sample-3") == "HumanEval/12"
    assert task_id_from_sample_id("HumanEval/12") == "HumanEval/12"


def test_split_records_by_task_keeps_task_rows_together() -> None:
    records = [
        {"id": "HumanEval/0#0"},
        {"id": "HumanEval/0#1"},
        {"id": "HumanEval/1#0"},
        {"id": "HumanEval/2#0"},
    ]

    splits = split_records_by_task(
        records,
        dev_ratio=0.25,
        calibration_ratio=0.5,
        seed=3,
    )

    assigned: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            task_id = task_id_from_sample_id(str(row["id"]))
            assert task_id not in assigned or assigned[task_id] == split_name
            assigned[task_id] = split_name
    assert sorted(row["id"] for rows in splits.values() for row in rows) == [
        "HumanEval/0#0",
        "HumanEval/0#1",
        "HumanEval/1#0",
        "HumanEval/2#0",
    ]


def test_auroc_pairwise_rank_statistic() -> None:
    assert auroc([0.8, 0.7], [0.2, 0.1]) == pytest.approx(1.0)
    assert auroc([0.2], [0.8]) == pytest.approx(0.0)
    assert auroc([0.5], [0.5]) == pytest.approx(0.5)


def test_quantile_summary_reports_core_quantiles() -> None:
    assert quantile_summary([0.0, 1.0, 2.0, 3.0]) == {
        "min": 0.0,
        "p25": 0.75,
        "p50": 1.5,
        "p75": 2.25,
        "max": 3.0,
    }


def test_wilson_ci_bounds_proportion() -> None:
    lo, hi = wilson_ci(5, 10)

    assert 0 <= lo < 0.5 < hi <= 1


def test_pearson_corr_returns_zero_for_constant_inputs() -> None:
    assert pearson_corr([1, 1, 1], [1, 2, 3]) == 0.0
    assert pearson_corr([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_build_detection_report_contains_required_metrics() -> None:
    positives = [
        _detail("pos-1", score=0.9, watermarked=True, decision=True, passed=True),
        _detail("pos-2", score=0.4, watermarked=True, decision=False, passed=False),
    ]
    negatives = [
        _detail("neg-1", score=0.3, watermarked=False, decision=False, passed=True),
        _detail("neg-2", score=0.7, watermarked=False, decision=True, passed=True),
    ]

    report = build_detection_report(positives, negatives)

    assert report["primary"]["tpr_at_5fpr"] == pytest.approx(0.5)
    assert report["primary"]["observed_fpr"] == pytest.approx(0.5)
    assert report["primary"]["auroc"] == pytest.approx(0.75)
    assert "positive_score_quantiles" in report["score_distributions"]
    assert "negative_score_quantiles" in report["score_distributions"]
    assert "score_vs_code_chars" in report["correlations"]
    assert report["pass_rates"]["positive"] == pytest.approx(0.5)
    assert report["pass_rates"]["negative"] == pytest.approx(1.0)


def test_build_detection_report_headline_rates_include_insufficient_rows() -> None:
    positives = [
        _detail("pos-1", score=0.9, watermarked=True, decision=True),
        _detail(
            "pos-2",
            score=0.2,
            watermarked=True,
            decision=False,
            insufficient=True,
        ),
    ]
    negatives = [
        _detail("neg-1", score=0.3, watermarked=False, decision=False),
        _detail(
            "neg-2",
            score=0.1,
            watermarked=False,
            decision=False,
            insufficient=True,
        ),
    ]

    report = build_detection_report(positives, negatives)

    assert report["primary"]["tpr_at_5fpr"] == pytest.approx(0.5)
    assert report["primary"]["observed_fpr"] == pytest.approx(0.0)
    assert report["primary"]["positive_sufficient_samples"] == 1
    assert report["primary"]["positive_insufficient_samples"] == 1
    assert report["primary"]["negative_sufficient_samples"] == 1
    assert report["primary"]["negative_insufficient_samples"] == 1
    assert report["score_distributions"]["positive_score_quantiles"]["max"] == 0.9


def test_build_detection_report_accepts_minimal_detail_rows() -> None:
    positives = [
        {
            "id": "pos-1",
            "score": 0.9,
            "is_watermarked": True,
            "insufficient_evidence": False,
        }
    ]
    negatives = [
        {
            "id": "neg-1",
            "score": 0.1,
            "is_watermarked": False,
            "insufficient_evidence": False,
        }
    ]

    report = build_detection_report(positives, negatives)

    assert report["primary"]["tpr_at_5fpr"] == pytest.approx(1.0)
    assert report["primary"]["observed_fpr"] == pytest.approx(0.0)
    assert report["score_distributions"]["positive_p_value_quantiles"] == {
        "min": 0.0,
        "p25": 0.0,
        "p50": 0.0,
        "p75": 0.0,
        "max": 0.0,
    }
    assert report["bucketed_fpr"]["code_chars"] == {}
    assert report["correlations"]["score_vs_code_chars"] == 0.0
    assert report["pass_rates"]["positive"] == 0.0
    assert report["data_coverage"]["code_chars_missing"] == 2


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("is_watermarked", "false"),
        ("insufficient_evidence", "false"),
        ("passed", "false"),
        ("score", True),
        ("score", "0.5"),
        ("code_chars", "20"),
    ],
)
def test_build_detection_report_rejects_malformed_detail_values(
    field: str,
    bad_value: object,
) -> None:
    row = _detail(
        "bad-1",
        score=0.3,
        watermarked=False,
        decision=False,
        passed=True,
    )
    row[field] = bad_value

    with pytest.raises(ValueError):
        build_detection_report(
            [row],
            [_detail("neg-1", score=0.1, watermarked=False, decision=False)],
        )


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"id": ""},
        {"id": 12},
    ],
)
def test_split_records_by_task_rejects_invalid_ids(
    record: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        split_records_by_task(
            [record],
            dev_ratio=0.0,
            calibration_ratio=0.0,
            seed=1,
        )


def test_package_exports_write_detection_report() -> None:
    from wfcllm.sawr.detect import write_detection_report as exported_write_report

    assert exported_write_report is write_detection_report


def test_write_detection_report_rejects_non_finite_values(tmp_path) -> None:
    report_path = tmp_path / "report.json"

    with pytest.raises(ValueError):
        write_detection_report(report_path, {"primary": {"auroc": float("nan")}})

    assert not report_path.exists()
