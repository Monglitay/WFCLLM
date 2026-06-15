from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = ("min", "p25", "p50", "p75", "max")
BUCKET_FIELDS = (
    "code_chars",
    "direct_statements",
    "scoreable_contexts",
    "proxy_windows",
)
WILSON_95_Z = 1.959963984540054


def task_id_from_sample_id(sample_id: str) -> str:
    return sample_id.split("#", 1)[0]


def split_records_by_task(
    records: list[dict[str, Any]],
    *,
    dev_ratio: float,
    calibration_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    _validate_split_ratios(dev_ratio, calibration_ratio)

    task_records: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        task_records.setdefault(_task_id_from_record(row), []).append(row)

    task_ids = sorted(task_records)
    random.Random(seed).shuffle(task_ids)

    task_count = len(task_ids)
    dev_count = round(task_count * dev_ratio)
    calibration_count = round(task_count * calibration_ratio)
    split_task_ids = {
        "dev": task_ids[:dev_count],
        "calibration": task_ids[dev_count : dev_count + calibration_count],
        "test": task_ids[dev_count + calibration_count :],
    }

    return {
        split_name: [
            row
            for task_id in split_task_ids[split_name]
            for row in task_records[task_id]
        ]
        for split_name in ("dev", "calibration", "test")
    }


def build_detection_report(
    positive_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sufficient_positives = _sufficient_rows(positive_rows)
    sufficient_negatives = _sufficient_rows(negative_rows)
    sufficient_rows = sufficient_positives + sufficient_negatives

    true_positives = sum(
        1 for row in sufficient_positives if bool(row.get("is_watermarked"))
    )
    false_positives = sum(
        1 for row in sufficient_negatives if bool(row.get("is_watermarked"))
    )
    positive_count = len(sufficient_positives)
    negative_count = len(sufficient_negatives)

    positive_scores = _field_values(sufficient_positives, "score")
    negative_scores = _field_values(sufficient_negatives, "score")

    return {
        "primary": {
            "tpr_at_5fpr": _rate(true_positives, positive_count),
            "observed_fpr": _rate(false_positives, negative_count),
            "auroc": auroc(positive_scores, negative_scores),
            "positive_samples": len(positive_rows),
            "negative_samples": len(negative_rows),
            "positive_sufficient_samples": positive_count,
            "negative_sufficient_samples": negative_count,
        },
        "confidence_intervals": {
            "tpr_wilson_95": wilson_ci(true_positives, positive_count),
            "fpr_wilson_95": wilson_ci(false_positives, negative_count),
        },
        "score_distributions": {
            "positive_score_quantiles": quantile_summary(positive_scores),
            "negative_score_quantiles": quantile_summary(negative_scores),
            "positive_p_value_quantiles": quantile_summary(
                _field_values(sufficient_positives, "p_value")
            ),
            "negative_p_value_quantiles": quantile_summary(
                _field_values(sufficient_negatives, "p_value")
            ),
        },
        "bucketed_fpr": {
            field: bucketed_fpr(sufficient_negatives, field) for field in BUCKET_FIELDS
        },
        "pass_rates": {
            "positive": pass_rate(positive_rows),
            "negative": pass_rate(negative_rows),
        },
        "correlations": {
            f"score_vs_{field}": pearson_corr(
                _field_values(sufficient_rows, "score"),
                _field_values(sufficient_rows, field),
            )
            for field in BUCKET_FIELDS
        },
    }


def auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    positives = [float(value) for value in positive_scores]
    negatives = [float(value) for value in negative_scores]
    if not positives or not negatives:
        return 0.0

    rank_sum = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                rank_sum += 1.0
            elif positive == negative:
                rank_sum += 0.5
    return rank_sum / (len(positives) * len(negatives))


def quantile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {field: 0.0 for field in SUMMARY_FIELDS}
    sorted_values = sorted(float(value) for value in values)
    return {
        "min": sorted_values[0],
        "p25": _quantile(sorted_values, 0.25),
        "p50": _quantile(sorted_values, 0.50),
        "p75": _quantile(sorted_values, 0.75),
        "max": sorted_values[-1],
    }


def wilson_ci(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    if n < 0 or k < 0 or k > n:
        raise ValueError("Wilson interval requires 0 <= k <= n")

    proportion = k / n
    z2 = WILSON_95_Z * WILSON_95_Z
    denominator = 1 + z2 / n
    center = proportion + z2 / (2 * n)
    margin = WILSON_95_Z * math.sqrt(
        (proportion * (1 - proportion) / n) + (z2 / (4 * n * n))
    )
    return [
        max(0.0, (center - margin) / denominator),
        min(1.0, (center + margin) / denominator),
    ]


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    x_values = [float(value) for value in xs]
    y_values = [float(value) for value in ys]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_deltas = [value - x_mean for value in x_values]
    y_deltas = [value - y_mean for value in y_values]
    x_var = sum(value * value for value in x_deltas)
    y_var = sum(value * value for value in y_deltas)
    if x_var == 0 or y_var == 0:
        return 0.0
    covariance = sum(
        x_delta * y_delta for x_delta, y_delta in zip(x_deltas, y_deltas)
    )
    return covariance / math.sqrt(x_var * y_var)


def bucketed_fpr(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, dict[str, float | int]] = {}
    counts: dict[str, int] = {}
    false_positives: dict[str, int] = {}

    for row in rows:
        bucket = _size_bucket(_numeric_field(row, field))
        counts[bucket] = counts.get(bucket, 0) + 1
        if bool(row.get("is_watermarked")):
            false_positives[bucket] = false_positives.get(bucket, 0) + 1

    for bucket in sorted(counts):
        n = counts[bucket]
        buckets[bucket] = {
            "fpr": _rate(false_positives.get(bucket, 0), n),
            "n": n,
        }
    return buckets


def pass_rate(rows: list[dict[str, Any]]) -> float:
    known_rows = [row for row in rows if row.get("passed") is not None]
    if not known_rows:
        return 0.0
    passed_count = sum(1 for row in known_rows if bool(row.get("passed")))
    return passed_count / len(known_rows)


def write_detection_report(path: str | Path, report: dict[str, Any]) -> str:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(report_path)


def _quantile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * p
    lower_index = math.floor(index)
    upper_index = math.ceil(index)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = index - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _size_bucket(value: float) -> str:
    if value <= 0:
        return "0"
    if value <= 1:
        return "1"
    if value <= 2:
        return "2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    if value <= 50:
        return "26-50"
    if value <= 100:
        return "51-100"
    if value <= 250:
        return "101-250"
    if value <= 500:
        return "251-500"
    if value <= 1000:
        return "501-1000"
    return "1001+"


def _validate_split_ratios(dev_ratio: float, calibration_ratio: float) -> None:
    if not 0 <= dev_ratio <= 1:
        raise ValueError("dev_ratio must be in [0, 1]")
    if not 0 <= calibration_ratio <= 1:
        raise ValueError("calibration_ratio must be in [0, 1]")
    if dev_ratio + calibration_ratio > 1:
        raise ValueError("dev_ratio and calibration_ratio must sum to at most 1")


def _task_id_from_record(row: dict[str, Any]) -> str:
    try:
        sample_id = row["id"]
    except KeyError as exc:
        raise ValueError("record missing required id") from exc
    return task_id_from_sample_id(str(sample_id))


def _sufficient_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not bool(row.get("insufficient_evidence"))]


def _field_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [_numeric_field(row, field) for row in rows]


def _numeric_field(row: dict[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except KeyError as exc:
        raise ValueError(f"record missing required numeric field: {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record field must be numeric: {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"record field must be finite: {field}")
    return value


def _rate(k: int, n: int) -> float:
    return k / n if n else 0.0
