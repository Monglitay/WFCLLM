from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from wfcllm.sawr.detect.pipeline import validate_final_code_detector_input_record


SUMMARY_FIELDS = ("min", "p25", "p50", "p75", "max")
BUCKET_FIELDS = (
    "code_chars",
    "direct_statements",
    "scoreable_contexts",
    "proxy_windows",
)
OPTIONAL_NUMERIC_FIELDS = ("p_value", "fpr_target", *BUCKET_FIELDS)
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
        validate_final_code_detector_input_record(row)
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
    positives = _validated_metric_rows(positive_rows, label="positive_rows")
    negatives = _validated_metric_rows(negative_rows, label="negative_rows")
    all_rows = positives + negatives

    true_positives = sum(1 for row in positives if row["is_watermarked"])
    false_positives = sum(1 for row in negatives if row["is_watermarked"])
    positive_count = len(positives)
    negative_count = len(negatives)
    positive_insufficient_count = sum(
        1 for row in positives if row["insufficient_evidence"]
    )
    negative_insufficient_count = sum(
        1 for row in negatives if row["insufficient_evidence"]
    )

    positive_scores = _field_values(positives, "score")
    negative_scores = _field_values(negatives, "score")
    tpr = _rate(true_positives, positive_count)
    fpr_target = _unique_fpr_target(positives + negatives)

    return {
        "primary": {
            "tpr_at_5fpr": tpr,
            "tpr_at_target_fpr": tpr,
            "fpr_target": fpr_target,
            "observed_fpr": _rate(false_positives, negative_count),
            "auroc": auroc(positive_scores, negative_scores),
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "positive_sufficient_samples": (
                positive_count - positive_insufficient_count
            ),
            "negative_sufficient_samples": (
                negative_count - negative_insufficient_count
            ),
            "positive_insufficient_samples": positive_insufficient_count,
            "negative_insufficient_samples": negative_insufficient_count,
        },
        "confidence_intervals": {
            "tpr_wilson_95": wilson_ci(true_positives, positive_count),
            "fpr_wilson_95": wilson_ci(false_positives, negative_count),
        },
        "score_distributions": {
            "positive_score_quantiles": quantile_summary(positive_scores),
            "negative_score_quantiles": quantile_summary(negative_scores),
            "positive_p_value_quantiles": quantile_summary(
                _optional_field_values(positives, "p_value")
            ),
            "negative_p_value_quantiles": quantile_summary(
                _optional_field_values(negatives, "p_value")
            ),
        },
        "bucketed_fpr": {
            field: bucketed_fpr(negatives, field) for field in BUCKET_FIELDS
        },
        "pass_rates": {
            "positive": pass_rate(positives),
            "negative": pass_rate(negatives),
        },
        "correlations": {
            f"score_vs_{field}": _pearson_for_field(all_rows, field)
            for field in BUCKET_FIELDS
        },
        "data_coverage": _data_coverage(
            positives=positives,
            negatives=negatives,
            positive_insufficient_count=positive_insufficient_count,
            negative_insufficient_count=negative_insufficient_count,
        ),
    }


def auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    positives = [
        _number_from_value(value, "positive_scores") for value in positive_scores
    ]
    negatives = [
        _number_from_value(value, "negative_scores") for value in negative_scores
    ]
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
    sorted_values = sorted(_number_from_value(value, "values") for value in values)
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

    x_values = [_number_from_value(value, "xs") for value in xs]
    y_values = [_number_from_value(value, "ys") for value in ys]
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

    for index, row in enumerate(rows):
        value = _optional_numeric_field(row, field, "rows", index)
        if value is None:
            continue
        bucket = _size_bucket(value)
        counts[bucket] = counts.get(bucket, 0) + 1
        if _required_bool_field(row, "is_watermarked", "rows", index):
            false_positives[bucket] = false_positives.get(bucket, 0) + 1

    for bucket in sorted(counts):
        n = counts[bucket]
        buckets[bucket] = {
            "fpr": _rate(false_positives.get(bucket, 0), n),
            "n": n,
        }
    return buckets


def pass_rate(rows: list[dict[str, Any]]) -> float:
    known_passed: list[bool] = []
    for index, row in enumerate(rows):
        passed = _optional_nullable_bool_field(row, "passed", "rows", index)
        if passed is not None:
            known_passed.append(passed)
    if not known_passed:
        return 0.0
    return sum(1 for value in known_passed if value) / len(known_passed)


def write_detection_report(path: str | Path, report: dict[str, Any]) -> str:
    report_path = Path(path)
    payload = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload, encoding="utf-8")
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
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("record id must be a non-empty string")
    return task_id_from_sample_id(sample_id)


def _validated_metric_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    validated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] must be a mapping")
        validated: dict[str, Any] = {
            "score": _required_numeric_field(row, "score", label, index),
            "is_watermarked": _required_bool_field(
                row,
                "is_watermarked",
                label,
                index,
            ),
            "insufficient_evidence": _optional_bool_field(
                row,
                "insufficient_evidence",
                label,
                index,
                default=False,
            ),
            "passed": _optional_nullable_bool_field(row, "passed", label, index),
        }
        for field in OPTIONAL_NUMERIC_FIELDS:
            value = _optional_numeric_field(row, field, label, index)
            if value is not None:
                validated[field] = value
        validated_rows.append(validated)
    return validated_rows


def _field_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [row[field] for row in rows]


def _optional_field_values(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    return [row[field] for row in rows if field in row]


def _paired_field_values(
    rows: list[dict[str, Any]],
    left_field: str,
    right_field: str,
) -> list[tuple[float, float]]:
    return [
        (row[left_field], row[right_field])
        for row in rows
        if left_field in row and right_field in row
    ]


def _pearson_for_field(rows: list[dict[str, Any]], field: str) -> float:
    pairs = _paired_field_values(rows, "score", field)
    if not pairs:
        return 0.0
    xs, ys = zip(*pairs, strict=True)
    return pearson_corr(list(xs), list(ys))


def _unique_fpr_target(rows: list[dict[str, Any]]) -> float:
    targets = _optional_field_values(rows, "fpr_target")
    if not targets:
        return 0.0
    unique_targets = sorted(set(targets))
    if len(unique_targets) > 1:
        raise ValueError(f"conflicting fpr_target values: {unique_targets}")
    return unique_targets[0]


def _required_numeric_field(
    row: dict[str, Any],
    field: str,
    label: str,
    index: int,
) -> float:
    try:
        value = row[field]
    except KeyError as exc:
        raise ValueError(
            f"{label}[{index}] missing required numeric field: {field}"
        ) from exc
    return _number_from_value(value, f"{label}[{index}].{field}")


def _optional_numeric_field(
    row: dict[str, Any],
    field: str,
    label: str,
    index: int,
) -> float | None:
    if field not in row:
        return None
    return _number_from_value(row[field], f"{label}[{index}].{field}")


def _number_from_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _required_bool_field(
    row: dict[str, Any],
    field: str,
    label: str,
    index: int,
) -> bool:
    try:
        value = row[field]
    except KeyError as exc:
        raise ValueError(
            f"{label}[{index}] missing required boolean field: {field}"
        ) from exc
    if not isinstance(value, bool):
        raise ValueError(f"{label}[{index}].{field} must be bool")
    return value


def _optional_bool_field(
    row: dict[str, Any],
    field: str,
    label: str,
    index: int,
    *,
    default: bool,
) -> bool:
    if field not in row:
        return default
    value = row[field]
    if not isinstance(value, bool):
        raise ValueError(f"{label}[{index}].{field} must be bool")
    return value


def _optional_nullable_bool_field(
    row: dict[str, Any],
    field: str,
    label: str,
    index: int,
) -> bool | None:
    if field not in row:
        return None
    value = row[field]
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{label}[{index}].{field} must be bool or None")
    return value


def _data_coverage(
    *,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    positive_insufficient_count: int,
    negative_insufficient_count: int,
) -> dict[str, int]:
    all_rows = positives + negatives
    total_count = len(all_rows)
    coverage = {
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "positive_sufficient_samples": len(positives) - positive_insufficient_count,
        "negative_sufficient_samples": len(negatives) - negative_insufficient_count,
        "positive_insufficient_samples": positive_insufficient_count,
        "negative_insufficient_samples": negative_insufficient_count,
        "score_present": total_count,
        "score_missing": 0,
    }
    for field in OPTIONAL_NUMERIC_FIELDS:
        present = sum(1 for row in all_rows if field in row)
        coverage[f"{field}_present"] = present
        coverage[f"{field}_missing"] = total_count - present
    passed_present = sum(1 for row in all_rows if row.get("passed") is not None)
    coverage["passed_present"] = passed_present
    coverage["passed_missing"] = total_count - passed_present
    return coverage


def _rate(k: int, n: int) -> float:
    return k / n if n else 0.0
