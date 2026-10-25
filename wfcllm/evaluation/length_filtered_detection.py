"""Length- and evidence-filtered diagnostic detection metrics."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from wfcllm.detection.code_only import validate_final_code_record_exact
from wfcllm.detection.metrics import auroc


SCHEMA_VERSION = "wfcllm-length-filtered-detection-report/v1"


@dataclass(frozen=True)
class SampleFacts:
    sample_id: str
    char_len: int
    whitespace_token_len: int
    detector_window_count: int
    reliable_window_count: int
    hit_rate: float
    insufficient_evidence: bool
    decision: str
    p_value: float | None


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: Callable[[SampleFacts], bool]


FILTER_SPECS: tuple[FilterSpec, ...] = (
    FilterSpec("char_len >= 200", lambda row: row.char_len >= 200),
    FilterSpec("char_len >= 400", lambda row: row.char_len >= 400),
    FilterSpec("char_len >= 600", lambda row: row.char_len >= 600),
    FilterSpec(
        "whitespace_token_len >= 50",
        lambda row: row.whitespace_token_len >= 50,
    ),
    FilterSpec(
        "detector_window_count >= 1",
        lambda row: row.detector_window_count >= 1,
    ),
    FilterSpec(
        "reliable_window_count >= 1",
        lambda row: row.reliable_window_count >= 1,
    ),
    FilterSpec(
        "detector_window_count >= 2",
        lambda row: row.detector_window_count >= 2,
    ),
    FilterSpec(
        "char_len >= 400 AND detector_window_count >= 1",
        lambda row: row.char_len >= 400 and row.detector_window_count >= 1,
    ),
)


def build_length_filtered_detection_report(
    *,
    positive_final_code_rows: Sequence[Mapping[str, Any]],
    positive_detail_rows: Sequence[Mapping[str, Any]],
    negative_final_code_rows: Sequence[Mapping[str, Any]] | None = None,
    negative_detail_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build fixed, predeclared diagnostic TPR filters.

    The report is secondary evidence only.  It never changes decisions and never
    drops rows from the all-sample metric.
    """

    positive_facts = _join_facts(
        final_code_rows=positive_final_code_rows,
        detail_rows=positive_detail_rows,
        label="positive",
    )
    negative_facts: list[SampleFacts] | None = None
    if negative_final_code_rows is not None or negative_detail_rows is not None:
        if negative_final_code_rows is None or negative_detail_rows is None:
            raise ValueError(
                "negative final-code rows and detail rows must be provided together"
            )
        negative_facts = _join_facts(
            final_code_rows=negative_final_code_rows,
            detail_rows=negative_detail_rows,
            label="negative",
        )

    filters: dict[str, dict[str, Any]] = {}
    for spec in FILTER_SPECS:
        kept_positive = [row for row in positive_facts if spec.predicate(row)]
        dropped_positive = [row for row in positive_facts if not spec.predicate(row)]
        kept_negative = (
            [row for row in negative_facts if spec.predicate(row)]
            if negative_facts is not None
            else None
        )
        filters[spec.name] = _filter_payload(
            kept_positive=kept_positive,
            dropped_positive=dropped_positive,
            kept_negative=kept_negative,
            total_positive=len(positive_facts),
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_role": "secondary_diagnostic",
        "all_sample": {
            "positive_count": len(positive_facts),
            "detected_count": _detected_count(positive_facts),
            "TPR_all_sample": _rate(_detected_count(positive_facts), len(positive_facts)),
            "insufficient_evidence_count": _insufficient_count(positive_facts),
            "mean_final_code_chars": _mean(row.char_len for row in positive_facts),
            "mean_detector_window_count": _mean(
                row.detector_window_count for row in positive_facts
            ),
            "mean_reliable_window_count": _mean(
                row.reliable_window_count for row in positive_facts
            ),
            "mean_hit_rate": _mean(row.hit_rate for row in positive_facts),
        },
        "negative_available": negative_facts is not None,
        "filters": filters,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(value)
    return rows


def _join_facts(
    *,
    final_code_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
    label: str,
) -> list[SampleFacts]:
    details_by_id: dict[str, Mapping[str, Any]] = {}
    for detail in detail_rows:
        sample_id = _required_str(detail, "id", label)
        if sample_id in details_by_id:
            raise ValueError(f"{label} details duplicate id: {sample_id}")
        details_by_id[sample_id] = detail

    facts: list[SampleFacts] = []
    seen_ids: set[str] = set()
    for row in final_code_rows:
        validate_final_code_record_exact(dict(row))
        sample_id = str(row["id"])
        if sample_id in seen_ids:
            raise ValueError(f"{label} final-code rows duplicate id: {sample_id}")
        seen_ids.add(sample_id)
        detail = details_by_id.get(sample_id)
        if detail is None:
            raise ValueError(f"{label} details missing id: {sample_id}")
        code = str(row["final_code"])
        facts.append(
            SampleFacts(
                sample_id=sample_id,
                char_len=len(code),
                whitespace_token_len=len(code.split()),
                detector_window_count=_window_count(detail),
                reliable_window_count=_nonnegative_int(
                    detail, "reliable_window_count", label
                ),
                hit_rate=_finite_rate(detail, "hit_rate", label),
                insufficient_evidence=_bool(detail, "insufficient_evidence", label),
                decision=_required_str(detail, "decision", label),
                p_value=_optional_float(detail, "p_value", label),
            )
        )
    extra = sorted(set(details_by_id) - seen_ids)
    if extra:
        raise ValueError(f"{label} details contain ids not in final-code rows: {extra[:3]}")
    return facts


def _filter_payload(
    *,
    kept_positive: Sequence[SampleFacts],
    dropped_positive: Sequence[SampleFacts],
    kept_negative: Sequence[SampleFacts] | None,
    total_positive: int,
) -> dict[str, Any]:
    kept_count = len(kept_positive)
    dropped_count = len(dropped_positive)
    payload: dict[str, Any] = {
        "kept_count": kept_count,
        "dropped_count": dropped_count,
        "drop_rate": _rate(dropped_count, total_positive),
        "TPR_filtered": _rate(_detected_count(kept_positive), kept_count),
        "FPR_filtered": None,
        "AUROC_filtered": None,
        "mean_final_code_chars_in_kept_set": _mean(
            row.char_len for row in kept_positive
        ),
        "mean_detector_window_count_in_kept_set": _mean(
            row.detector_window_count for row in kept_positive
        ),
        "mean_reliable_window_count_in_kept_set": _mean(
            row.reliable_window_count for row in kept_positive
        ),
        "mean_hit_rate_in_kept_set": _mean(row.hit_rate for row in kept_positive),
        "insufficient_evidence_count_in_kept_set": _insufficient_count(
            kept_positive
        ),
        "insufficient_evidence_count_in_dropped_set": _insufficient_count(
            dropped_positive
        ),
    }
    if kept_negative is not None:
        payload["negative_kept_count"] = len(kept_negative)
        payload["FPR_filtered"] = _rate(
            _detected_count(kept_negative), len(kept_negative)
        )
        payload["AUROC_filtered"] = _auroc_from_rows(kept_positive, kept_negative)
    return payload


def _auroc_from_rows(
    positives: Sequence[SampleFacts],
    negatives: Sequence[SampleFacts],
) -> float | None:
    positive_scores = [_score(row) for row in positives if _score(row) is not None]
    negative_scores = [_score(row) for row in negatives if _score(row) is not None]
    if not positive_scores or not negative_scores:
        return None
    return auroc(positive_scores, negative_scores)


def _score(row: SampleFacts) -> float | None:
    if row.p_value is None:
        return None
    return -row.p_value


def _detected_count(rows: Sequence[SampleFacts]) -> int:
    return sum(1 for row in rows if row.decision == "watermarked")


def _insufficient_count(rows: Sequence[SampleFacts]) -> int:
    return sum(1 for row in rows if row.insufficient_evidence)


def _window_count(row: Mapping[str, Any]) -> int:
    windows = row.get("windows")
    if not isinstance(windows, list):
        raise ValueError("detail row windows must be a list")
    return len(windows)


def _required_str(row: Mapping[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{field} must be a non-empty string")
    return value


def _bool(row: Mapping[str, Any], field: str, label: str) -> bool:
    value = row.get(field)
    if type(value) is not bool:
        raise ValueError(f"{label}.{field} must be boolean")
    return value


def _nonnegative_int(row: Mapping[str, Any], field: str, label: str) -> int:
    value = row.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}.{field} must be a non-negative integer")
    return value


def _finite_rate(row: Mapping[str, Any], field: str, label: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{field} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label}.{field} must be in [0, 1]")
    return result


def _optional_float(row: Mapping[str, Any], field: str, label: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{field} must be numeric or null")
    return float(value)


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
