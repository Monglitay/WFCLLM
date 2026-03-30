"""Offline helpers for comparing saved watermark extraction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

DETAIL_FIELDS = (
    "is_watermarked",
    "z_score",
    "p_value",
    "independent_blocks",
    "hits",
)
EMBEDDING_FIELDS = (
    "total_blocks",
    "embedded_blocks",
    "failed_blocks",
    "fallback_blocks",
    "embed_rate",
)
ROUTE_ONE_COUNT_FIELDS = (
    "retry_summary",
    "cascade_summary",
    "failure_reason_counts",
)
ROUTE_ONE_SCALAR_FIELDS = (
    "rescued_blocks",
    "unrescued_blocks",
)


@dataclass(frozen=True)
class SummaryArtifact:
    path: Path
    payload: dict[str, Any]
    dataset: str | None
    watermark_params: dict[str, Any]


@dataclass(frozen=True)
class DetailArtifact:
    path: Path
    records: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class WatermarkedArtifact:
    path: Path
    records: dict[str, dict[str, Any]]
    watermark_params: dict[str, Any]


@dataclass(frozen=True)
class RunParameterComparison:
    left_source: str
    left_params: dict[str, Any]
    right_source: str
    right_params: dict[str, Any]
    differing_keys: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactCompatibility:
    same_id_set: bool
    missing_in_left: tuple[str, ...]
    missing_in_right: tuple[str, ...]
    comparable_details: bool
    missing_detail_fields: dict[str, tuple[str, ...]]

    @property
    def is_compatible(self) -> bool:
        return self.same_id_set and self.comparable_details


@dataclass(frozen=True)
class DetailDelta:
    sample_id: str
    left_is_watermarked: bool
    right_is_watermarked: bool
    detection_flipped: bool
    flip_direction: str | None
    z_score_delta: float
    p_value_delta: float
    independent_blocks_delta: int
    hits_delta: int
    anomaly_flags: tuple[str, ...]


@dataclass(frozen=True)
class DetailDeltaReport:
    deltas: dict[str, DetailDelta]
    detection_loss_ids: tuple[str, ...]

    @property
    def total_samples(self) -> int:
        return len(self.deltas)


def _empty_detail_delta_report() -> DetailDeltaReport:
    return DetailDeltaReport(deltas={}, detection_loss_ids=())


def load_summary_artifact(path: str | Path) -> SummaryArtifact:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    watermark_params = payload.get("watermark_params") or {}
    if not isinstance(watermark_params, dict):
        watermark_params = {}
    return SummaryArtifact(
        path=artifact_path,
        payload=payload,
        dataset=payload.get("dataset"),
        watermark_params=dict(watermark_params),
    )


def load_detail_artifact(path: str | Path) -> DetailArtifact:
    artifact_path = Path(path)
    return DetailArtifact(path=artifact_path, records=_load_jsonl_records(artifact_path))


def load_watermarked_artifact(path: str | Path) -> WatermarkedArtifact:
    artifact_path = Path(path)
    records = _load_jsonl_records(artifact_path)
    watermark_params = _extract_watermark_params(records)
    return WatermarkedArtifact(
        path=artifact_path,
        records=records,
        watermark_params=watermark_params,
    )


def compare_run_parameters(
    left_summary: SummaryArtifact | None,
    right_summary: SummaryArtifact | None,
    left_watermarked: WatermarkedArtifact | None,
    right_watermarked: WatermarkedArtifact | None,
) -> RunParameterComparison:
    left_source, left_params = _preferred_params(left_summary, left_watermarked)
    right_source, right_params = _preferred_params(right_summary, right_watermarked)
    differing_keys = tuple(
        sorted(
            key
            for key in set(left_params) | set(right_params)
            if left_params.get(key) != right_params.get(key)
        )
    )
    return RunParameterComparison(
        left_source=left_source,
        left_params=left_params,
        right_source=right_source,
        right_params=right_params,
        differing_keys=differing_keys,
    )


def check_artifact_compatibility(
    left: DetailArtifact,
    right: DetailArtifact,
) -> ArtifactCompatibility:
    left_ids = set(left.records)
    right_ids = set(right.records)
    shared_ids = tuple(sorted(left_ids & right_ids))

    missing_detail_fields: dict[str, tuple[str, ...]] = {}
    for sample_id in shared_ids:
        missing = tuple(
            field_name
            for field_name in DETAIL_FIELDS
            if field_name not in left.records[sample_id]
            or field_name not in right.records[sample_id]
        )
        if missing:
            missing_detail_fields[sample_id] = missing

    return ArtifactCompatibility(
        same_id_set=left_ids == right_ids,
        missing_in_left=tuple(sorted(right_ids - left_ids)),
        missing_in_right=tuple(sorted(left_ids - right_ids)),
        comparable_details=not missing_detail_fields,
        missing_detail_fields=missing_detail_fields,
    )


def build_detail_delta_report(
    left: DetailArtifact,
    right: DetailArtifact,
) -> DetailDeltaReport:
    compatibility = check_artifact_compatibility(left, right)
    if not compatibility.is_compatible:
        raise ValueError("detail artifacts are not directly comparable")

    deltas: dict[str, DetailDelta] = {}
    detection_loss_ids: list[str] = []
    for sample_id in sorted(left.records):
        left_record = left.records[sample_id]
        right_record = right.records[sample_id]
        detail_delta = _build_detail_delta(sample_id, left_record, right_record)
        deltas[sample_id] = detail_delta
        if detail_delta.flip_direction == "true_to_false":
            detection_loss_ids.append(sample_id)

    return DetailDeltaReport(
        deltas=deltas,
        detection_loss_ids=tuple(detection_loss_ids),
    )


def _build_detail_delta(
    sample_id: str,
    left_record: dict[str, Any],
    right_record: dict[str, Any],
) -> DetailDelta:
    left_is_watermarked = _require_boolean_field(left_record, "is_watermarked")
    right_is_watermarked = _require_boolean_field(right_record, "is_watermarked")
    detection_flipped = left_is_watermarked != right_is_watermarked
    flip_direction: str | None = None
    anomaly_flags: list[str] = []
    if detection_flipped:
        flip_direction = (
            "true_to_false" if left_is_watermarked and not right_is_watermarked else "false_to_true"
        )
    if flip_direction == "true_to_false":
        anomaly_flags.append("detection_lost")

    z_score_delta = float(right_record["z_score"]) - float(left_record["z_score"])
    p_value_delta = float(right_record["p_value"]) - float(left_record["p_value"])
    independent_blocks_delta = int(right_record["independent_blocks"]) - int(
        left_record["independent_blocks"]
    )
    hits_delta = int(right_record["hits"]) - int(left_record["hits"])

    if z_score_delta < 0:
        anomaly_flags.append("z_score_drop")
    if independent_blocks_delta > 0 and hits_delta <= 0:
        anomaly_flags.append("block_growth_without_hit_gain")

    return DetailDelta(
        sample_id=sample_id,
        left_is_watermarked=left_is_watermarked,
        right_is_watermarked=right_is_watermarked,
        detection_flipped=detection_flipped,
        flip_direction=flip_direction,
        z_score_delta=z_score_delta,
        p_value_delta=p_value_delta,
        independent_blocks_delta=independent_blocks_delta,
        hits_delta=hits_delta,
        anomaly_flags=tuple(anomaly_flags),
    )




@dataclass(frozen=True)
class EmbeddingDelta:
    sample_id: str
    total_blocks_delta: int
    embedded_blocks_delta: int
    failed_blocks_delta: int
    fallback_blocks_delta: int
    embed_rate_delta: float
    route_one_summary: dict[str, Any] | None = None


def build_offline_regression_report(
    *,
    left_summary: SummaryArtifact | None,
    left_details: DetailArtifact,
    left_watermarked: WatermarkedArtifact | None,
    right_summary: SummaryArtifact | None,
    right_details: DetailArtifact,
    right_watermarked: WatermarkedArtifact | None,
) -> dict[str, Any]:
    parameter_comparison = compare_run_parameters(
        left_summary,
        right_summary,
        left_watermarked,
        right_watermarked,
    )
    compatibility = check_artifact_compatibility(left_details, right_details)
    detail_delta_report = (
        build_detail_delta_report(left_details, right_details)
        if compatibility.is_compatible
        else _empty_detail_delta_report()
    )
    embedding_delta = _build_embedding_delta_report(left_watermarked, right_watermarked)
    anomalies = _build_anomalies(detail_delta_report, embedding_delta)
    regression_classification = _classify_regression(
        parameter_comparison,
        detail_delta_report,
        compatibility,
        left_summary,
        right_summary,
    )

    return {
        "compatibility": {
            "is_compatible": compatibility.is_compatible,
            "same_id_set": compatibility.same_id_set,
            "missing_in_left": list(compatibility.missing_in_left),
            "missing_in_right": list(compatibility.missing_in_right),
            "comparable_details": compatibility.comparable_details,
            "missing_detail_fields": {
                sample_id: list(fields)
                for sample_id, fields in compatibility.missing_detail_fields.items()
            },
        },
        "parameter_diff": {
            "left_source": parameter_comparison.left_source,
            "left_params": parameter_comparison.left_params,
            "right_source": parameter_comparison.right_source,
            "right_params": parameter_comparison.right_params,
            "differing_keys": list(parameter_comparison.differing_keys),
        },
        "detail_delta": {
            "total_samples": detail_delta_report.total_samples,
            "detection_loss_ids": list(detail_delta_report.detection_loss_ids),
            "samples": {
                sample_id: {
                    "left_is_watermarked": delta.left_is_watermarked,
                    "right_is_watermarked": delta.right_is_watermarked,
                    "detection_flipped": delta.detection_flipped,
                    "flip_direction": delta.flip_direction,
                    "z_score_delta": delta.z_score_delta,
                    "p_value_delta": delta.p_value_delta,
                    "independent_blocks_delta": delta.independent_blocks_delta,
                    "hits_delta": delta.hits_delta,
                    "anomaly_flags": list(delta.anomaly_flags),
                }
                for sample_id, delta in detail_delta_report.deltas.items()
            },
        },
        "embedding_delta": embedding_delta,
        "anomalies": anomalies,
        "regression_classification": regression_classification,
    }


def write_offline_regression_report(path: str | Path, report: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _build_embedding_delta_report(
    left: WatermarkedArtifact | None,
    right: WatermarkedArtifact | None,
) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "comparable": False,
            "missing_artifact": "left" if left is None else "right",
            "samples": {},
        }

    shared_ids = sorted(set(left.records) & set(right.records))
    samples: dict[str, dict[str, Any]] = {}
    for sample_id in shared_ids:
        left_record = left.records[sample_id]
        right_record = right.records[sample_id]
        if any(field not in left_record or field not in right_record for field in EMBEDDING_FIELDS):
            continue
        route_one_summary = _build_route_one_summary(left_record, right_record)
        delta = EmbeddingDelta(
            sample_id=sample_id,
            total_blocks_delta=int(right_record["total_blocks"]) - int(left_record["total_blocks"]),
            embedded_blocks_delta=int(right_record["embedded_blocks"])
            - int(left_record["embedded_blocks"]),
            failed_blocks_delta=int(right_record["failed_blocks"]) - int(left_record["failed_blocks"]),
            fallback_blocks_delta=int(right_record["fallback_blocks"])
            - int(left_record["fallback_blocks"]),
            embed_rate_delta=float(right_record["embed_rate"]) - float(left_record["embed_rate"]),
            route_one_summary=route_one_summary,
        )
        sample_report = {
            "total_blocks_delta": delta.total_blocks_delta,
            "embedded_blocks_delta": delta.embedded_blocks_delta,
            "failed_blocks_delta": delta.failed_blocks_delta,
            "fallback_blocks_delta": delta.fallback_blocks_delta,
            "embed_rate_delta": delta.embed_rate_delta,
        }
        if delta.route_one_summary is not None:
            sample_report["route_one_summary"] = delta.route_one_summary
        samples[sample_id] = sample_report

    return {
        "comparable": True,
        "missing_artifact": None,
        "samples": samples,
    }


def _build_anomalies(
    detail_delta_report: DetailDeltaReport,
    embedding_delta: dict[str, Any],
) -> dict[str, Any]:
    detail_flags: dict[str, list[str]] = {}
    for sample_id, delta in detail_delta_report.deltas.items():
        if delta.anomaly_flags:
            detail_flags[sample_id] = list(delta.anomaly_flags)

    embedding_flags: dict[str, list[str]] = {}
    for sample_id, sample_delta in embedding_delta.get("samples", {}).items():
        sample_flags: list[str] = []
        if sample_delta["embed_rate_delta"] < 0:
            sample_flags.append("embed_rate_drop")
        if sample_delta["failed_blocks_delta"] > 0:
            sample_flags.append("failed_blocks_increase")
        if sample_flags:
            embedding_flags[sample_id] = sample_flags

    route_one_flags: dict[str, list[str]] = {}
    for sample_id, sample_delta in embedding_delta.get("samples", {}).items():
        route_one_summary = sample_delta.get("route_one_summary") or {}
        right_summary = route_one_summary.get("right") or {}
        retry_summary = right_summary.get("retry_summary") or {}
        cascade_summary = right_summary.get("cascade_summary") or {}
        sample_flags: list[str] = []

        detail_delta = detail_delta_report.deltas.get(sample_id)
        if (
            detail_delta is not None
            and detail_delta.flip_direction == "true_to_false"
            and retry_summary.get("retry_exhausted_blocks", 0) > 0
        ):
            sample_flags.append("near_miss_with_exhausted_retry")

        if (
            cascade_summary.get("cascade_triggers", 0) > 0
            and cascade_summary.get("cascade_rescued_blocks", 0) == 0
            and right_summary.get("unrescued_blocks", 0) > 0
        ):
            sample_flags.append("cascade_no_recovery")

        if sample_flags:
            route_one_flags[sample_id] = sample_flags

    return {
        "detail": detail_flags,
        "embedding": embedding_flags,
        "route_one": route_one_flags,
    }


def _classify_regression(
    parameter_comparison: RunParameterComparison,
    detail_delta_report: DetailDeltaReport,
    compatibility: ArtifactCompatibility,
    left_summary: SummaryArtifact | None,
    right_summary: SummaryArtifact | None,
) -> dict[str, Any]:
    differing_keys = set(parameter_comparison.differing_keys)
    left_rate = _extract_watermark_rate(left_summary)
    right_rate = _extract_watermark_rate(right_summary)
    rate_drop = (
        left_rate is not None and right_rate is not None and right_rate < left_rate
    )
    z_score_drop_count = sum(
        1 for delta in detail_delta_report.deltas.values() if delta.z_score_delta < 0
    )
    extraction_conservatism = bool(detail_delta_report.detection_loss_ids)
    adaptive_gamma_shift = any(
        key.startswith("adaptive_gamma") or key == "lsh_gamma" for key in differing_keys
    )
    parameter_drift = bool(differing_keys)
    calibration_drift = "fpr_threshold" in differing_keys
    implementation_bug = bool(
        extraction_conservatism
        and not parameter_drift
        and z_score_drop_count == detail_delta_report.total_samples
    )

    if not compatibility.is_compatible:
        recommended_branch = "stop"
    elif implementation_bug:
        recommended_branch = "B"
    elif calibration_drift:
        recommended_branch = "C"
    elif parameter_drift:
        recommended_branch = "A"
    elif extraction_conservatism or rate_drop:
        recommended_branch = "C"
    else:
        recommended_branch = "A"

    return {
        "parameter_drift": parameter_drift,
        "adaptive_gamma_shift": adaptive_gamma_shift,
        "extraction_conservatism": extraction_conservatism,
        "calibration_drift": calibration_drift,
        "implementation_bug": implementation_bug,
        "recommended_branch": recommended_branch,
    }


def _extract_watermark_rate(summary: SummaryArtifact | None) -> float | None:
    if summary is None:
        return None
    payload_summary = summary.payload.get("summary") or {}
    watermark_rate = payload_summary.get("watermark_rate")
    if watermark_rate is None:
        return None
    return float(watermark_rate)


def _load_jsonl_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record["id"]
        if sample_id in records:
            raise ValueError(f"duplicate id: {sample_id}")
        records[sample_id] = record
    return records


def _extract_watermark_params(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected_params: dict[str, Any] | None = None
    saw_missing_or_invalid = False
    for record in records.values():
        watermark_params = record.get("watermark_params")
        if watermark_params is None:
            saw_missing_or_invalid = True
            continue
        if not isinstance(watermark_params, dict) or not watermark_params:
            saw_missing_or_invalid = True
            continue
        current_params = dict(watermark_params)
        if selected_params is None:
            selected_params = current_params
            continue
        if current_params != selected_params:
            raise ValueError("inconsistent watermark_params across artifact rows")
    if selected_params is not None and saw_missing_or_invalid:
        raise ValueError("inconsistent watermark_params across artifact rows")
    return selected_params or {}


def _preferred_params(
    summary: SummaryArtifact | None,
    watermarked: WatermarkedArtifact | None,
) -> tuple[str, dict[str, Any]]:
    if watermarked is not None and watermarked.watermark_params:
        return "watermarked", dict(watermarked.watermark_params)
    if summary is not None and summary.watermark_params:
        return "summary", dict(summary.watermark_params)
    return "missing", {}


def _require_boolean_field(record: dict[str, Any], field_name: str) -> bool:
    value = record[field_name]
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _build_route_one_summary(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
) -> dict[str, Any] | None:
    left_summary = _extract_route_one_record_summary(left_record)
    right_summary = _extract_route_one_record_summary(right_record)
    if left_summary is None and right_summary is None:
        return None

    delta: dict[str, Any] = {}
    for field_name in ROUTE_ONE_SCALAR_FIELDS:
        if field_name in (left_summary or {}) or field_name in (right_summary or {}):
            delta[f"{field_name}_delta"] = int((right_summary or {}).get(field_name, 0)) - int(
                (left_summary or {}).get(field_name, 0)
            )

    retry_delta = _build_route_one_count_delta(
        (left_summary or {}).get("retry_summary"),
        (right_summary or {}).get("retry_summary"),
        suffix_keys=True,
    )
    if retry_delta:
        delta["retry_summary"] = retry_delta

    cascade_delta = _build_route_one_count_delta(
        (left_summary or {}).get("cascade_summary"),
        (right_summary or {}).get("cascade_summary"),
        suffix_keys=True,
    )
    if cascade_delta:
        delta["cascade_summary"] = cascade_delta

    failure_reason_delta = _build_route_one_count_delta(
        (left_summary or {}).get("failure_reason_counts"),
        (right_summary or {}).get("failure_reason_counts"),
        suffix_keys=False,
    )
    if failure_reason_delta:
        delta["failure_reason_counts"] = failure_reason_delta

    return {
        "left": left_summary or {},
        "right": right_summary or {},
        "delta": delta,
    }


def _extract_route_one_record_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    summary: dict[str, Any] = {}
    if "diagnostics_version" in record:
        summary["diagnostics_version"] = record["diagnostics_version"]

    for field_name in ROUTE_ONE_COUNT_FIELDS:
        counts = _extract_count_mapping(record.get(field_name))
        if counts is not None:
            summary[field_name] = counts

    for field_name in ROUTE_ONE_SCALAR_FIELDS:
        value = _extract_optional_int(record, field_name)
        if value is not None:
            summary[field_name] = value

    return summary or None


def _extract_count_mapping(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None

    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str):
            continue
        try:
            counts[key] = int(count)
        except (TypeError, ValueError):
            continue
    return counts


def _extract_optional_int(record: dict[str, Any], field_name: str) -> int | None:
    if field_name not in record:
        return None
    try:
        return int(record[field_name])
    except (TypeError, ValueError):
        return None


def _build_route_one_count_delta(
    left_counts: dict[str, int] | None,
    right_counts: dict[str, int] | None,
    *,
    suffix_keys: bool,
) -> dict[str, int]:
    if left_counts is None and right_counts is None:
        return {}

    delta: dict[str, int] = {}
    for key in sorted(set(left_counts or {}) | set(right_counts or {})):
        output_key = f"{key}_delta" if suffix_keys else key
        delta[output_key] = int((right_counts or {}).get(key, 0)) - int(
            (left_counts or {}).get(key, 0)
        )
    return delta
