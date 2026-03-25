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
