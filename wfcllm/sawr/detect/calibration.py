from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wfcllm.sawr.detect.config import (
    DETECTOR_MODE,
    SawrDetectionConfig,
    bucket_label,
)


ARTIFACT_TYPE = "sawr_detection_calibration"
SCHEMA_VERSION = "sawr-detect-calibration/v1"


@dataclass(frozen=True)
class ContextCalibrationInput:
    context_id: str
    structure_type: str
    parent_node_type: str
    context_raw: float
    context_window_count: int
    context_statement_count: int
    window_length_mix: str
    sample_proxy_windows: int


@dataclass(frozen=True)
class CalibrationArtifact:
    artifact_type: str
    schema_version: str
    detector_mode: str
    config: dict[str, Any]
    bucket_edges: dict[str, list[int]]
    context_nulls: dict[str, list[float]]
    structure_nulls: dict[str, list[float]]
    global_context_null: list[float]
    sample_scores: list[float]
    threshold_5fpr: float
    context_negative_count: int
    sample_negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_bucket_key(
    context: ContextCalibrationInput,
    *,
    config: SawrDetectionConfig,
) -> str:
    payload = {
        "structure_type": (
            context.structure_type if config.structure_aware else "any_structure"
        ),
        "parent_node_type": (
            context.parent_node_type if config.structure_aware else "any_parent"
        ),
        "window_count_bucket": bucket_label(
            context.context_window_count,
            config.bucket_edges.window_count,
        ),
        "statement_count_bucket": bucket_label(
            context.context_statement_count,
            config.bucket_edges.statement_count,
        ),
        "sample_window_count_bucket": bucket_label(
            context.sample_proxy_windows,
            config.bucket_edges.sample_window_count,
        ),
        "window_length_mix": context.window_length_mix,
        "use_ordinal_keying": config.use_ordinal_keying,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_calibration_artifact(
    samples: list[list[ContextCalibrationInput]],
    *,
    config: SawrDetectionConfig,
) -> CalibrationArtifact:
    context_nulls: dict[str, list[float]] = {}
    structure_nulls: dict[str, list[float]] = {}
    global_context_null: list[float] = []

    for sample in samples:
        for context in sample:
            context_raw = float(context.context_raw)
            bucket_key = build_bucket_key(context, config=config)
            context_nulls.setdefault(bucket_key, []).append(context_raw)
            structure_nulls.setdefault(context.structure_type, []).append(context_raw)
            global_context_null.append(context_raw)

    sorted_context_nulls = _sorted_null_map(context_nulls)
    sorted_structure_nulls = _sorted_null_map(structure_nulls)
    sorted_global_context_null = sorted(global_context_null)

    sample_scores = [
        _sample_score(
            sample,
            config=config,
            context_nulls=sorted_context_nulls,
            structure_nulls=sorted_structure_nulls,
            global_context_null=sorted_global_context_null,
        )
        for sample in samples
    ]
    threshold_5fpr = (
        percentile_threshold(sample_scores, config.target_fpr) if sample_scores else 0.0
    )

    return CalibrationArtifact(
        artifact_type=ARTIFACT_TYPE,
        schema_version=SCHEMA_VERSION,
        detector_mode=DETECTOR_MODE,
        config=config.to_public_dict(),
        bucket_edges=config.bucket_edges.to_dict(),
        context_nulls=sorted_context_nulls,
        structure_nulls=sorted_structure_nulls,
        global_context_null=sorted_global_context_null,
        sample_scores=sample_scores,
        threshold_5fpr=threshold_5fpr,
        context_negative_count=len(sorted_global_context_null),
        sample_negative_count=len(samples),
    )


def null_for_context_from_artifact(
    context: ContextCalibrationInput,
    *,
    config: SawrDetectionConfig,
    artifact: CalibrationArtifact,
) -> tuple[list[float], str]:
    return _null_for_context(
        context,
        config=config,
        context_nulls=artifact.context_nulls,
        structure_nulls=artifact.structure_nulls,
        global_context_null=artifact.global_context_null,
    )


def empirical_upper_tail_p(observed: float, null_values: list[float]) -> float:
    if not null_values:
        return 1.0
    upper_tail_count = sum(1 for value in null_values if value >= observed)
    return (1 + upper_tail_count) / (1 + len(null_values))


def context_score_from_null(observed: float, null_values: list[float]) -> float:
    p_value = empirical_upper_tail_p(observed, null_values)
    p_floor = 1 / (len(null_values) + 1) if null_values else 1.0
    return -math.log10(max(p_value, p_floor))


def percentile_threshold(values: list[float], fpr: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    percentile_index = (len(sorted_values) - 1) * (1 - fpr)
    lower_index = math.floor(percentile_index)
    upper_index = math.ceil(percentile_index)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = percentile_index - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def write_calibration_artifact(
    path: str | Path,
    artifact: CalibrationArtifact,
) -> str:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            artifact.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(artifact_path)


def load_calibration_artifact(path: str | Path) -> CalibrationArtifact:
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid calibration artifact JSON: {artifact_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("calibration artifact must be a JSON object")
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"artifact_type must be {ARTIFACT_TYPE!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if payload.get("detector_mode") != DETECTOR_MODE:
        raise ValueError(f"detector_mode must be {DETECTOR_MODE!r}")

    config = _require_dict(payload.get("config"), "config")
    if "secret_key" in config:
        raise ValueError("calibration artifact config must not contain secret_key")

    return CalibrationArtifact(
        artifact_type=ARTIFACT_TYPE,
        schema_version=SCHEMA_VERSION,
        detector_mode=DETECTOR_MODE,
        config=dict(config),
        bucket_edges=_int_list_map(payload.get("bucket_edges"), "bucket_edges"),
        context_nulls=_float_list_map(payload.get("context_nulls"), "context_nulls"),
        structure_nulls=_float_list_map(
            payload.get("structure_nulls"),
            "structure_nulls",
        ),
        global_context_null=_float_list(
            payload.get("global_context_null"),
            "global_context_null",
        ),
        sample_scores=_float_list(payload.get("sample_scores"), "sample_scores"),
        threshold_5fpr=float(_required(payload, "threshold_5fpr")),
        context_negative_count=int(_required(payload, "context_negative_count")),
        sample_negative_count=int(_required(payload, "sample_negative_count")),
    )


def _sample_score(
    sample: list[ContextCalibrationInput],
    *,
    config: SawrDetectionConfig,
    context_nulls: dict[str, list[float]],
    structure_nulls: dict[str, list[float]],
    global_context_null: list[float],
) -> float:
    if not sample:
        return 0.0
    scores = []
    for context in sample:
        null_values, _level = _null_for_context(
            context,
            config=config,
            context_nulls=context_nulls,
            structure_nulls=structure_nulls,
            global_context_null=global_context_null,
        )
        scores.append(context_score_from_null(context.context_raw, null_values))
    return sum(scores) / len(scores)


def _null_for_context(
    context: ContextCalibrationInput,
    *,
    config: SawrDetectionConfig,
    context_nulls: dict[str, list[float]],
    structure_nulls: dict[str, list[float]],
    global_context_null: list[float],
) -> tuple[list[float], str]:
    bucket_key = build_bucket_key(context, config=config)
    exact_null = context_nulls.get(bucket_key)
    if exact_null:
        return list(exact_null), "exact"

    structure_null = structure_nulls.get(context.structure_type)
    if structure_null:
        return list(structure_null), "structure_type"

    return list(global_context_null), "global"


def _sorted_null_map(nulls: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: sorted(values) for key, values in sorted(nulls.items())}


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return payload[key]


def _float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


def _float_list_map(value: Any, name: str) -> dict[str, list[float]]:
    payload = _require_dict(value, name)
    return {
        str(key): _float_list(items, f"{name}[{key!r}]")
        for key, items in payload.items()
    }


def _int_list_map(value: Any, name: str) -> dict[str, list[int]]:
    payload = _require_dict(value, name)
    try:
        return {
            str(key): [int(item) for item in items]
            for key, items in payload.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer lists") from exc
