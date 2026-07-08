from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from wfcllm.detection.config import (
    DETECTOR_MODE,
    EVIDENCE_MODES,
    STATISTIC_MODES,
    BucketEdges,
    WFCLLMDetectionConfig,
    bucket_label,
)


ARTIFACT_TYPE = "sawr_detection_calibration"
SCHEMA_VERSION = "sawr-detect-calibration/v1"
FORBIDDEN_ARTIFACT_KEYS = {
    "audit",
    "secret_key",
    "logits",
    "retry_trace",
    "rollback_trace",
    "watermark_params",
}
FORBIDDEN_ARTIFACT_KEY_PREFIXES = ("generation_", "audit_")
PUBLIC_CONFIG_FIELDS = frozenset(
    {
        "lsh_d",
        "gamma",
        "semantic_margin",
        "max_group_statements",
        "min_scoreable_contexts",
        "min_proxy_windows",
        "target_fpr",
        "use_ordinal_keying",
        "evidence_mode",
        "statistic",
        "proxy_penalty_alpha",
        "code_length_adjustment_beta",
        "code_length_reference_chars",
        "structure_aware",
        "bucket_edges",
        "detector_mode",
        "secret_key_sha256",
        "k",
        "gamma_effective",
    }
)
BUCKET_EDGE_FIELDS = frozenset(
    {"window_count", "statement_count", "sample_window_count"}
)


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


ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    field.name for field in fields(CalibrationArtifact)
)


def build_bucket_key(
    context: ContextCalibrationInput,
    *,
    config: WFCLLMDetectionConfig,
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
    config: WFCLLMDetectionConfig,
) -> CalibrationArtifact:
    context_nulls: dict[str, list[float]] = {}
    structure_nulls: dict[str, list[float]] = {}
    global_context_null: list[float] = []

    for sample in samples:
        for context in sample:
            context_raw = _json_float(context.context_raw, "context_raw")
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
    config: WFCLLMDetectionConfig,
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
    _validate_fpr(fpr)
    sorted_values = sorted(_float_list(values, "values"))
    if not sorted_values:
        return 0.0
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
    payload = artifact.to_dict()
    _validated_artifact_values(payload)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
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
    values = _validated_artifact_values(payload)

    return CalibrationArtifact(
        artifact_type=values["artifact_type"],
        schema_version=values["schema_version"],
        detector_mode=values["detector_mode"],
        config=values["config"],
        bucket_edges=values["bucket_edges"],
        context_nulls=values["context_nulls"],
        structure_nulls=values["structure_nulls"],
        global_context_null=values["global_context_null"],
        sample_scores=values["sample_scores"],
        threshold_5fpr=values["threshold_5fpr"],
        context_negative_count=values["context_negative_count"],
        sample_negative_count=values["sample_negative_count"],
    )


def _sample_score(
    sample: list[ContextCalibrationInput],
    *,
    config: WFCLLMDetectionConfig,
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
    config: WFCLLMDetectionConfig,
    context_nulls: dict[str, list[float]],
    structure_nulls: dict[str, list[float]],
    global_context_null: list[float],
) -> tuple[list[float], str]:
    bucket_key = build_bucket_key(context, config=config)
    exact_null = context_nulls.get(bucket_key)
    if exact_null:
        return list(exact_null), "exact"

    if config.structure_aware:
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


def _validated_artifact_values(payload: dict[str, Any]) -> dict[str, Any]:
    _reject_forbidden_keys(payload)
    _validate_top_level_fields(payload)
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"artifact_type must be {ARTIFACT_TYPE!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if payload.get("detector_mode") != DETECTOR_MODE:
        raise ValueError(f"detector_mode must be {DETECTOR_MODE!r}")

    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "detector_mode": DETECTOR_MODE,
        "config": _validate_public_config(payload.get("config")),
        "bucket_edges": _bucket_edges_dict(payload.get("bucket_edges"), "bucket_edges"),
        "context_nulls": _float_list_map(payload.get("context_nulls"), "context_nulls"),
        "structure_nulls": _float_list_map(
            payload.get("structure_nulls"),
            "structure_nulls",
        ),
        "global_context_null": _float_list(
            payload.get("global_context_null"),
            "global_context_null",
        ),
        "sample_scores": _float_list(payload.get("sample_scores"), "sample_scores"),
        "threshold_5fpr": _json_float(
            _required(payload, "threshold_5fpr"),
            "threshold_5fpr",
        ),
        "context_negative_count": _non_negative_int(
            _required(payload, "context_negative_count"),
            "context_negative_count",
        ),
        "sample_negative_count": _non_negative_int(
            _required(payload, "sample_negative_count"),
            "sample_negative_count",
        ),
    }


def _validate_top_level_fields(payload: dict[str, Any]) -> None:
    unknown_fields = set(payload) - ALLOWED_TOP_LEVEL_FIELDS
    if unknown_fields:
        raise ValueError(
            "unknown top-level calibration artifact fields: "
            f"{sorted(unknown_fields)}"
        )


def _validate_public_config(value: Any) -> dict[str, Any]:
    config = _require_dict(value, "config")
    config = _with_legacy_defaults(config)
    missing_fields = PUBLIC_CONFIG_FIELDS - set(config)
    if missing_fields:
        raise ValueError(f"missing public config fields: {sorted(missing_fields)}")
    unknown_fields = set(config) - PUBLIC_CONFIG_FIELDS
    if unknown_fields:
        raise ValueError(f"unknown public config fields: {sorted(unknown_fields)}")

    result = dict(config)
    secret_hash = result["secret_key_sha256"]
    if (
        not isinstance(secret_hash, str)
        or len(secret_hash) != 64
        or any(char not in "0123456789abcdef" for char in secret_hash)
    ):
        raise ValueError("secret_key_sha256 must be 64 lowercase hex chars")

    result["lsh_d"] = _positive_int(result["lsh_d"], "lsh_d")
    result["gamma"] = _bounded_float(result["gamma"], "gamma", lower=0.0, upper=1.0)
    result["k"] = _positive_int(result["k"], "k")
    result["gamma_effective"] = _bounded_float(
        result["gamma_effective"],
        "gamma_effective",
        lower=0.0,
        upper=1.0,
    )
    result["semantic_margin"] = _non_negative_float(
        result["semantic_margin"],
        "semantic_margin",
    )
    result["max_group_statements"] = _positive_int(
        result["max_group_statements"],
        "max_group_statements",
    )
    result["min_scoreable_contexts"] = _positive_int(
        result["min_scoreable_contexts"],
        "min_scoreable_contexts",
    )
    result["min_proxy_windows"] = _positive_int(
        result["min_proxy_windows"],
        "min_proxy_windows",
    )
    result["target_fpr"] = _target_fpr(result["target_fpr"])
    result["use_ordinal_keying"] = _bool_value(
        result["use_ordinal_keying"],
        "use_ordinal_keying",
    )
    result["structure_aware"] = _bool_value(
        result["structure_aware"],
        "structure_aware",
    )
    result["evidence_mode"] = _enum_value(
        result["evidence_mode"],
        "evidence_mode",
        EVIDENCE_MODES,
    )
    result["statistic"] = _enum_value(
        result["statistic"],
        "statistic",
        STATISTIC_MODES,
    )
    result["proxy_penalty_alpha"] = _non_negative_float(
        result["proxy_penalty_alpha"],
        "proxy_penalty_alpha",
    )
    result["code_length_adjustment_beta"] = _json_float(
        result["code_length_adjustment_beta"],
        "code_length_adjustment_beta",
    )
    result["code_length_reference_chars"] = _positive_int(
        result["code_length_reference_chars"],
        "code_length_reference_chars",
    )
    if result["detector_mode"] != DETECTOR_MODE:
        raise ValueError(f"detector_mode must be {DETECTOR_MODE!r}")
    result["bucket_edges"] = _bucket_edges_dict(
        result["bucket_edges"],
        "config.bucket_edges",
    )
    return result


def _with_legacy_defaults(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.setdefault("code_length_adjustment_beta", 0.0)
    result.setdefault("code_length_reference_chars", 700)
    return result


def _reject_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if _is_forbidden_artifact_key(key_text):
                raise ValueError(
                    f"forbidden calibration artifact key {key_text!r} at {path}"
                )
            _reject_forbidden_keys(child, f"{path}.{key_text}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _is_forbidden_artifact_key(key: str) -> bool:
    return key in FORBIDDEN_ARTIFACT_KEYS or key.startswith(
        FORBIDDEN_ARTIFACT_KEY_PREFIXES
    )


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return payload[key]


def _validate_fpr(fpr: Any) -> None:
    if not _is_json_number(fpr) or not 0 < fpr < 1:
        raise ValueError("fpr must be a finite JSON number in (0, 1)")


def _is_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_float(value: Any, name: str) -> float:
    if not _is_json_number(value):
        raise ValueError(f"{name} must contain finite JSON-native numeric values")
    return float(value)


def _bounded_float(value: Any, name: str, *, lower: float, upper: float) -> float:
    numeric_value = _json_float(value, name)
    if not lower <= numeric_value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return numeric_value


def _non_negative_float(value: Any, name: str) -> float:
    numeric_value = _json_float(value, name)
    if numeric_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric_value


def _target_fpr(value: Any) -> float:
    numeric_value = _json_float(value, "target_fpr")
    if not 0 < numeric_value < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    return numeric_value


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _enum_value(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return value


def _float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_json_float(item, name) for item in value]


def _float_list_map(value: Any, name: str) -> dict[str, list[float]]:
    payload = _require_dict(value, name)
    return {
        str(key): _float_list(items, f"{name}[{key!r}]")
        for key, items in payload.items()
    }


def _bucket_edges_dict(value: Any, name: str) -> dict[str, list[int]]:
    payload = _require_dict(value, name)
    missing_fields = BUCKET_EDGE_FIELDS - set(payload)
    if missing_fields:
        raise ValueError(f"{name} missing bucket edge fields: {sorted(missing_fields)}")
    unknown_fields = set(payload) - BUCKET_EDGE_FIELDS
    if unknown_fields:
        raise ValueError(
            f"{name} has unknown bucket edge fields: {sorted(unknown_fields)}"
        )
    for field_name in BUCKET_EDGE_FIELDS:
        if not isinstance(payload[field_name], list):
            raise ValueError(f"{name}.{field_name} must be a list")
    try:
        edges = BucketEdges(
            window_count=tuple(payload["window_count"]),
            statement_count=tuple(payload["statement_count"]),
            sample_window_count=tuple(payload["sample_window_count"]),
        )
    except ValueError as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc
    return edges.to_dict()
