from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from wfcllm.detection.calibration import empirical_upper_tail_p
from wfcllm.detection.pipeline import validate_final_code_detector_input_record
from wfcllm.detection.signature_v2 import (
    SUPPORTED_AGGREGATIONS,
    TRIMMED_UNIT_MEAN,
)

V2_DETECTOR_MODE = "wfcllm-aligned-canonical-signature/v2"
V2_METHOD_SCHEMA_VERSION = "wfcllm-method/v2"
V2_CANONICAL_UNIT_SCHEMA_VERSION = "wfcllm-canonical-unit/v2"
V2_CALIBRATION_ARTIFACT_TYPE = "wfcllm_v2_detection_calibration"
V2_CALIBRATION_SCHEMA_VERSION = "wfcllm-detect-calibration/v2"

_FORBIDDEN_KEYS = frozenset(
    {
        "secret_key",
        "raw_secret",
        "audit",
        "candidate_list",
        "generation_score",
        "retry_trace",
    }
)


@dataclass(frozen=True)
class WFCLLMV2DetectionConfig:
    secret_key: str
    signature_bits: int = 16
    min_canonical_units: int = 1
    target_fpr: float = 0.05
    aggregation: str = TRIMMED_UNIT_MEAN
    detector_mode: str = V2_DETECTOR_MODE
    method_schema_version: str = V2_METHOD_SCHEMA_VERSION
    canonical_unit_schema_version: str = V2_CANONICAL_UNIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.secret_key, str) or not self.secret_key:
            raise ValueError("secret_key must be non-empty")
        if (
            isinstance(self.signature_bits, bool)
            or not isinstance(self.signature_bits, int)
            or not 2 <= self.signature_bits <= 128
        ):
            raise ValueError("signature_bits must be an integer in [2, 128]")
        if (
            isinstance(self.min_canonical_units, bool)
            or not isinstance(self.min_canonical_units, int)
            or self.min_canonical_units <= 0
        ):
            raise ValueError("min_canonical_units must be positive")
        if (
            isinstance(self.target_fpr, bool)
            or not isinstance(self.target_fpr, (int, float))
            or not math.isfinite(float(self.target_fpr))
            or not 0 < float(self.target_fpr) < 1
        ):
            raise ValueError("target_fpr must be in (0, 1)")
        if self.aggregation not in SUPPORTED_AGGREGATIONS:
            raise ValueError(f"unsupported V2 aggregation: {self.aggregation!r}")
        if self.detector_mode != V2_DETECTOR_MODE:
            raise ValueError(f"detector_mode must be {V2_DETECTOR_MODE!r}")
        if self.method_schema_version != V2_METHOD_SCHEMA_VERSION:
            raise ValueError(
                f"method_schema_version must be {V2_METHOD_SCHEMA_VERSION!r}"
            )
        if self.canonical_unit_schema_version != V2_CANONICAL_UNIT_SCHEMA_VERSION:
            raise ValueError(
                "canonical_unit_schema_version must be "
                f"{V2_CANONICAL_UNIT_SCHEMA_VERSION!r}"
            )

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("secret_key")
        payload["secret_key_sha256"] = hashlib.sha256(
            self.secret_key.encode("utf-8")
        ).hexdigest()
        return payload


@dataclass(frozen=True)
class V2CalibrationArtifact:
    artifact_type: str
    schema_version: str
    detector_mode: str
    config: dict[str, Any]
    sample_scores: list[float]
    threshold_at_target_fpr: float
    target_fpr: float
    sample_negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WFCLLMV2DetectionResult:
    id: str
    is_watermarked: bool
    score: float
    threshold_5fpr: float
    threshold_at_target_fpr: float
    p_value: float
    fpr_target: float
    unit_count: int
    duplicate_units: int
    total_signature_bits: int
    matched_signature_bits: int
    insufficient_evidence: bool
    detector_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WFCLLMV2DetectionPipeline:
    def __init__(self, *, config: WFCLLMV2DetectionConfig, scorer: Any) -> None:
        self._config = config
        self._scorer = scorer

    def calibrate(self, records: list[dict[str, Any]]) -> V2CalibrationArtifact:
        sample_scores = [self._score_record(record).raw_score for record in records]
        threshold = strict_upper_order_threshold(
            sample_scores,
            self._config.target_fpr,
        )
        return V2CalibrationArtifact(
            artifact_type=V2_CALIBRATION_ARTIFACT_TYPE,
            schema_version=V2_CALIBRATION_SCHEMA_VERSION,
            detector_mode=V2_DETECTOR_MODE,
            config=self._config.to_public_dict(),
            sample_scores=[float(value) for value in sample_scores],
            threshold_at_target_fpr=threshold,
            target_fpr=float(self._config.target_fpr),
            sample_negative_count=len(records),
        )

    def detect_one(
        self,
        record: dict[str, Any],
        *,
        artifact: V2CalibrationArtifact,
    ) -> WFCLLMV2DetectionResult:
        self._validate_artifact_compatible(artifact)
        score = self._score_record(record)
        insufficient = score.unit_count < self._config.min_canonical_units
        is_watermarked = (
            not insufficient
            and score.raw_score >= artifact.threshold_at_target_fpr
        )
        return WFCLLMV2DetectionResult(
            id=str(record["id"]),
            is_watermarked=is_watermarked,
            score=float(score.raw_score),
            threshold_5fpr=artifact.threshold_at_target_fpr,
            threshold_at_target_fpr=artifact.threshold_at_target_fpr,
            p_value=empirical_upper_tail_p(
                score.raw_score,
                artifact.sample_scores,
            ),
            fpr_target=float(self._config.target_fpr),
            unit_count=int(score.unit_count),
            duplicate_units=int(score.duplicate_count),
            total_signature_bits=int(score.total_bits),
            matched_signature_bits=int(score.matched_bits),
            insufficient_evidence=insufficient,
            detector_mode=V2_DETECTOR_MODE,
        )

    def detect_to_jsonl(
        self,
        records: list[dict[str, Any]],
        *,
        artifact: V2CalibrationArtifact,
        output_path: str | Path,
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        self.detect_one(record, artifact=artifact).to_dict(),
                        allow_nan=False,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return str(path)

    def _score_record(self, record: dict[str, Any]) -> Any:
        validate_final_code_detector_input_record(record)
        return self._scorer.score_code(str(record["final_code"]))

    def _validate_artifact_compatible(
        self,
        artifact: V2CalibrationArtifact,
    ) -> None:
        mismatches: list[str] = []
        if artifact.artifact_type != V2_CALIBRATION_ARTIFACT_TYPE:
            mismatches.append("artifact_type")
        if artifact.schema_version != V2_CALIBRATION_SCHEMA_VERSION:
            mismatches.append("schema_version")
        if artifact.detector_mode != V2_DETECTOR_MODE:
            mismatches.append("detector_mode")
        if artifact.config != self._config.to_public_dict():
            mismatches.append("config")
        if float(artifact.target_fpr) != float(self._config.target_fpr):
            mismatches.append("target_fpr")
        if mismatches:
            raise ValueError(
                "v2 calibration artifact is incompatible with detector config: "
                f"{sorted(set(mismatches))}"
            )


def strict_upper_order_threshold(values: list[float], target_fpr: float) -> float:
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    index = math.ceil((len(ordered) + 1) * (1 - target_fpr)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return math.nextafter(ordered[index], math.inf)


def write_v2_calibration_artifact(
    path: str | Path,
    artifact: V2CalibrationArtifact,
) -> str:
    payload = artifact.to_dict()
    _reject_forbidden_keys(payload)
    _validate_v2_payload(payload)
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(artifact_path)


def load_v2_calibration_artifact(path: str | Path) -> V2CalibrationArtifact:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("failed to load v2 calibration artifact") from exc
    try:
        _validate_v2_payload(payload)
        _reject_forbidden_keys(payload)
        return V2CalibrationArtifact(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid v2 calibration artifact") from exc


def _validate_v2_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("v2 calibration artifact must be an object")
    if payload.get("artifact_type") != V2_CALIBRATION_ARTIFACT_TYPE:
        raise ValueError("v2 calibration artifact_type mismatch")
    if payload.get("schema_version") != V2_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("v2 calibration schema_version mismatch")
    allowed = {field.name for field in fields(V2CalibrationArtifact)}
    if set(payload) != allowed:
        raise ValueError("v2 calibration artifact fields mismatch")
    if payload.get("detector_mode") != V2_DETECTOR_MODE:
        raise ValueError("v2 calibration detector_mode mismatch")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("v2 calibration config must be an object")
    if not isinstance(payload.get("sample_scores"), list):
        raise ValueError("v2 calibration sample_scores must be a list")


def _reject_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden key in public v2 artifact at {path}.{key}")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")
