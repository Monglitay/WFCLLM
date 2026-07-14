from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CALIBRATION_SCHEMA_VERSION = "wfcllm-dynamic-semantic-calibration/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ConformalCalibration:
    schema_version: str
    source_role: str
    source_manifest_sha256: str
    target_fpr: float
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CALIBRATION_SCHEMA_VERSION}"
            )
        if self.source_role != "calibration_negative":
            raise ValueError("source_role must be calibration_negative")
        if not _SHA256_PATTERN.fullmatch(self.source_manifest_sha256):
            raise ValueError("source_manifest_sha256 must be lowercase SHA-256")
        if not math.isfinite(self.target_fpr) or not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be finite and in (0, 1)")
        if not self.scores:
            raise ValueError("calibration scores must not be empty")
        if not all(math.isfinite(score) for score in self.scores):
            raise ValueError("calibration scores must be finite")

    @property
    def count(self) -> int:
        return len(self.scores)

    def p_value(self, observed_score: float) -> float:
        if not math.isfinite(observed_score):
            raise ValueError("observed_score must be finite")
        exceedances = sum(score >= observed_score for score in self.scores)
        return (1 + exceedances) / (self.count + 1)

    def decide(self, observed_score: float, *, eligible: bool) -> bool:
        return bool(eligible and self.p_value(observed_score) <= self.target_fpr)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_role": self.source_role,
            "source_manifest_sha256": self.source_manifest_sha256,
            "target_fpr": self.target_fpr,
            "count": self.count,
            "scores": list(self.scores),
            "tie_policy": "count calibration scores greater than or equal",
        }


def fit_calibration(
    scores: Iterable[float],
    *,
    source_role: str,
    source_manifest_sha256: str,
    target_fpr: float,
) -> ConformalCalibration:
    if source_role != "calibration_negative":
        raise ValueError("source_role must be calibration_negative")
    normalized = tuple(sorted(float(score) for score in scores))
    return ConformalCalibration(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        source_role=source_role,
        source_manifest_sha256=source_manifest_sha256,
        target_fpr=float(target_fpr),
        scores=normalized,
    )


def save_calibration(
    calibration: ConformalCalibration,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        calibration.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    output_path.write_text(payload + "\n", encoding="utf-8", newline="\n")


def load_calibration(path: str | Path) -> ConformalCalibration:
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load calibration: {input_path}") from exc
    required = {
        "schema_version",
        "source_role",
        "source_manifest_sha256",
        "target_fpr",
        "count",
        "scores",
        "tie_policy",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("calibration artifact fields do not match schema")
    calibration = ConformalCalibration(
        schema_version=str(payload["schema_version"]),
        source_role=str(payload["source_role"]),
        source_manifest_sha256=str(payload["source_manifest_sha256"]),
        target_fpr=float(payload["target_fpr"]),
        scores=tuple(float(score) for score in payload["scores"]),
    )
    if int(payload["count"]) != calibration.count:
        raise ValueError("calibration count does not match scores")
    if payload["tie_policy"] != "count calibration scores greater than or equal":
        raise ValueError("calibration tie_policy mismatch")
    return calibration
