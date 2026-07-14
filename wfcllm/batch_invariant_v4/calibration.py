from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CALIBRATION_SCHEMA_VERSION = "wfcllm-v4-calibration/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: str
    source_role: str
    source_sha256: str
    scores: tuple[float, ...]
    target_fpr: float
    minimum_independent_units: int
    conservative_ties: str = "greater-than-or-equal"
    secret_metadata_included: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("calibration schema version mismatch")
        if self.source_role != "calibration_negative":
            raise ValueError("source_role must be calibration_negative")
        if not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be lowercase SHA-256")
        if not self.scores or not all(math.isfinite(value) for value in self.scores):
            raise ValueError("calibration scores must be finite and non-empty")
        if not 0 < self.target_fpr <= 1:
            raise ValueError("target_fpr must be in (0, 1]")
        if self.minimum_independent_units <= 0:
            raise ValueError("minimum_independent_units must be positive")
        if self.conservative_ties != "greater-than-or-equal":
            raise ValueError("calibration ties must be conservative")
        if self.secret_metadata_included is not False:
            raise ValueError("calibration cannot contain secret metadata")

    @classmethod
    def from_scores_for_test(
        cls,
        scores: Iterable[float],
        *,
        target_fpr: float,
        minimum_independent_units: int,
    ) -> CalibrationArtifact:
        return cls(
            schema_version=CALIBRATION_SCHEMA_VERSION,
            source_role="calibration_negative",
            source_sha256="0" * 64,
            scores=tuple(float(value) for value in scores),
            target_fpr=target_fpr,
            minimum_independent_units=minimum_independent_units,
        )

    def empirical_p_value(self, score: float) -> float:
        if not math.isfinite(score):
            raise ValueError("observed score must be finite")
        return (1 + sum(value >= score for value in self.scores)) / (
            len(self.scores) + 1
        )

    def decide(self, *, score: float, independent_units: int) -> bool:
        return (
            independent_units >= self.minimum_independent_units
            and self.empirical_p_value(score) <= self.target_fpr
        )


def fit_calibration(
    *,
    scores: Iterable[float],
    source_role: str,
    source_sha256: str,
    target_fpr: float,
    minimum_independent_units: int,
) -> CalibrationArtifact:
    return CalibrationArtifact(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        source_role=source_role,
        source_sha256=source_sha256,
        scores=tuple(float(value) for value in scores),
        target_fpr=float(target_fpr),
        minimum_independent_units=minimum_independent_units,
    )


def save_calibration(artifact: CalibrationArtifact, path: str | Path) -> None:
    if not isinstance(artifact, CalibrationArtifact):
        raise ValueError("artifact must be CalibrationArtifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(artifact), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> CalibrationArtifact:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load calibration: {path}") from exc
    expected = {
        "schema_version",
        "source_role",
        "source_sha256",
        "scores",
        "target_fpr",
        "minimum_independent_units",
        "conservative_ties",
        "secret_metadata_included",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("calibration artifact fields do not match schema")
    try:
        return CalibrationArtifact(
            schema_version=payload["schema_version"],
            source_role=payload["source_role"],
            source_sha256=payload["source_sha256"],
            scores=tuple(float(value) for value in payload["scores"]),
            target_fpr=float(payload["target_fpr"]),
            minimum_independent_units=int(payload["minimum_independent_units"]),
            conservative_ties=payload["conservative_ties"],
            secret_metadata_included=payload["secret_metadata_included"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid calibration artifact") from exc
