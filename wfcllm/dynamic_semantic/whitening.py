from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


WHITENING_SCHEMA_VERSION = "wfcllm-dynamic-semantic-whitening/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class WhiteningArtifact:
    schema_version: str
    source_role: str
    source_manifest_sha256: str
    input_dimensions: int
    output_dimensions: int
    epsilon: float
    mean: tuple[float, ...]
    projection: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.schema_version != WHITENING_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {WHITENING_SCHEMA_VERSION}"
            )
        if self.source_role != "calibration_negative":
            raise ValueError("source_role must be calibration_negative")
        if not _SHA256_PATTERN.fullmatch(self.source_manifest_sha256):
            raise ValueError("source_manifest_sha256 must be lowercase SHA-256")
        if self.input_dimensions <= 0:
            raise ValueError("input_dimensions must be positive")
        if not 0 < self.output_dimensions <= self.input_dimensions:
            raise ValueError("output_dimensions must be in [1, input_dimensions]")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        if len(self.mean) != self.input_dimensions:
            raise ValueError("mean dimensions do not match input_dimensions")
        if len(self.projection) != self.output_dimensions or any(
            len(row) != self.input_dimensions for row in self.projection
        ):
            raise ValueError("projection dimensions are incompatible")
        numeric_values = [*self.mean, *(value for row in self.projection for value in row)]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("whitening artifact values must be finite")

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        values = embeddings.detach().to(dtype=torch.float32, device="cpu")
        if values.ndim != 2 or values.shape[1] != self.input_dimensions:
            raise ValueError("embedding input dimensions do not match whitening")
        if not torch.isfinite(values).all():
            raise ValueError("embedding values must be finite")
        mean = torch.tensor(self.mean, dtype=torch.float32)
        projection = torch.tensor(self.projection, dtype=torch.float32)
        transformed = (values - mean) @ projection.T
        return F.normalize(transformed, p=2, dim=1).to(torch.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonicalize_component_signs(components: torch.Tensor) -> torch.Tensor:
    normalized = components.clone()
    for index in range(normalized.shape[0]):
        row = normalized[index]
        pivot = int(torch.argmax(torch.abs(row)).item())
        if row[pivot] < 0:
            normalized[index] = -row
    return normalized


def fit_whitening(
    embeddings: torch.Tensor,
    *,
    output_dimensions: int,
    source_role: str,
    source_manifest_sha256: str,
    epsilon: float = 1e-6,
) -> WhiteningArtifact:
    if source_role != "calibration_negative":
        raise ValueError("source_role must be calibration_negative")
    values = embeddings.detach().to(dtype=torch.float64, device="cpu")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] <= 0:
        raise ValueError("embeddings must be a rank-2 tensor with at least two rows")
    if not torch.isfinite(values).all():
        raise ValueError("embedding values must be finite")
    input_dimensions = int(values.shape[1])
    if (
        isinstance(output_dimensions, bool)
        or not isinstance(output_dimensions, int)
        or not 0 < output_dimensions <= input_dimensions
    ):
        raise ValueError("output_dimensions must be in [1, input dimensions]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:output_dimensions]
    selected_values = torch.clamp(eigenvalues[order], min=epsilon)
    components = eigenvectors[:, order].T
    components = _canonicalize_component_signs(components)
    projection = components / torch.sqrt(selected_values).unsqueeze(1)
    return WhiteningArtifact(
        schema_version=WHITENING_SCHEMA_VERSION,
        source_role=source_role,
        source_manifest_sha256=source_manifest_sha256,
        input_dimensions=input_dimensions,
        output_dimensions=output_dimensions,
        epsilon=float(epsilon),
        mean=tuple(float(value) for value in mean.tolist()),
        projection=tuple(
            tuple(float(value) for value in row)
            for row in projection.tolist()
        ),
    )


def save_whitening(artifact: WhiteningArtifact, path: str | Path) -> None:
    if not isinstance(artifact, WhiteningArtifact):
        raise ValueError("artifact must be WhiteningArtifact")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    output_path.write_text(payload + "\n", encoding="utf-8", newline="\n")


def load_whitening(path: str | Path) -> WhiteningArtifact:
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load whitening artifact: {artifact_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("whitening artifact must be an object")
    expected = {
        "schema_version",
        "source_role",
        "source_manifest_sha256",
        "input_dimensions",
        "output_dimensions",
        "epsilon",
        "mean",
        "projection",
    }
    if set(payload) != expected:
        raise ValueError("whitening artifact fields do not match schema")
    try:
        return WhiteningArtifact(
            schema_version=str(payload["schema_version"]),
            source_role=str(payload["source_role"]),
            source_manifest_sha256=str(payload["source_manifest_sha256"]),
            input_dimensions=int(payload["input_dimensions"]),
            output_dimensions=int(payload["output_dimensions"]),
            epsilon=float(payload["epsilon"]),
            mean=tuple(float(value) for value in payload["mean"]),
            projection=tuple(
                tuple(float(value) for value in row)
                for row in payload["projection"]
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "schema_version" in str(exc):
            raise
        raise ValueError("invalid whitening artifact") from exc
