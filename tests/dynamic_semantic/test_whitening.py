from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wfcllm.dynamic_semantic.whitening import (
    WHITENING_SCHEMA_VERSION,
    WhiteningArtifact,
    fit_whitening,
    load_whitening,
    save_whitening,
)


def _embeddings() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_fit_whitening_requires_calibration_negatives() -> None:
    for role in ("positive", "heldout_negative", "pilot"):
        with pytest.raises(ValueError, match="calibration_negative"):
            fit_whitening(
                _embeddings(),
                output_dimensions=2,
                source_role=role,
                source_manifest_sha256="a" * 64,
            )


def test_fit_and_transform_are_deterministic_and_l2_normalized() -> None:
    first = fit_whitening(
        _embeddings(),
        output_dimensions=2,
        source_role="calibration_negative",
        source_manifest_sha256="b" * 64,
    )
    second = fit_whitening(
        _embeddings(),
        output_dimensions=2,
        source_role="calibration_negative",
        source_manifest_sha256="b" * 64,
    )

    assert first == second
    transformed = first.transform(_embeddings())
    assert transformed.shape == (5, 2)
    assert transformed.dtype == torch.float32
    assert torch.linalg.vector_norm(transformed, dim=1).tolist() == pytest.approx(
        [1.0] * 5,
        abs=1e-6,
    )


def test_whitening_artifact_round_trip_and_schema_validation(tmp_path: Path) -> None:
    artifact = fit_whitening(
        _embeddings(),
        output_dimensions=2,
        source_role="calibration_negative",
        source_manifest_sha256="c" * 64,
    )
    path = tmp_path / "whitening.json"

    save_whitening(artifact, path)
    loaded = load_whitening(path)

    assert loaded == artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == WHITENING_SCHEMA_VERSION
    assert payload["source_role"] == "calibration_negative"
    assert payload["source_manifest_sha256"] == "c" * 64

    payload["schema_version"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_whitening(path)


def test_whitening_rejects_invalid_dimensions_or_nonfinite_data() -> None:
    with pytest.raises(ValueError, match="output_dimensions"):
        fit_whitening(
            _embeddings(),
            output_dimensions=4,
            source_role="calibration_negative",
            source_manifest_sha256="d" * 64,
        )

    invalid = _embeddings()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_whitening(
            invalid,
            output_dimensions=2,
            source_role="calibration_negative",
            source_manifest_sha256="d" * 64,
        )


def test_artifact_transform_rejects_wrong_input_dimensions() -> None:
    artifact = WhiteningArtifact(
        schema_version=WHITENING_SCHEMA_VERSION,
        source_role="calibration_negative",
        source_manifest_sha256="e" * 64,
        input_dimensions=3,
        output_dimensions=2,
        epsilon=1e-6,
        mean=(0.0, 0.0, 0.0),
        projection=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    with pytest.raises(ValueError, match="input dimensions"):
        artifact.transform(torch.ones((2, 4)))
