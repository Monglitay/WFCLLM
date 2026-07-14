from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.batch_invariant_v4.calibration import (
    CalibrationArtifact,
    fit_calibration,
    load_calibration,
    save_calibration,
)


def test_calibration_uses_only_calibration_role_and_conservative_ties() -> None:
    artifact = fit_calibration(
        scores=(0.1, 0.2, 0.2, 0.4),
        source_role="calibration_negative",
        source_sha256="a" * 64,
        target_fpr=0.05,
        minimum_independent_units=3,
    )

    assert artifact.empirical_p_value(0.2) == pytest.approx(4 / 5)
    assert artifact.empirical_p_value(0.4) == pytest.approx(2 / 5)
    with pytest.raises(ValueError, match="calibration_negative"):
        fit_calibration(
            scores=(0.1,),
            source_role="heldout_negative",
            source_sha256="a" * 64,
            target_fpr=0.05,
            minimum_independent_units=3,
        )


def test_minimum_units_and_empirical_decision_are_frozen() -> None:
    artifact = CalibrationArtifact.from_scores_for_test(
        tuple(float(index) for index in range(100)),
        target_fpr=0.05,
        minimum_independent_units=3,
    )

    assert artifact.decide(score=100.0, independent_units=2) is False
    assert artifact.decide(score=100.0, independent_units=3) is True
    assert artifact.empirical_p_value(100.0) == pytest.approx(1 / 101)


def test_calibration_roundtrip_and_malformed_artifact_fail_fast(tmp_path: Path) -> None:
    artifact = fit_calibration(
        scores=(0.1, 0.2, 0.3),
        source_role="calibration_negative",
        source_sha256="b" * 64,
        target_fpr=0.05,
        minimum_independent_units=3,
    )
    path = tmp_path / "calibration.json"
    save_calibration(artifact, path)
    assert load_calibration(path) == artifact

    payload = json.loads(path.read_text())
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_calibration(path)
