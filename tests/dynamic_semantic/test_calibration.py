from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.dynamic_semantic.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    fit_calibration,
    load_calibration,
    save_calibration,
)


def test_conformal_p_value_counts_ties_conservatively() -> None:
    calibration = fit_calibration(
        [0.1, 0.2, 0.2, 0.4],
        source_role="calibration_negative",
        source_manifest_sha256="a" * 64,
        target_fpr=0.05,
    )

    assert calibration.p_value(0.2) == pytest.approx(4 / 5)
    assert calibration.p_value(0.3) == pytest.approx(2 / 5)
    assert calibration.p_value(0.5) == pytest.approx(1 / 5)


def test_calibration_requires_exact_role_and_finite_scores() -> None:
    with pytest.raises(ValueError, match="calibration_negative"):
        fit_calibration(
            [0.0],
            source_role="heldout_negative",
            source_manifest_sha256="b" * 64,
            target_fpr=0.05,
        )
    with pytest.raises(ValueError, match="finite"):
        fit_calibration(
            [float("nan")],
            source_role="calibration_negative",
            source_manifest_sha256="b" * 64,
            target_fpr=0.05,
        )


def test_calibration_round_trip_preserves_all_scores(tmp_path: Path) -> None:
    calibration = fit_calibration(
        [index / 250 for index in range(250)],
        source_role="calibration_negative",
        source_manifest_sha256="c" * 64,
        target_fpr=0.05,
    )
    path = tmp_path / "calibration.json"

    save_calibration(calibration, path)
    loaded = load_calibration(path)

    assert loaded == calibration
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert payload["count"] == 250


def test_detection_requires_p_at_most_target_and_eligible() -> None:
    calibration = fit_calibration(
        [0.0] * 250,
        source_role="calibration_negative",
        source_manifest_sha256="d" * 64,
        target_fpr=0.05,
    )

    assert calibration.decide(1.0, eligible=True) is True
    assert calibration.decide(1.0, eligible=False) is False
    assert calibration.decide(0.0, eligible=True) is False
