from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wfcllm.sawr.detect.calibration import (
    CalibrationArtifact,
    ContextCalibrationInput,
    build_bucket_key,
    build_calibration_artifact,
    context_score_from_null,
    empirical_upper_tail_p,
    load_calibration_artifact,
    percentile_threshold,
    write_calibration_artifact,
)
from wfcllm.sawr.detect.config import SawrDetectionConfig


def _context(raw: float, *, windows: int = 3, statements: int = 2) -> ContextCalibrationInput:
    return ContextCalibrationInput(
        context_id="ctx",
        structure_type="function_body",
        parent_node_type="function_definition",
        context_raw=raw,
        context_window_count=windows,
        context_statement_count=statements,
        window_length_mix="1:2|2:1",
        sample_proxy_windows=5,
    )


def test_empirical_upper_tail_p_uses_plus_one_smoothing() -> None:
    assert empirical_upper_tail_p(0.5, [0.1, 0.5, 0.9]) == pytest.approx(3 / 4)
    assert empirical_upper_tail_p(1.0, [0.1, 0.5, 0.9]) == pytest.approx(1 / 4)


def test_context_score_from_null_is_negative_log10_p() -> None:
    score = context_score_from_null(1.0, [0.1, 0.5, 0.9])

    assert score == pytest.approx(-math.log10(1 / 4))


def test_percentile_threshold_uses_linear_interpolation() -> None:
    assert percentile_threshold([0, 1, 2, 3], fpr=0.25) == pytest.approx(2.25)


def test_build_bucket_key_includes_frozen_conditioning_variables() -> None:
    config = SawrDetectionConfig(secret_key="1010", use_ordinal_keying=True)

    key = build_bucket_key(_context(0.4), config=config)

    assert json.loads(key) == {
        "structure_type": "function_body",
        "parent_node_type": "function_definition",
        "window_count_bucket": "2-3",
        "statement_count_bucket": "2-3",
        "sample_window_count_bucket": "2-5",
        "window_length_mix": "1:2|2:1",
        "use_ordinal_keying": True,
    }


def test_build_calibration_artifact_is_secret_free_and_scores_samples() -> None:
    config = SawrDetectionConfig(secret_key="1010", target_fpr=0.05)
    samples = [
        [_context(0.1), _context(0.2)],
        [_context(0.4), _context(0.5)],
        [_context(0.9)],
    ]

    artifact = build_calibration_artifact(samples, config=config)

    assert artifact.artifact_type == "sawr_detection_calibration"
    assert artifact.schema_version == "sawr-detect-calibration/v1"
    assert artifact.config["secret_key_sha256"]
    assert "secret_key" not in artifact.config
    assert artifact.context_negative_count == 5
    assert artifact.sample_negative_count == 3
    assert len(artifact.sample_scores) == 3
    assert artifact.threshold_5fpr == pytest.approx(
        percentile_threshold(artifact.sample_scores, fpr=0.05)
    )


def test_calibration_artifact_roundtrip(tmp_path: Path) -> None:
    config = SawrDetectionConfig(secret_key="1010")
    artifact = build_calibration_artifact([[_context(0.1)]], config=config)
    path = tmp_path / "calibration.json"

    write_calibration_artifact(path, artifact)
    loaded = load_calibration_artifact(path)

    assert isinstance(loaded, CalibrationArtifact)
    assert loaded.to_dict() == artifact.to_dict()
