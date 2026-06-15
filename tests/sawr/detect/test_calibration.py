from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import wfcllm.sawr.detect as detect
import wfcllm.sawr.detect.calibration as calibration
from wfcllm.sawr.detect.calibration import (
    CalibrationArtifact,
    ContextCalibrationInput,
    build_bucket_key,
    build_calibration_artifact,
    context_score_from_null,
    empirical_upper_tail_p,
    load_calibration_artifact,
    null_for_context_from_artifact,
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


def _artifact(
    *,
    config: dict[str, object] | None = None,
    bucket_edges: dict[str, list[int]] | None = None,
    context_nulls: dict[str, list[float]] | None = None,
    structure_nulls: dict[str, list[float]] | None = None,
    global_context_null: list[float] | None = None,
    sample_scores: list[float] | None = None,
    threshold_5fpr: float = 0.0,
    context_negative_count: int = 0,
    sample_negative_count: int = 0,
) -> CalibrationArtifact:
    public_config = SawrDetectionConfig(secret_key="1010").to_public_dict()
    public_config.update(config or {})
    return CalibrationArtifact(
        artifact_type="sawr_detection_calibration",
        schema_version="sawr-detect-calibration/v1",
        detector_mode="sawr-structure-aware-proxy-window/v1",
        config=public_config,
        bucket_edges=bucket_edges
        or {
            "window_count": [1, 2, 4, 8, 16],
            "statement_count": [1, 2, 4, 8, 16],
            "sample_window_count": [2, 6, 16, 32, 64],
        },
        context_nulls=context_nulls or {},
        structure_nulls=structure_nulls or {},
        global_context_null=global_context_null or [],
        sample_scores=sample_scores or [],
        threshold_5fpr=threshold_5fpr,
        context_negative_count=context_negative_count,
        sample_negative_count=sample_negative_count,
    )


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    return path


def test_empirical_upper_tail_p_uses_plus_one_smoothing() -> None:
    assert empirical_upper_tail_p(0.5, [0.1, 0.5, 0.9]) == pytest.approx(3 / 4)
    assert empirical_upper_tail_p(1.0, [0.1, 0.5, 0.9]) == pytest.approx(1 / 4)


def test_context_score_from_null_is_negative_log10_p() -> None:
    score = context_score_from_null(1.0, [0.1, 0.5, 0.9])

    assert score == pytest.approx(-math.log10(1 / 4))


def test_percentile_threshold_uses_linear_interpolation() -> None:
    assert percentile_threshold([0, 1, 2, 3], fpr=0.25) == pytest.approx(2.25)


@pytest.mark.parametrize("fpr", [0.0, 1.0, -0.1, 1.1, True, "0.05"])
def test_percentile_threshold_rejects_invalid_fpr(fpr: object) -> None:
    with pytest.raises(ValueError, match="fpr"):
        percentile_threshold([0.1, 0.2], fpr=fpr)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, "0.2", float("nan"), float("inf")])
def test_percentile_threshold_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="values"):
        percentile_threshold([0.1, value], fpr=0.05)  # type: ignore[list-item]


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


def test_empty_calibration_samples_produce_empty_nulls_and_zero_threshold() -> None:
    config = SawrDetectionConfig(secret_key="1010")

    artifact = build_calibration_artifact([], config=config)

    assert artifact.context_nulls == {}
    assert artifact.structure_nulls == {}
    assert artifact.global_context_null == []
    assert artifact.sample_scores == []
    assert artifact.threshold_5fpr == 0.0
    assert artifact.context_negative_count == 0
    assert artifact.sample_negative_count == 0


def test_null_for_context_falls_back_exact_then_structure_then_global() -> None:
    config = SawrDetectionConfig(secret_key="1010")
    exact_context = _context(0.0)
    exact_key = build_bucket_key(exact_context, config=config)
    artifact = _artifact(
        context_nulls={exact_key: [0.1]},
        structure_nulls={"function_body": [0.2]},
        global_context_null=[0.3],
    )

    null_values, level = null_for_context_from_artifact(
        exact_context,
        config=config,
        artifact=artifact,
    )
    assert null_values == [0.1]
    assert level == "exact"

    null_values, level = null_for_context_from_artifact(
        _context(0.0, windows=6),
        config=config,
        artifact=artifact,
    )
    assert null_values == [0.2]
    assert level == "structure_type"

    null_values, level = null_for_context_from_artifact(
        ContextCalibrationInput(
            context_id="ctx",
            structure_type="class_body",
            parent_node_type="class_definition",
            context_raw=0.0,
            context_window_count=6,
            context_statement_count=2,
            window_length_mix="1:2|2:1",
            sample_proxy_windows=5,
        ),
        config=config,
        artifact=artifact,
    )
    assert null_values == [0.3]
    assert level == "global"


def test_structure_unaware_unmatched_context_skips_real_structure_fallback() -> None:
    config = SawrDetectionConfig(secret_key="1010", structure_aware=False)
    matched_context = _context(0.0)
    matched_key = build_bucket_key(matched_context, config=config)
    artifact = _artifact(
        context_nulls={matched_key: [0.1]},
        structure_nulls={"function_body": [0.2]},
        global_context_null=[0.3],
    )

    null_values, level = null_for_context_from_artifact(
        _context(0.0, windows=6),
        config=config,
        artifact=artifact,
    )

    assert null_values == [0.3]
    assert level == "global"


def test_load_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    payload = _artifact().to_dict()
    payload["extra"] = "unexpected"
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="unknown top-level"):
        load_calibration_artifact(path)


def test_load_rejects_top_level_secret_key(tmp_path: Path) -> None:
    payload = _artifact().to_dict()
    payload["secret_key"] = "raw"
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="secret_key"):
        load_calibration_artifact(path)


def test_load_rejects_nested_secret_key(tmp_path: Path) -> None:
    payload = _artifact(config={"nested": {"secret_key": "raw"}}).to_dict()
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="secret_key"):
        load_calibration_artifact(path)


@pytest.mark.parametrize("key", ["watermark_params", "retry_trace"])
def test_load_rejects_forbidden_trace_fields(tmp_path: Path, key: str) -> None:
    payload = _artifact(config={"nested": {key: {}}}).to_dict()
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=key):
        load_calibration_artifact(path)


@pytest.mark.parametrize("key", ["secret_key", "watermark_params", "retry_trace"])
def test_write_rejects_forbidden_nested_fields(tmp_path: Path, key: str) -> None:
    artifact = _artifact(config={"nested": {key: "bad"}})

    with pytest.raises(ValueError, match=key):
        write_calibration_artifact(tmp_path / "calibration.json", artifact)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sample_scores": [float("nan")]}, "sample_scores"),
        ({"context_nulls": {"bucket": [float("inf")]}}, "context_nulls"),
        ({"threshold_5fpr": float("nan")}, "threshold_5fpr"),
        ({"threshold_5fpr": -0.1}, "threshold_5fpr"),
        ({"context_negative_count": True}, "context_negative_count"),
        ({"sample_negative_count": -1}, "sample_negative_count"),
        ({"sample_scores": [True]}, "sample_scores"),
        ({"global_context_null": ["0.1"]}, "global_context_null"),
        ({"context_nulls": {"bucket": "0.1"}}, "context_nulls"),
    ],
)
def test_load_rejects_invalid_numeric_payloads(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    payload = _artifact().to_dict()
    payload.update(updates)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_calibration_artifact(path)


@pytest.mark.parametrize(
    "artifact",
    [
        _artifact(sample_scores=[float("nan")]),
        _artifact(context_nulls={"bucket": [float("inf")]}),
        _artifact(threshold_5fpr=-0.1),
        _artifact(context_negative_count=True),
        _artifact(global_context_null=[True]),  # type: ignore[list-item]
    ],
)
def test_write_rejects_invalid_numeric_payloads(
    tmp_path: Path,
    artifact: CalibrationArtifact,
) -> None:
    with pytest.raises(ValueError):
        write_calibration_artifact(tmp_path / "calibration.json", artifact)


def test_detector_package_exports_calibration_helpers() -> None:
    assert detect.build_bucket_key is calibration.build_bucket_key
    assert (
        detect.null_for_context_from_artifact
        is calibration.null_for_context_from_artifact
    )
    assert detect.empirical_upper_tail_p is calibration.empirical_upper_tail_p
    assert detect.context_score_from_null is calibration.context_score_from_null
    assert detect.percentile_threshold is calibration.percentile_threshold
