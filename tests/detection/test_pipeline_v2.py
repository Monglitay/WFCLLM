from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wfcllm.detection.pipeline_v2 import (
    V2_CALIBRATION_ARTIFACT_TYPE,
    V2_CALIBRATION_SCHEMA_VERSION,
    WFCLLMV2DetectionConfig,
    WFCLLMV2DetectionPipeline,
    load_v2_calibration_artifact,
    strict_upper_order_threshold,
    write_v2_calibration_artifact,
)


class _FakeScorer:
    def score_code(self, code: str):
        value = float(code.rsplit(" ", 1)[-1])
        return SimpleNamespace(
            raw_score=value,
            unit_count=1,
            duplicate_count=0,
            total_bits=16,
            matched_bits=round(value * 16),
            unit_evidence=(),
        )


def _row(index: int, score: float) -> dict[str, str]:
    return {
        "id": f"sample-{index}",
        "dataset": "humaneval",
        "prompt": "def target():\n",
        "final_code": f"def target():\n    return {score} # {score}",
    }


def _config(
    secret_key: str = "secret",
    aggregation: str = "trimmed_unit_mean",
) -> WFCLLMV2DetectionConfig:
    return WFCLLMV2DetectionConfig(
        secret_key=secret_key,
        signature_bits=16,
        min_canonical_units=1,
        target_fpr=0.05,
        aggregation=aggregation,
    )


def test_strict_upper_order_threshold_is_conservative_with_ties() -> None:
    threshold = strict_upper_order_threshold([0.0] * 18 + [0.5, 0.5], 0.05)

    assert threshold > 0.5


def test_v2_pipeline_calibrates_and_detects_from_strict_final_code_rows() -> None:
    config = _config()
    pipeline = WFCLLMV2DetectionPipeline(config=config, scorer=_FakeScorer())
    negatives = [_row(index, index / 100.0) for index in range(82)]

    artifact = pipeline.calibrate(negatives)
    result = pipeline.detect_one(_row(100, 0.99), artifact=artifact)

    assert artifact.artifact_type == V2_CALIBRATION_ARTIFACT_TYPE
    assert artifact.schema_version == V2_CALIBRATION_SCHEMA_VERSION
    assert artifact.sample_negative_count == 82
    assert "secret_key" not in artifact.to_dict()["config"]
    assert artifact.config["secret_key_sha256"]
    assert result.is_watermarked is True
    assert result.unit_count == 1
    assert result.total_signature_bits == 16
    assert result.insufficient_evidence is False


def test_v2_pipeline_rejects_sidecar_fields() -> None:
    pipeline = WFCLLMV2DetectionPipeline(config=_config(), scorer=_FakeScorer())
    artifact = pipeline.calibrate([_row(index, index / 100.0) for index in range(20)])
    record = {**_row(21, 0.5), "generation_score": 0.5}

    with pytest.raises(ValueError, match="exactly four fields"):
        pipeline.detect_one(record, artifact=artifact)


def test_v2_loader_rejects_v1_or_unversioned_artifacts(tmp_path: Path) -> None:
    for name, payload in (
        (
            "v1.json",
            {
                "artifact_type": "wfcllm_detection_calibration",
                "schema_version": "wfcllm-detect-calibration/v1",
            },
        ),
        ("unversioned.json", {"artifact_type": V2_CALIBRATION_ARTIFACT_TYPE}),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="v2 calibration"):
            load_v2_calibration_artifact(path)


def test_v1_loader_rejects_v2_artifact(tmp_path: Path) -> None:
    from wfcllm.detection.calibration import load_calibration_artifact

    pipeline = WFCLLMV2DetectionPipeline(config=_config(), scorer=_FakeScorer())
    artifact = pipeline.calibrate([_row(index, index / 100.0) for index in range(20)])
    path = tmp_path / "v2.json"
    write_v2_calibration_artifact(path, artifact)

    with pytest.raises(ValueError):
        load_calibration_artifact(path)


def test_v2_artifact_rejects_wrong_key_and_config(tmp_path: Path) -> None:
    original = WFCLLMV2DetectionPipeline(config=_config("left"), scorer=_FakeScorer())
    artifact = original.calibrate([_row(index, index / 100.0) for index in range(20)])
    path = tmp_path / "v2.json"
    write_v2_calibration_artifact(path, artifact)
    loaded = load_v2_calibration_artifact(path)
    wrong = WFCLLMV2DetectionPipeline(config=_config("right"), scorer=_FakeScorer())

    with pytest.raises(ValueError, match="incompatible"):
        wrong.detect_one(_row(22, 0.8), artifact=loaded)


def test_v2_artifact_writer_rejects_recursive_raw_secret(tmp_path: Path) -> None:
    pipeline = WFCLLMV2DetectionPipeline(config=_config(), scorer=_FakeScorer())
    artifact = pipeline.calibrate([_row(index, index / 100.0) for index in range(20)])
    object.__setattr__(artifact, "config", {**artifact.config, "nested": {"secret_key": "x"}})

    with pytest.raises(ValueError, match="forbidden"):
        write_v2_calibration_artifact(tmp_path / "bad.json", artifact)


def test_standardized_aggregation_artifact_is_explicit_and_not_silently_mixed(
    tmp_path: Path,
) -> None:
    repaired = WFCLLMV2DetectionPipeline(
        config=_config(aggregation="standardized_bit_sum"),
        scorer=_FakeScorer(),
    )
    artifact = repaired.calibrate([_row(index, index / 100.0) for index in range(20)])
    path = tmp_path / "standardized.json"
    write_v2_calibration_artifact(path, artifact)
    loaded = load_v2_calibration_artifact(path)

    assert loaded.config["aggregation"] == "standardized_bit_sum"
    trimmed = WFCLLMV2DetectionPipeline(
        config=_config(aggregation="trimmed_unit_mean"),
        scorer=_FakeScorer(),
    )
    with pytest.raises(ValueError, match="incompatible"):
        trimmed.detect_one(_row(22, 0.8), artifact=loaded)
