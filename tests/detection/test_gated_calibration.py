from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from wfcllm.detection.gated_pipeline import (
    BINOMIAL_P_VALUE_RULE,
    EMPIRICAL_BINOMIAL_SURPRISAL_RULE,
    EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE,
    EMPIRICAL_P_VALUE_RULE,
    QUANTILE_THRESHOLD_RULE,
    GatedCalibrationArtifact,
    GatedDetectionBindings,
    GatedDetectionPipeline,
    empirical_right_tail_plus_one,
)
from wfcllm.detection.scoring import GatedWindowScorer


class _Semantic:
    def score(self, *, window_text: str, parent_descriptor: str):
        return SimpleNamespace(hit=window_text == "hit", stable=True, margin=0.5)


class _Extractor:
    window_contract_version = "python-statement-window/v1"

    def extract(self, final_code: str):
        return SimpleNamespace(windows=tuple(
            SimpleNamespace(
                suitable=True, window_text=token, parent_descriptor="descriptor",
                start_byte=i, end_byte=i + 1,
                close_probability=0.8, suitable_probability=0.9,
            ) for i, token in enumerate(final_code.split())
        ))


def _row(i: int, code: str):
    return {"id": str(i), "dataset": "mbpp", "prompt": "", "final_code": code}


def _bindings() -> GatedDetectionBindings:
    return GatedDetectionBindings(
        gate_bundle_sha256="a" * 64,
        semantic_encoder_sha256="b" * 64,
        lsh_config_sha256="c" * 64,
        key_identifier_sha256="d" * 64,
        negative_corpus_manifest_sha256="e" * 64,
    )


def test_calibration_binds_gate_and_semantic_hashes() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
    )
    artifact = pipeline.calibrate([_row(1, "miss miss"), _row(2, "hit miss")])

    assert artifact.detector_mode == "wfcllm-gated-semantic-window/v1"
    assert artifact.gate_bundle_sha256
    assert artifact.semantic_encoder_sha256
    assert artifact.window_contract_version == "python-statement-window/v1"
    assert artifact.reliable_window_count_buckets
    assert artifact.empirical_p_value_rule == EMPIRICAL_P_VALUE_RULE
    assert "raw_key" not in artifact.to_dict()


def test_empirical_p_value_uses_right_tail_plus_one() -> None:
    assert empirical_right_tail_plus_one(0.5, (0.1, 0.5, 0.9)) == pytest.approx(0.75)


def test_pooled_calibration_keeps_target_fpr_with_variable_window_counts() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
        calibration_group_by="pooled_reliable_hit_rate",
    )

    artifact = pipeline.calibrate(
        [_row(1, "miss miss"), _row(2, "hit miss hit"), _row(3, "hit hit miss miss")]
    )

    assert set(artifact.reliable_window_count_buckets) == {"2"}
    assert len(artifact.reliable_window_count_buckets["2"]) == 3


def test_pooled_quantile_threshold_uses_negative_only_fpr_boundary() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(
            semantic_scorer=_Semantic(),
            minimum_reliable_windows=3,
        ),
        bindings=_bindings(),
        target_fpr=0.05,
        calibration_group_by="pooled_reliable_hit_rate_quantile",
    )
    negatives = [
        *(_row(index, "miss miss miss miss") for index in range(18)),
        _row(18, "hit hit hit miss"),
        _row(19, "hit hit hit hit"),
    ]

    artifact = pipeline.calibrate(negatives)
    negative_results = pipeline.detect(negatives, artifact=artifact)
    below, boundary = pipeline.detect(
        [_row(20, "hit hit hit miss"), _row(21, "hit hit hit hit")],
        artifact=artifact,
    )

    assert artifact.empirical_p_value_rule == QUANTILE_THRESHOLD_RULE
    assert artifact.thresholds_by_reliable_window_count == {"3": 1.0}
    assert sum(result.is_watermarked for result in negative_results) == 1
    assert below.decision == "not_watermarked"
    assert boundary.decision == "watermarked"
    assert boundary.p_value > 0.05


def test_pooled_binomial_calibration_uses_window_level_negative_null() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
        calibration_group_by="pooled_binomial_tail",
    )

    artifact = pipeline.calibrate(
        [_row(1, "miss miss"), _row(2, "hit hit"), _row(3, "hit miss")]
    )

    assert artifact.empirical_p_value_rule == BINOMIAL_P_VALUE_RULE
    assert artifact.reliable_window_count_buckets["2"] == (
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
    )


def test_pooled_binomial_detection_separates_long_hits_from_short_chance_hits() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
        calibration_group_by="pooled_binomial_tail",
    )
    artifact = pipeline.calibrate(
        [_row(1, "miss miss"), _row(2, "hit hit"), _row(3, "hit miss")]
    )

    short, long = pipeline.detect(
        [_row(4, "hit hit"), _row(5, "hit hit hit hit hit hit hit hit hit hit")],
        artifact=artifact,
    )

    assert short.decision == "not_watermarked"
    assert short.p_value == pytest.approx(0.25)
    assert long.decision == "watermarked"
    assert long.p_value == pytest.approx(1 / 1024)


def test_empirical_binomial_surprisal_calibrates_sample_level_count_score() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(
            semantic_scorer=_Semantic(),
            minimum_reliable_windows=2,
        ),
        bindings=_bindings(),
        target_fpr=0.05,
        calibration_group_by="pooled_empirical_binomial_surprisal",
    )
    negatives = [_row(index, "miss miss") for index in range(20)]

    artifact = pipeline.calibrate(negatives)
    result = pipeline.detect(
        [_row(21, "hit hit hit hit")],
        artifact=artifact,
    )[0]

    assert artifact.empirical_p_value_rule == EMPIRICAL_BINOMIAL_SURPRISAL_RULE
    assert artifact.null_hit_probability == pytest.approx(1 / 42)
    assert len(artifact.reliable_window_count_buckets["2"]) == 20
    assert result.decision == "watermarked"
    assert result.p_value == pytest.approx(1 / 21)


def test_empirical_standardized_hit_surplus_uses_negative_null_and_sample_tail() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(
            semantic_scorer=_Semantic(),
            minimum_reliable_windows=2,
        ),
        bindings=_bindings(),
        target_fpr=0.05,
        calibration_group_by="pooled_empirical_standardized_hit_surplus",
    )
    negatives = [_row(index, "miss miss") for index in range(20)]

    artifact = pipeline.calibrate(negatives)
    result = pipeline.detect(
        [_row(21, "hit hit hit hit")],
        artifact=artifact,
    )[0]

    assert (
        artifact.empirical_p_value_rule
        == EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE
    )
    assert artifact.null_hit_probability == pytest.approx(1 / 42)
    assert result.decision == "watermarked"
    assert result.p_value == pytest.approx(1 / 21)


def test_binomial_surprisal_calibration_survives_insufficient_negative_record() -> None:
    """One insufficient-window negative must not break surprisal calibration.

    The sentinel for insufficient records is 0.0 — the domain lower bound of
    the surprisal statistic (all real surprisals are >= 0) — so it sorts before
    every sufficient-record value while satisfying the artifact's non-negative
    domain check.
    """
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(
            semantic_scorer=_Semantic(),
            minimum_reliable_windows=2,
        ),
        bindings=_bindings(),
        target_fpr=0.05,
        calibration_group_by="pooled_empirical_binomial_surprisal",
    )
    negatives = [
        *(_row(index, "hit miss") for index in range(20)),
        _row(20, "miss"),
    ]

    artifact = pipeline.calibrate(negatives)

    assert artifact.empirical_p_value_rule == EMPIRICAL_BINOMIAL_SURPRISAL_RULE
    bucket = artifact.reliable_window_count_buckets["2"]
    # The sentinel enters the pool and counts in the denominator.
    assert len(bucket) == 21
    assert bucket[-1] == 0.0
    assert all(value > 0.0 for value in bucket[:20])
    # The insufficient record's reliable window still feeds the null estimate.
    assert artifact.null_hit_probability == pytest.approx(21 / 43)
    assert GatedCalibrationArtifact.from_dict(artifact.to_dict()) == artifact


def test_standardized_hit_surplus_uses_floor_sentinel() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(
            semantic_scorer=_Semantic(),
            minimum_reliable_windows=2,
        ),
        bindings=_bindings(),
        target_fpr=0.05,
        calibration_group_by="pooled_empirical_standardized_hit_surplus",
    )
    negatives = [
        *(_row(index, "miss miss") for index in range(20)),
        _row(20, "miss"),
    ]

    artifact = pipeline.calibrate(negatives)

    bucket = artifact.reliable_window_count_buckets["2"]
    assert len(bucket) == 21
    assert bucket[-1] == -1e300


def test_pooled_calibration_counts_insufficient_sentinels() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(
            semantic_scorer=_Semantic(),
            minimum_reliable_windows=2,
        ),
        bindings=_bindings(),
        target_fpr=0.5,
        calibration_group_by="pooled_reliable_hit_rate",
    )
    negatives = [
        _row(0, "hit miss"),
        _row(1, "hit hit"),
        _row(2, "miss"),
        _row(3, "miss"),
    ]

    artifact = pipeline.calibrate(negatives)

    # Current contract: insufficient records enter the pool as 0.0.
    assert artifact.reliable_window_count_buckets["2"] == (0.5, 1.0, 0.0, 0.0)
    assert artifact.thresholds_by_reliable_window_count["2"] == pytest.approx(0.0)


def test_hash_mismatch_is_rejected() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
    )
    artifact = pipeline.calibrate([_row(1, "miss miss")])
    bad = replace(artifact, semantic_encoder_sha256="f" * 64)
    with pytest.raises(ValueError, match="semantic_encoder_sha256"):
        pipeline.detect([_row(2, "hit hit")], artifact=bad)


def test_artifact_strict_round_trip() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
    )
    artifact = pipeline.calibrate([_row(1, "miss miss")])
    assert GatedCalibrationArtifact.from_dict(artifact.to_dict()) == artifact
