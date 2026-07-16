from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from wfcllm.detection.gated_pipeline import (
    EMPIRICAL_P_VALUE_RULE,
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
