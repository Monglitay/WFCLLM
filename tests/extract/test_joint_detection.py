"""Tests for semantic and lexical joint detection."""

from __future__ import annotations

import pytest

from wfcllm.extract.hypothesis import JointDetectionResult
from wfcllm.extract.hypothesis import LexicalDetectionResult
from wfcllm.extract.hypothesis import fuse_joint_detection
from wfcllm.watermark.token_channel.config import TokenChannelConfig


def test_joint_detection_uses_weighted_lexical_support_factor() -> None:
    lexical = LexicalDetectionResult(
        num_positions_scored=8,
        num_green_hits=6,
        green_fraction=0.75,
        lexical_z_score=2.0,
        lexical_p_value=0.1,
    )
    config = TokenChannelConfig(enabled=True)
    config.joint.semantic_weight = 1.0
    config.joint.lexical_weight = 0.5
    config.joint.lexical_full_weight_min_positions = 16
    config.joint.threshold = 4.0

    result = fuse_joint_detection(semantic_z_score=3.0, lexical_result=lexical, config=config)

    assert result.joint_score == pytest.approx(3.5)
    assert result.p_joint == pytest.approx(1.0 - 0.9997673709209645)
    assert result.prediction is False
    assert result.confidence == pytest.approx(1.0 - result.p_joint)
    assert result.rationale == "semantic borderline, lexical supportive"


def test_lexical_result_can_be_promoted_for_lexical_only_mode() -> None:
    lexical = LexicalDetectionResult(
        num_positions_scored=12,
        num_green_hits=10,
        green_fraction=10 / 12,
        lexical_z_score=4.2,
        lexical_p_value=0.000013,
    )

    result = lexical.to_joint_equivalent(threshold=4.0)

    assert isinstance(result, JointDetectionResult)
    assert result.joint_score == pytest.approx(4.2)
    assert result.p_joint == pytest.approx(0.000013)
    assert result.prediction is True
    assert result.rationale == "lexical-only evidence"
