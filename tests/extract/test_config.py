"""Tests for extract config and data structures."""

from __future__ import annotations

from wfcllm.extract.config import (
    AdaptiveDetectionConfig,
    BlockScore,
    DetectionResult,
    ExtractConfig,
)
from wfcllm.watermark.token_channel.config import TokenChannelConfig


class TestExtractConfig:
    def test_defaults(self):
        cfg = ExtractConfig(secret_key="test-key")
        assert cfg.secret_key == "test-key"
        assert cfg.embed_dim == 128
        assert cfg.fpr_threshold == 3.0

    def test_adaptive_detection_defaults(self):
        cfg = ExtractConfig(secret_key="test-key")
        adaptive = cfg.adaptive_detection
        assert isinstance(adaptive, AdaptiveDetectionConfig)
        assert adaptive.prefer_adaptive is False
        assert adaptive.require_block_contract_check is True
        assert adaptive.fail_on_structure_mismatch is True
        assert adaptive.warn_on_numeric_mismatch is True
        assert adaptive.exclude_invalid_samples is True

    def test_custom_threshold(self):
        cfg = ExtractConfig(secret_key="k", fpr_threshold=2.5)
        assert cfg.fpr_threshold == 2.5

    def test_token_channel_defaults(self):
        cfg = ExtractConfig(secret_key="test-key")
        token_channel = cfg.token_channel
        assert isinstance(token_channel, TokenChannelConfig)
        assert token_channel.mode == "dual-channel"
        assert token_channel.joint_threshold == 4.0

    def test_token_channel_can_be_overridden(self):
        token_channel = TokenChannelConfig(enabled=True, mode="semantic-only")
        cfg = ExtractConfig(secret_key="test-key", token_channel=token_channel)
        assert cfg.token_channel is token_channel
        assert cfg.token_channel.mode == "semantic-only"


class TestBlockScore:
    def test_fields(self):
        bs = BlockScore(
            block_id="0",
            score=1,
            min_margin=0.42,
            selected=False,
        )
        assert bs.block_id == "0"
        assert bs.score == 1
        assert bs.min_margin == 0.42
        assert bs.selected is False


def test_block_score_has_min_margin():
    from wfcllm.extract.config import BlockScore
    s = BlockScore(block_id="0", score=1, min_margin=0.3)
    assert s.min_margin == 0.3

def test_block_score_no_projection_field():
    import dataclasses
    from wfcllm.extract.config import BlockScore
    fields = {f.name for f in dataclasses.fields(BlockScore)}
    assert "projection" not in fields
    assert "target_sign" not in fields


class TestDetectionResult:
    def test_watermarked(self):
        dr = DetectionResult(
            is_watermarked=True,
            z_score=3.5,
            p_value=0.0002,
            total_blocks=10,
            independent_blocks=6,
            hit_blocks=5,
            block_details=[],
        )
        assert dr.is_watermarked is True
        assert dr.z_score == 3.5
        assert dr.independent_blocks == 6
        assert dr.hit_blocks == 5


class TestPublicAPI:
    def test_high_level_imports(self):
        from wfcllm.extract import (
            DetectionResult,
            ExtractConfig,
            WatermarkDetector,
        )

    def test_low_level_imports(self):
        from wfcllm.extract import (
            BlockScore,
            BlockScorer,
            DPSelector,
            HypothesisTester,
        )
