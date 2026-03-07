"""Tests for extract config and data structures."""

from __future__ import annotations

from wfcllm.extract.config import BlockScore, DetectionResult, ExtractConfig


class TestExtractConfig:
    def test_defaults(self):
        cfg = ExtractConfig(secret_key="test-key")
        assert cfg.secret_key == "test-key"
        assert cfg.embed_dim == 128
        assert cfg.z_threshold == 3.0

    def test_custom_threshold(self):
        cfg = ExtractConfig(secret_key="k", z_threshold=2.5)
        assert cfg.z_threshold == 2.5


class TestBlockScore:
    def test_fields(self):
        bs = BlockScore(
            block_id="0",
            score=1,
            projection=0.42,
            target_sign=1,
            selected=False,
        )
        assert bs.block_id == "0"
        assert bs.score == 1
        assert bs.projection == 0.42
        assert bs.target_sign == 1
        assert bs.selected is False


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
