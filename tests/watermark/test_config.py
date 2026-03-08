"""Tests for wfcllm.watermark.config."""

from wfcllm.watermark.config import WatermarkConfig


class TestWatermarkConfig:
    def test_required_secret_key(self):
        cfg = WatermarkConfig(secret_key="test-key")
        assert cfg.secret_key == "test-key"

    def test_default_encoder_path(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.encoder_model_path == "data/models/codet5-base"

    def test_default_embed_dim(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.encoder_embed_dim == 128

    def test_default_margin_params(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.margin_base == 0.1
        assert cfg.margin_alpha == 0.05

    def test_default_sampling_params(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.max_retries == 5
        assert cfg.temperature == 0.8
        assert cfg.top_p == 0.95
        assert cfg.top_k == 50

    def test_default_generation_params(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.max_new_tokens == 512
        assert cfg.eos_token_id is None
        assert cfg.enable_fallback is True

    def test_custom_values(self):
        cfg = WatermarkConfig(
            secret_key="my-key",
            margin_base=0.2,
            max_retries=3,
            temperature=0.6,
        )
        assert cfg.margin_base == 0.2
        assert cfg.max_retries == 3
        assert cfg.temperature == 0.6
