"""Tests for wfcllm.watermark.config."""

from wfcllm.watermark.config import AdaptiveGammaConfig, WatermarkConfig
from wfcllm.watermark.token_channel.config import TokenChannelConfig


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


def test_repetition_penalty_default():
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.repetition_penalty == 1.3


def test_repetition_penalty_custom():
    cfg = WatermarkConfig(secret_key="k", repetition_penalty=1.5)
    assert cfg.repetition_penalty == 1.5


def test_lsh_defaults():
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.lsh_d == 3
    assert cfg.lsh_gamma == 0.5


def test_cascade_config_defaults():
    """New cascade fields have correct defaults."""
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.enable_cascade is True
    assert cfg.cascade_max_depth == 1
    assert cfg.cuda_empty_cache_interval == 10
    assert cfg.retry_token_budget is None


def test_cascade_config_custom():
    """Cascade fields can be overridden."""
    cfg = WatermarkConfig(
        secret_key="k",
        enable_cascade=True,
        cascade_max_depth=3,
        cuda_empty_cache_interval=5,
        retry_token_budget=128,
    )
    assert cfg.enable_cascade is True
    assert cfg.cascade_max_depth == 3
    assert cfg.cuda_empty_cache_interval == 5
    assert cfg.retry_token_budget == 128


def test_adaptive_gamma_defaults():
    cfg = WatermarkConfig(secret_key="k")
    adaptive = cfg.adaptive_gamma
    assert isinstance(adaptive, AdaptiveGammaConfig)
    assert adaptive.enabled is False
    assert adaptive.strategy == "piecewise_quantile"
    assert adaptive.gamma_min == 0.25
    assert adaptive.gamma_max == 0.95
    assert adaptive.anchors == {
        "p10": 0.95,
        "p50": 0.75,
        "p75": 0.55,
        "p90": 0.35,
        "p95": 0.25,
    }


def test_token_channel_defaults():
    cfg = WatermarkConfig(secret_key="k")
    token_channel = cfg.token_channel
    assert isinstance(token_channel, TokenChannelConfig)
    assert token_channel.mode == "dual-channel"
    assert token_channel.lexical_min_block_tokens == 8
    assert token_channel.lexical_retry_decay_start == 2
    assert token_channel.lexical_retry_disable_after == 4
    assert token_channel.lexical_gate_probe_tokens == 16
    assert token_channel.lexical_gate_min_fraction == 0.10


def test_token_channel_can_be_overridden():
    token_channel = TokenChannelConfig(enabled=True, mode="lexical-only")
    cfg = WatermarkConfig(secret_key="k", token_channel=token_channel)
    assert cfg.token_channel is token_channel
    assert cfg.token_channel.enabled is True
    assert cfg.token_channel.mode == "lexical-only"
