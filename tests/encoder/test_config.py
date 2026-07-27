"""Tests for wfcllm.encoder.config."""

from wfcllm.encoder.config import EncoderConfig


class TestEncoderConfig:
    def test_default_values(self):
        cfg = EncoderConfig()
        assert cfg.model_name == "Salesforce/codet5-base"
        assert cfg.embed_dim == 128
        assert cfg.pooling == "first"
        assert cfg.max_seq_length == 256

    def test_lora_defaults(self):
        cfg = EncoderConfig()
        assert cfg.use_lora is True
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.1

    def test_bf16_default(self):
        cfg = EncoderConfig()
        assert cfg.use_bf16 is True

    def test_disable_lora(self):
        cfg = EncoderConfig(use_lora=False)
        assert cfg.use_lora is False

    def test_disable_bf16(self):
        cfg = EncoderConfig(use_bf16=False)
        assert cfg.use_bf16 is False

    def test_custom_architecture_values(self):
        cfg = EncoderConfig(
            model_name="/local/codet5",
            embed_dim=64,
            pooling="masked_mean",
            max_seq_length=128,
        )
        assert cfg.model_name == "/local/codet5"
        assert cfg.embed_dim == 64
        assert cfg.pooling == "masked_mean"
        assert cfg.max_seq_length == 128
