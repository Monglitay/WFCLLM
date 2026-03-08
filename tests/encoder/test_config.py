"""Tests for wfcllm.encoder.config."""

from wfcllm.encoder.config import EncoderConfig


class TestEncoderConfig:
    def test_default_values(self):
        cfg = EncoderConfig()
        assert cfg.model_name == "Salesforce/codet5-base"
        assert cfg.embed_dim == 128
        assert cfg.lr == 2e-5
        assert cfg.batch_size == 32
        assert cfg.epochs == 10
        assert cfg.margin == 0.3
        assert cfg.max_seq_length == 256
        assert cfg.warmup_ratio == 0.1
        assert cfg.early_stopping_patience == 3
        assert cfg.negative_ratio == 0.5

    def test_lora_defaults(self):
        cfg = EncoderConfig()
        assert cfg.use_lora is True
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
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

    def test_data_sources(self):
        cfg = EncoderConfig()
        assert "mbpp" in cfg.data_sources
        assert "humaneval" in cfg.data_sources

    def test_custom_values(self):
        cfg = EncoderConfig(lr=1e-4, batch_size=16)
        assert cfg.lr == 1e-4
        assert cfg.batch_size == 16

    def test_paths(self):
        cfg = EncoderConfig()
        assert "checkpoints" in cfg.checkpoint_dir
        assert "results" in cfg.results_dir

    def test_encoder_config_has_local_paths(self):
        cfg = EncoderConfig()
        assert cfg.local_model_dir == "data/models"
        assert cfg.local_dataset_dir == "data/datasets"
