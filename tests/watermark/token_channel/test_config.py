"""Tests for token-channel configuration."""

import pytest

from wfcllm.watermark.token_channel.config import (
    TokenChannelConfig,
    TokenChannelJointConfig,
)


def test_token_channel_joint_defaults():
    cfg = TokenChannelJointConfig()
    assert cfg.semantic_weight == 1.0
    assert cfg.lexical_weight == 0.75
    assert cfg.lexical_full_weight_min_positions == 32
    assert cfg.threshold == 4.0


def test_token_channel_defaults():
    cfg = TokenChannelConfig()
    assert cfg.enabled is False
    assert cfg.mode == "dual-channel"
    assert cfg.model_path == "data/models/token-channel"
    assert cfg.context_width == 128
    assert cfg.switch_threshold == 0.0
    assert cfg.delta == 2.0
    assert cfg.ignore_repeated_ngrams is False
    assert cfg.ignore_repeated_prefixes is False
    assert cfg.debug_mode is False
    assert cfg.lexical_min_block_tokens == 8
    assert cfg.lexical_retry_decay_start == 2
    assert cfg.lexical_retry_disable_after == 4
    assert cfg.lexical_gate_probe_tokens == 16
    assert cfg.lexical_gate_min_fraction == 0.10
    assert cfg.joint == TokenChannelJointConfig()


def test_supported_modes_are_available():
    assert TokenChannelConfig(mode="semantic-only").mode == "semantic-only"
    assert TokenChannelConfig(mode="lexical-only").mode == "lexical-only"
    assert TokenChannelConfig(mode="dual-channel").mode == "dual-channel"


def test_invalid_mode_raises_value_error():
    with pytest.raises(ValueError, match="mode"):
        TokenChannelConfig(mode="invalid")


def test_invalid_numeric_invariants_raise_value_error():
    with pytest.raises(ValueError, match="lexical_retry_disable_after"):
        TokenChannelConfig(
            lexical_retry_decay_start=3,
            lexical_retry_disable_after=2,
        )

    with pytest.raises(ValueError, match="lexical_gate_min_fraction"):
        TokenChannelConfig(lexical_gate_min_fraction=1.5)


def test_from_mapping_accepts_channel_mode_alias():
    cfg = TokenChannelConfig.from_mapping(
        {
            "enabled": True,
            "channel_mode": "lexical-only",
            "delta": 1.5,
            "joint": {"threshold": 5.0},
        }
    )

    assert cfg.enabled is True
    assert cfg.mode == "lexical-only"
    assert cfg.delta == 1.5
    assert cfg.joint_threshold == 5.0


def test_from_mapping_rejects_invalid_channel_mode():
    with pytest.raises(ValueError, match="channel_mode"):
        TokenChannelConfig.from_mapping({"channel_mode": "invalid"})
