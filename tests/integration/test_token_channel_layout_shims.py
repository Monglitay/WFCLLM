"""Tests for wfcllm/watermark/token_channel/* three-layer migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import warnings


# --- core: new path works ---

def test_core_config_new_path_importable():
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
    assert callable(TokenChannelConfig)
