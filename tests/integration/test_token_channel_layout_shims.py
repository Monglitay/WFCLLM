"""Tests for wfcllm/watermark/token_channel/* three-layer migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import warnings


# --- core: new path works ---

def test_core_config_new_path_importable():
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
    assert callable(TokenChannelConfig)


def test_core_config_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.config", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.config")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "core.config" in str(w.message)
            for w in caught
        )


def test_core_config_symbol_identity():
    from wfcllm.watermark.token_channel.config import TokenChannelConfig as old_cls
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig as new_cls
    assert old_cls is new_cls
