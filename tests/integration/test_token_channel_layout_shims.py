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
    key = "wfcllm.watermark.token_channel.config"
    original = sys.modules.pop(key, None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(key)
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "core.config" in str(w.message)
                for w in caught
            )
    finally:
        if original is not None:
            sys.modules[key] = original
        else:
            sys.modules.pop(key, None)


def test_core_config_symbol_identity():
    from wfcllm.watermark.token_channel.config import TokenChannelConfig as old_cls
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig as new_cls
    assert old_cls is new_cls


def test_core_protocol_new_path_importable():
    from wfcllm.watermark.token_channel.core.protocol import build_partition
    assert callable(build_partition)


def test_core_protocol_old_path_emits_warning():
    import importlib
    import sys
    key = "wfcllm.watermark.token_channel.protocol"
    original = sys.modules.pop(key, None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(key)
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "core.protocol" in str(w.message)
                for w in caught
            )
    finally:
        if original is not None:
            sys.modules[key] = original
        else:
            sys.modules.pop(key, None)


def test_core_protocol_symbol_identity():
    from wfcllm.watermark.token_channel.protocol import build_partition as old_fn
    from wfcllm.watermark.token_channel.core.protocol import build_partition as new_fn
    assert old_fn is new_fn
