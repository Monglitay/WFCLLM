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


def test_core_features_new_path_importable():
    from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
    assert callable(TokenChannelFeatures)


def test_core_features_old_path_emits_warning():
    import importlib
    import sys
    key = "wfcllm.watermark.token_channel.features"
    original = sys.modules.pop(key, None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(key)
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "core.features" in str(w.message)
                for w in caught
            )
    finally:
        if original is not None:
            sys.modules[key] = original
        else:
            sys.modules.pop(key, None)


def test_core_features_symbol_identity():
    from wfcllm.watermark.token_channel.features import TokenChannelFeatures as old_cls
    from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures as new_cls
    assert old_cls is new_cls


def test_core_model_new_path_importable():
    from wfcllm.watermark.token_channel.core.model import TokenChannelModel
    assert callable(TokenChannelModel)


def test_core_model_old_path_emits_warning():
    import importlib
    import sys
    key = "wfcllm.watermark.token_channel.model"
    original = sys.modules.pop(key, None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(key)
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "core.model" in str(w.message)
                for w in caught
            )
    finally:
        if original is not None:
            sys.modules[key] = original
        else:
            sys.modules.pop(key, None)


def test_core_model_symbol_identity():
    from wfcllm.watermark.token_channel.model import TokenChannelModel as old_cls
    from wfcllm.watermark.token_channel.core.model import TokenChannelModel as new_cls
    assert old_cls is new_cls


def test_runtime_injector_new_path_importable():
    from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime
    assert callable(TokenChannelRuntime)


def test_runtime_package_level_reexport():
    """Old code does `from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime`.

    Note: no DeprecationWarning is emitted because runtime/ is now a real subpackage,
    and emitting a warning from runtime/__init__.py would fire on every canonical import too.
    """
    from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime as pkg_cls
    from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime as mod_cls
    assert pkg_cls is mod_cls


def test_training_teacher_new_path_importable():
    from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows
    assert callable(extract_teacher_rows)


def test_training_teacher_old_path_emits_warning():
    import importlib
    import sys
    key = "wfcllm.watermark.token_channel.teacher"
    original = sys.modules.pop(key, None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(key)
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "training.teacher" in str(w.message)
                for w in caught
            )
    finally:
        if original is not None:
            sys.modules[key] = original
        else:
            sys.modules.pop(key, None)


def test_training_teacher_symbol_identity():
    from wfcllm.watermark.token_channel.teacher import extract_teacher_rows as old_fn
    from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows as new_fn
    assert old_fn is new_fn


def test_training_trainer_new_path_importable():
    from wfcllm.watermark.token_channel.training.trainer import train_one_epoch
    assert callable(train_one_epoch)


def test_training_trainer_old_path_emits_warning():
    import importlib
    import sys
    key = "wfcllm.watermark.token_channel.train"
    original = sys.modules.pop(key, None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(key)
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "training.trainer" in str(w.message)
                for w in caught
            )
    finally:
        if original is not None:
            sys.modules[key] = original
        else:
            sys.modules.pop(key, None)


def test_training_trainer_symbol_identity():
    from wfcllm.watermark.token_channel.train import train_one_epoch as old_fn
    from wfcllm.watermark.token_channel.training.trainer import train_one_epoch as new_fn
    assert old_fn is new_fn
