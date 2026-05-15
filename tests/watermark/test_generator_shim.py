"""Shim test: old wfcllm.watermark.generator path keeps working after Phase 8."""
from __future__ import annotations

import warnings


def test_orchestrator_module_exists():
    from wfcllm.watermark import orchestrator
    assert hasattr(orchestrator, "WatermarkGenerator")
    assert hasattr(orchestrator, "GenerateResult")
    assert hasattr(orchestrator, "EmbedStats")
    assert hasattr(orchestrator, "TokenChannelRuntimeState")


def test_orchestrator_exports_same_class_as_old_path():
    from wfcllm.watermark.orchestrator import WatermarkGenerator as W_new
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from wfcllm.watermark.orchestrator import WatermarkGenerator as W_old
    assert W_new is W_old


def test_deprecation_warning_emitted_on_old_path_import():
    import importlib
    import sys
    for mod in ["wfcllm.watermark.generator"]:
        sys.modules.pop(mod, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.import_module("wfcllm.watermark.generator")
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("wfcllm.watermark.generator" in str(w.message) for w in deprecations)


def test_old_symbols_resolvable_through_shim():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from wfcllm.watermark.orchestrator import (
            EmbedStats,
            GenerateResult,
            TokenChannelRuntimeState,
            WatermarkGenerator,
        )
    from wfcllm.watermark.orchestrator import (
        EmbedStats as E2,
        GenerateResult as G2,
        TokenChannelRuntimeState as T2,
        WatermarkGenerator as W2,
    )
    assert EmbedStats is E2
    assert GenerateResult is G2
    assert TokenChannelRuntimeState is T2
    assert WatermarkGenerator is W2
