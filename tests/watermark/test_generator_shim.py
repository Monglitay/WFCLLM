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
        from wfcllm.watermark.generator import WatermarkGenerator as W_old
    assert W_new is W_old
