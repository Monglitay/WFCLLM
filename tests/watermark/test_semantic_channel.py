"""Unit tests for SemanticChannel — the LSH+gamma slice of WatermarkGenerator."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_semantic_channel_module_exists():
    from wfcllm.watermark import semantic_channel
    assert hasattr(semantic_channel, "SemanticChannel")


def test_semantic_channel_holds_orchestrator_reference():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace()
    sc = SemanticChannel(orch)
    assert sc._orch is orch
