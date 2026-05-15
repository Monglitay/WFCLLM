"""Deprecated import path. Use wfcllm.watermark.orchestrator instead.

This shim re-exports both the public dataclasses/classes that lived on the
old `wfcllm.watermark.generator` module AND the third-party symbols that the
original file imported at top-level. The latter matters because pre-Phase-8
tests `monkeypatch.setattr("wfcllm.watermark.generator.GenerationContext", ...)`
etc. — the attribute lookup must still succeed on this module.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.generator is deprecated; use wfcllm.watermark.orchestrator",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.watermark.context import GenerationContext
from wfcllm.watermark.orchestrator import (
    EmbedStats,
    GenerateResult,
    TokenChannelRuntimeState,
    WatermarkGenerator,
)
from wfcllm.watermark.retry_loop import RetryLoop
from wfcllm.watermark.token_channel.core.features import build_token_channel_features
from wfcllm.watermark.token_channel.core.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime

__all__ = [
    "EmbedStats",
    "GenerateResult",
    "GenerationContext",
    "RetryLoop",
    "TokenChannelRuntime",
    "TokenChannelRuntimeState",
    "WatermarkGenerator",
    "build_block_contracts",
    "build_token_channel_features",
    "load_token_channel_artifact",
]
