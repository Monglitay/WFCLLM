"""Deprecated import path. Use wfcllm.watermark.orchestrator instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.generator is deprecated; use wfcllm.watermark.orchestrator",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.orchestrator import (
    EmbedStats,
    GenerateResult,
    TokenChannelRuntimeState,
    WatermarkGenerator,
)

__all__ = ["EmbedStats", "GenerateResult", "TokenChannelRuntimeState", "WatermarkGenerator"]
