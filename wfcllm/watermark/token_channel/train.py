"""Deprecated import path. Use wfcllm.watermark.token_channel.training.trainer instead."""
from __future__ import annotations

import sys
import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train is deprecated; "
    "use wfcllm.watermark.token_channel.training.trainer",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training import trainer as _trainer_module

# Replace this shim module with the real module so that monkeypatch and
# attribute lookups on `train` resolve against the canonical module
# (mirrors the train_workflow shim approach for the same reason —
# tests/watermark/token_channel/test_model.py:322 does
# `import wfcllm.watermark.token_channel.train as train_module` then
# monkeypatches private attributes on it).
sys.modules[__name__] = _trainer_module
