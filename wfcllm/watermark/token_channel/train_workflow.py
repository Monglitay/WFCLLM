"""Deprecated import path. Use wfcllm.watermark.token_channel.training.workflow instead."""
from __future__ import annotations

import sys
import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train_workflow is deprecated; "
    "use wfcllm.watermark.token_channel.training.workflow",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training import workflow as _workflow_module

# Replace this shim module with the real module so that monkeypatch and
# attribute lookups on `train_workflow` resolve against the canonical module.
sys.modules[__name__] = _workflow_module
