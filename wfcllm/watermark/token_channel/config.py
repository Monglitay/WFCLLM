"""Deprecated import path. Use wfcllm.watermark.token_channel.core.config instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.config is deprecated; "
    "use wfcllm.watermark.token_channel.core.config",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.config import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.config import (  # noqa: E402, F401
    TokenChannelConfig,
    TokenChannelJointConfig,
)
