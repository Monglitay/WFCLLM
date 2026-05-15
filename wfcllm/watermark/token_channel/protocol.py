"""Deprecated import path. Use wfcllm.watermark.token_channel.core.protocol instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.protocol is deprecated; "
    "use wfcllm.watermark.token_channel.core.protocol",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.protocol import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.protocol import (  # noqa: E402, F401
    PartitionResult,
    build_partition,
    make_prefix_key,
    make_scored_token_key,
)
