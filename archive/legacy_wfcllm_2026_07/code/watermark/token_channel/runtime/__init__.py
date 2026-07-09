"""Token-channel runtime layer.

Public canonical path: wfcllm.watermark.token_channel.runtime.injector
Importing from this package directly (e.g. `from … import TokenChannelRuntime`) is
preserved for backward compatibility; new code should import from the explicit submodule.
"""
from __future__ import annotations

from wfcllm.watermark.token_channel.runtime.injector import *  # noqa: F401, F403
from wfcllm.watermark.token_channel.runtime.injector import (  # noqa: F401
    TokenChannelDecision,
    TokenChannelRuntime,
)
