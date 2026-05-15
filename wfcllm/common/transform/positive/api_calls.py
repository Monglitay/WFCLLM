"""Deprecated import path. Use wfcllm.lang.python.transform.positive.api_calls instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.positive.api_calls is deprecated; "
    "use wfcllm.lang.python.transform.positive.api_calls",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.positive.api_calls import *  # noqa: E402, F401, F403
