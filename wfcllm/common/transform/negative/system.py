"""Deprecated import path. Use wfcllm.lang.python.transform.negative.system instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative.system is deprecated; "
    "use wfcllm.lang.python.transform.negative.system",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative.system import *  # noqa: E402, F401, F403
