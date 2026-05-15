"""Deprecated import path. Use wfcllm.lang.python.transform.engine instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.engine is deprecated; use wfcllm.lang.python.transform.engine",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.engine import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.engine import TransformEngine  # noqa: E402, F401
