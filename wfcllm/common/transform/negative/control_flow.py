"""Deprecated import path. Use wfcllm.lang.python.transform.negative.control_flow instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative.control_flow is deprecated; "
    "use wfcllm.lang.python.transform.negative.control_flow",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative.control_flow import *  # noqa: E402, F401, F403
