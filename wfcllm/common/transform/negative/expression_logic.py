"""Deprecated import path. Use wfcllm.lang.python.transform.negative.expression_logic instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative.expression_logic is deprecated; "
    "use wfcllm.lang.python.transform.negative.expression_logic",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative.expression_logic import *  # noqa: E402, F401, F403
