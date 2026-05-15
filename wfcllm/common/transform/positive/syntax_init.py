"""Deprecated import path. Use wfcllm.lang.python.transform.positive.syntax_init instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.positive.syntax_init is deprecated; "
    "use wfcllm.lang.python.transform.positive.syntax_init",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.positive.syntax_init import *  # noqa: E402, F401, F403
