"""Deprecated import path. Use wfcllm.lang.python.transform.positive.formatting instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.positive.formatting is deprecated; "
    "use wfcllm.lang.python.transform.positive.formatting",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.positive.formatting import *  # noqa: E402, F401, F403
