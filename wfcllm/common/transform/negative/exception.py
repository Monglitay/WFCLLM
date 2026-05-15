"""Deprecated import path. Use wfcllm.lang.python.transform.negative.exception instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative.exception is deprecated; "
    "use wfcllm.lang.python.transform.negative.exception",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative.exception import *  # noqa: E402, F401, F403
