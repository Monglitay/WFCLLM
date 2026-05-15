"""Deprecated import path. Use wfcllm.lang.python.transform.positive.identifier instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.positive.identifier is deprecated; "
    "use wfcllm.lang.python.transform.positive.identifier",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.positive.identifier import *  # noqa: E402, F401, F403
