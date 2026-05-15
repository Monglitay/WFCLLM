"""Deprecated import path. Use wfcllm.lang.python.transform.negative.identifier instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative.identifier is deprecated; "
    "use wfcllm.lang.python.transform.negative.identifier",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative.identifier import *  # noqa: E402, F401, F403
