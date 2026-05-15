"""Deprecated import path. Use wfcllm.lang.python.transform.negative.data_structure instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative.data_structure is deprecated; "
    "use wfcllm.lang.python.transform.negative.data_structure",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative.data_structure import *  # noqa: E402, F401, F403
