"""Deprecated import path. Use wfcllm.lang.python.transform.negative instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative is deprecated; "
    "use wfcllm.lang.python.transform.negative",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.negative import (  # noqa: E402, F401
    get_all_negative_rules,
)
