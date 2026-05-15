"""Deprecated import path. Use wfcllm.lang.python.transform.positive instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.positive is deprecated; "
    "use wfcllm.lang.python.transform.positive",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.positive import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.positive import (  # noqa: E402, F401
    get_all_positive_rules,
)
