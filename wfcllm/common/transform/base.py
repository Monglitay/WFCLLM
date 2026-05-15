"""Deprecated import path. Use wfcllm.lang.python.transform.base instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.base is deprecated; use wfcllm.lang.python.transform.base",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.base import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.base import (  # noqa: E402, F401
    PY_LANGUAGE,
    Match,
    Rule,
    parse_code,
)
