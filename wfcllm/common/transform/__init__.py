"""Deprecated import path. Use wfcllm.lang.python.transform instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform is deprecated; use wfcllm.lang.python.transform",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform import (  # noqa: E402, F401
    Match,
    Rule,
    TransformEngine,
    parse_code,
)
