from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.config is deprecated; import wfcllm.method.config instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.method.config import *  # noqa: F401,F403,E402
