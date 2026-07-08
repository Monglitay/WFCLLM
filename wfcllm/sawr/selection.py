from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.selection is deprecated; use wfcllm.diagnostics.static_selector",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.diagnostics.static_selector import *  # noqa: F401,F403,E402
