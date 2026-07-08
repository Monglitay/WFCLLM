from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.boundary is deprecated; import wfcllm.generation.boundary instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.generation.boundary import *  # noqa: F401,F403,E402
