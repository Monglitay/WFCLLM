from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.pipeline is deprecated; import wfcllm.generation.pipeline instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.generation.pipeline import *  # noqa: F401,F403,E402
