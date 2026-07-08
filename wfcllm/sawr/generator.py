from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.generator is deprecated; import wfcllm.generation.generator instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.generation.generator import *  # noqa: F401,F403,E402
