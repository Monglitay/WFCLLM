from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.semantic_lsh is deprecated; import wfcllm.semantic.lsh instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.semantic.lsh import *  # noqa: F401,F403,E402
