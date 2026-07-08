from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.lsh_space is deprecated; import wfcllm.semantic.lsh_space instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.semantic.lsh_space import *  # noqa: F401,F403,E402
