from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.kv_cache is deprecated; import wfcllm.generation.kv_cache instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.generation.kv_cache import *  # noqa: F401,F403,E402
