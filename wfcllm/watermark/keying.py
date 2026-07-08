from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.keying is deprecated; import wfcllm.semantic.keying instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.semantic.keying import *  # noqa: F401,F403,E402
