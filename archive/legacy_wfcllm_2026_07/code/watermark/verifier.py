from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.verifier is deprecated; import wfcllm.semantic.verifier instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.semantic.verifier import *  # noqa: F401,F403,E402
