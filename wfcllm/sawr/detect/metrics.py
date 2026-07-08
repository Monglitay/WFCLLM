from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.detect.metrics is deprecated; "
    "import wfcllm.detection.metrics instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.detection.metrics import *  # noqa: F401,F403,E402
