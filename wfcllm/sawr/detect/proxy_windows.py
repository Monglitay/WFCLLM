from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.detect.proxy_windows is deprecated; "
    "import wfcllm.detection.proxy_windows instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.detection.proxy_windows import *  # noqa: F401,F403,E402
