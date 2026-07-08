from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.detect.config is deprecated; "
    "import wfcllm.detection.config instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.detection.config import *  # noqa: F401,F403,E402
from wfcllm.detection.config import WFCLLMDetectionConfig as SawrDetectionConfig
