from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.detect.pipeline is deprecated; "
    "import wfcllm.detection.pipeline instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.detection.pipeline import *  # noqa: F401,F403,E402
from wfcllm.detection.pipeline import (
    WFCLLMDetectionPipeline as SawrDetectionPipeline,
)
from wfcllm.detection.pipeline import WFCLLMDetectionResult as SawrDetectionResult
