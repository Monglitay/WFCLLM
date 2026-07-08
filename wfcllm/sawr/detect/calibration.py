from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.detect.calibration is deprecated; "
    "import wfcllm.detection.calibration instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.detection.calibration import *  # noqa: F401,F403,E402
