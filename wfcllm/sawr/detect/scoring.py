from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.detect.scoring is deprecated; "
    "import wfcllm.detection.scoring instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.detection.scoring import *  # noqa: F401,F403,E402
from wfcllm.detection.scoring import WFCLLMWindowScorer as SawrWindowScorer
from wfcllm.detection.scoring import (
    load_wfcllm_window_scorer as load_sawr_window_scorer,
)
