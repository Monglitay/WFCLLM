"""Deprecated import path. Use wfcllm.extract.calibration.threshold instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.extract.calibrator is deprecated; use wfcllm.extract.calibration.threshold",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.extract.calibration.threshold import *  # noqa: E402, F401, F403
from wfcllm.extract.calibration.threshold import ThresholdCalibrator  # noqa: E402, F401
