"""Deprecated import path. Use wfcllm.extract.calibration.negative_corpus instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.extract.negative_corpus is deprecated; use wfcllm.extract.calibration.negative_corpus",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.extract.calibration.negative_corpus import *  # noqa: E402, F401, F403
from wfcllm.extract.calibration.negative_corpus import (  # noqa: E402, F401
    NegativeCorpusConfig,
    NegativeCorpusGenerator,
)
