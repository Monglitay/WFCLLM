from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.rules is deprecated; import wfcllm.semantic.rules instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.semantic.rules import *  # noqa: F401,F403,E402
from wfcllm.semantic.rules import (  # noqa: F401,E402
    _normalize_candidate_text,
    _quantize_gamma,
)
