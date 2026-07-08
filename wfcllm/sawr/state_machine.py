from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.state_machine is deprecated; import wfcllm.generation.state_machine instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.generation.state_machine import *  # noqa: F401,F403,E402
