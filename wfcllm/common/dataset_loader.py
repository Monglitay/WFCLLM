"""Deprecated import path. Use wfcllm.datasets.loaders.local instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.dataset_loader is deprecated; use wfcllm.datasets.loaders.local",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.datasets.loaders.local import *  # noqa: E402, F401, F403
from wfcllm.datasets.loaders.local import (  # noqa: E402, F401
    SUPPORTED_DATASETS,
    load_prompts,
    load_reference_solutions,
)
