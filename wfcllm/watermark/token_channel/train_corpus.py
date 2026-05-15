"""Deprecated import path. Use wfcllm.watermark.token_channel.training.corpus instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train_corpus is deprecated; "
    "use wfcllm.watermark.token_channel.training.corpus",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.corpus import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.corpus import (  # noqa: E402, F401
    TRAINING_CACHE_SCHEMA_VERSION,
    build_augmented_variants,
    build_training_rows,
    load_training_cache,
    save_training_cache,
    save_training_cache_streaming,
)
