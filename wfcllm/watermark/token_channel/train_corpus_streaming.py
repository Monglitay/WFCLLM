"""Deprecated import path. Use wfcllm.watermark.token_channel.training.corpus_streaming instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train_corpus_streaming is deprecated; "
    "use wfcllm.watermark.token_channel.training.corpus_streaming",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.corpus_streaming import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.corpus_streaming import (  # noqa: E402, F401
    count_training_cache_rows,
    load_rows_by_indices,
    split_training_cache_streaming,
    stream_training_cache,
)
