"""Deprecated import path. Use wfcllm.watermark.token_channel.training.teacher instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.teacher is deprecated; "
    "use wfcllm.watermark.token_channel.training.teacher",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.teacher import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.teacher import (  # noqa: E402, F401
    batch_extract_teacher_rows,
    extract_teacher_rows,
    load_teacher_cache,
    save_teacher_cache,
    _batch_forward_all_positions,
    _compute_dynamic_batch_size,
    _extract_single_text_all_positions,
    _get_available_memory_gb,
    _get_vocab_size,
    _group_texts_by_length,
)
