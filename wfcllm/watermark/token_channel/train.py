"""Deprecated import path. Use wfcllm.watermark.token_channel.training.trainer instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train is deprecated; "
    "use wfcllm.watermark.token_channel.training.trainer",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.trainer import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.trainer import (  # noqa: E402, F401
    TokenChannelEpochMetrics,
    TokenChannelTrainingEvidence,
    build_token_channel_batch,
    build_training_evidence,
    evaluate_batch_loss,
    main,
    run_training_step,
    save_token_channel_training_artifacts,
    save_training_evidence,
    train_one_epoch,
)
