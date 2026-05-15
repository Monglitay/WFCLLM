"""Deprecated import path. Use wfcllm.watermark.token_channel.core.model instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.model is deprecated; "
    "use wfcllm.watermark.token_channel.core.model",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.model import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.model import (  # noqa: E402, F401
    TokenChannelArtifact,
    TokenChannelArtifactMetadata,
    TokenChannelCheckpointExport,
    TokenChannelCompatibility,
    TokenChannelLossWeights,
    TokenChannelModel,
    TokenChannelModelOutput,
    check_token_channel_compatibility,
    export_token_channel_checkpoint,
    load_token_channel_artifact,
    load_token_channel_artifact_metadata,
    load_training_state,
    require_token_channel_compatibility,
    save_token_channel_artifact_metadata,
)
