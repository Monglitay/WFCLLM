"""Deprecated import path. Use wfcllm.watermark.token_channel.core.features instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.features is deprecated; "
    "use wfcllm.watermark.token_channel.core.features",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.features import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.features import (  # noqa: E402, F401
    FEATURE_VERSION,
    ExcludedSpan,
    TokenChannelFeatureContext,
    TokenChannelFeatures,
    build_structure_masks,
    build_token_channel_features,
    build_token_channel_features_from_context,
    collect_excluded_token_spans,
    is_structure_safe_span,
    prepare_token_channel_feature_context,
)
