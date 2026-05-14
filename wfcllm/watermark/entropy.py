"""Deprecated: re-exports for backwards compatibility.

Moved to ``wfcllm.watermark.adaptive_gamma.entropy`` in the Phase 3 refactor
(see docs/superpowers/specs/2026-05-14-repo-refactor-design.md §4.2, §8.2).
"""
import warnings

warnings.warn(
    "wfcllm.watermark.entropy is deprecated; "
    "use wfcllm.watermark.adaptive_gamma.entropy instead.",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.adaptive_gamma.entropy import *  # noqa: F401,F403,E402
from wfcllm.watermark.adaptive_gamma.entropy import (  # noqa: F401,E402
    ENTROPY_SCALE,
    NodeEntropyEstimator,
)
