"""Deprecated: re-exports for backwards compatibility.

Moved to ``wfcllm.evaluation.code_execution`` in the Phase 2 refactor
(see docs/superpowers/specs/2026-05-14-repo-refactor-design.md §4.4, §8.2).
"""
import warnings

warnings.warn(
    "wfcllm.common.offline_code_eval is deprecated; "
    "use wfcllm.evaluation.code_execution instead.",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.evaluation.code_execution import *  # noqa: F401,F403,E402
