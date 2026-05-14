"""Deprecated: re-exports for backwards compatibility.

Moved to ``wfcllm.evaluation.detection_report`` in the Phase 2 refactor
(see docs/superpowers/specs/2026-05-14-repo-refactor-design.md §4.4, §8.2).
"""
import warnings

warnings.warn(
    "wfcllm.extract.offline_analysis is deprecated; "
    "use wfcllm.evaluation.detection_report instead.",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.evaluation.detection_report import *  # noqa: F401,F403,E402
