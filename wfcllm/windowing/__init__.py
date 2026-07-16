"""Shared statement-window contracts, partitioning, and extractors."""

from wfcllm.windowing.contracts import (
    WINDOW_CONTRACT_VERSION,
    GateScores,
    ParentDescriptor,
    StatementUnit,
)
from wfcllm.windowing.partitioner import (
    CloseReason,
    GateDecision,
    GatePredictor,
    GateThresholds,
    PartitionResult,
    SemanticWindow,
    SkipReason,
    SkippedContext,
    WindowPartitioner,
)
from wfcllm.windowing.python import PythonStatementUnitExtractor

__all__ = [
    "CloseReason",
    "GateDecision",
    "GatePredictor",
    "GateScores",
    "GateThresholds",
    "WINDOW_CONTRACT_VERSION",
    "ParentDescriptor",
    "PartitionResult",
    "PythonStatementUnitExtractor",
    "SemanticWindow",
    "SkipReason",
    "SkippedContext",
    "StatementUnit",
    "WindowPartitioner",
]
