"""Shared statement-window contracts, partitioning, and extractors."""

from wfcllm.windowing.contracts import (
    WINDOW_CONTRACT_VERSION,
    WINDOW_CONTRACT_VERSIONS,
    GateScores,
    ParentDescriptor,
    StatementUnit,
    is_supported_window_contract,
    language_for_window_contract,
    window_contract_for_language,
)
from wfcllm.windowing.multilanguage import (
    TreeSitterStatementUnitExtractor,
    get_statement_unit_extractor,
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
    "WINDOW_CONTRACT_VERSIONS",
    "ParentDescriptor",
    "PartitionResult",
    "PythonStatementUnitExtractor",
    "TreeSitterStatementUnitExtractor",
    "SemanticWindow",
    "SkipReason",
    "SkippedContext",
    "StatementUnit",
    "WindowPartitioner",
    "get_statement_unit_extractor",
    "is_supported_window_contract",
    "language_for_window_contract",
    "window_contract_for_language",
]
