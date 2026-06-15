"""SAWR generation-time embedding smoke runner."""

from __future__ import annotations

from wfcllm.sawr.boundary import BoundaryEvent, BoundaryEventKind, Candidate
from wfcllm.sawr.config import (
    DEFAULT_HUMANEVAL_STOP_SEQUENCES,
    SawrGenerationConfig,
    SawrPipelineConfig,
    SawrRuleConfig,
)
from wfcllm.sawr.detect import DETECTOR_MODE, BucketEdges, SawrDetectionConfig
from wfcllm.sawr.generator import (
    SawrCheckpoint,
    SawrGenerateResult,
    SawrGenerator,
    SawrModelContext,
    build_chat_prompt,
    build_generation_prompt,
    load_sawr_model,
    resolve_torch_dtype,
    strip_repeated_prompt_function,
    truncate_at_stop_sequences,
)
from wfcllm.sawr.pipeline import (
    ALLOWED_AUDIT_EVENTS,
    FORBIDDEN_FINAL_FIELDS,
    SawrPipeline,
)
from wfcllm.sawr.rules import (
    EmbeddingRule,
    HashEmbeddingRule,
    RuleDecision,
    RuleRequest,
    SemanticLshEmbeddingRule,
)
from wfcllm.sawr.state_machine import (
    AuditEvent,
    CheckpointT,
    DecisionAction,
    LayerFrame,
    SawrStateMachine,
    StateMachineDecision,
    StateMachineSnapshot,
)

__all__ = [
    "AuditEvent",
    "ALLOWED_AUDIT_EVENTS",
    "BoundaryEvent",
    "BoundaryEventKind",
    "Candidate",
    "CheckpointT",
    "DEFAULT_HUMANEVAL_STOP_SEQUENCES",
    "DETECTOR_MODE",
    "DecisionAction",
    "EmbeddingRule",
    "FORBIDDEN_FINAL_FIELDS",
    "HashEmbeddingRule",
    "LayerFrame",
    "RuleDecision",
    "RuleRequest",
    "SemanticLshEmbeddingRule",
    "BucketEdges",
    "SawrCheckpoint",
    "SawrDetectionConfig",
    "SawrGenerateResult",
    "SawrGenerationConfig",
    "SawrGenerator",
    "SawrModelContext",
    "SawrPipeline",
    "SawrPipelineConfig",
    "SawrRuleConfig",
    "SawrStateMachine",
    "StateMachineDecision",
    "StateMachineSnapshot",
    "build_chat_prompt",
    "build_generation_prompt",
    "load_sawr_model",
    "resolve_torch_dtype",
    "strip_repeated_prompt_function",
    "truncate_at_stop_sequences",
]
