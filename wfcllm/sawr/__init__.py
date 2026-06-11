"""SAWR generation-time embedding smoke runner."""

from __future__ import annotations

from wfcllm.sawr.boundary import Candidate
from wfcllm.sawr.config import (
    SawrGenerationConfig,
    SawrPipelineConfig,
    SawrRuleConfig,
)
from wfcllm.sawr.generator import (
    SawrCheckpoint,
    SawrGenerateResult,
    SawrGenerator,
    SawrModelContext,
    build_chat_prompt,
    load_sawr_model,
    resolve_torch_dtype,
    strip_repeated_prompt_function,
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
)
from wfcllm.sawr.state_machine import (
    AuditEvent,
    CheckpointT,
    DecisionAction,
    SawrStateMachine,
    StateMachineDecision,
)

__all__ = [
    "AuditEvent",
    "ALLOWED_AUDIT_EVENTS",
    "Candidate",
    "CheckpointT",
    "DecisionAction",
    "EmbeddingRule",
    "FORBIDDEN_FINAL_FIELDS",
    "HashEmbeddingRule",
    "RuleDecision",
    "RuleRequest",
    "SawrCheckpoint",
    "SawrGenerateResult",
    "SawrGenerationConfig",
    "SawrGenerator",
    "SawrModelContext",
    "SawrPipeline",
    "SawrPipelineConfig",
    "SawrRuleConfig",
    "SawrStateMachine",
    "StateMachineDecision",
    "build_chat_prompt",
    "load_sawr_model",
    "resolve_torch_dtype",
    "strip_repeated_prompt_function",
]
