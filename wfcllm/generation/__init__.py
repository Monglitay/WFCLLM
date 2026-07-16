from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AuditEvent",
    "BoundaryEvent",
    "BoundaryEventKind",
    "Candidate",
    "CausalWindowRewriter",
    "GatedGenerationResult",
    "GatedGenerationPipeline",
    "GatedGenerationPipelineConfig",
    "GatedGenerator",
    "GatedWindowAudit",
    "ParsedRewrite",
    "RewriteGeneration",
    "RewriteTokens",
    "SawrStateMachine",
    "WFCLLMStateMachine",
    "evidence_retry_key",
]

_EXPORTS = {
    "AuditEvent": ("wfcllm.generation.state_machine", "AuditEvent"),
    "BoundaryEvent": ("wfcllm.generation.boundary", "BoundaryEvent"),
    "BoundaryEventKind": ("wfcllm.generation.boundary", "BoundaryEventKind"),
    "Candidate": ("wfcllm.generation.boundary", "Candidate"),
    "CausalWindowRewriter": (
        "wfcllm.generation.window_rewriter",
        "CausalWindowRewriter",
    ),
    "GatedGenerationResult": (
        "wfcllm.generation.gated_generator",
        "GatedGenerationResult",
    ),
    "GatedGenerationPipeline": (
        "wfcllm.generation.gated_pipeline",
        "GatedGenerationPipeline",
    ),
    "GatedGenerationPipelineConfig": (
        "wfcllm.generation.gated_pipeline",
        "GatedGenerationPipelineConfig",
    ),
    "GatedGenerator": ("wfcllm.generation.gated_generator", "GatedGenerator"),
    "GatedWindowAudit": (
        "wfcllm.generation.gated_generator",
        "GatedWindowAudit",
    ),
    "ParsedRewrite": ("wfcllm.generation.window_rewriter", "ParsedRewrite"),
    "RewriteGeneration": (
        "wfcllm.generation.window_rewriter",
        "RewriteGeneration",
    ),
    "RewriteTokens": ("wfcllm.generation.gated_generator", "RewriteTokens"),
    "SawrStateMachine": ("wfcllm.generation.state_machine", "SawrStateMachine"),
    "WFCLLMStateMachine": ("wfcllm.generation.state_machine", "SawrStateMachine"),
    "evidence_retry_key": ("wfcllm.generation.retry", "evidence_retry_key"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
