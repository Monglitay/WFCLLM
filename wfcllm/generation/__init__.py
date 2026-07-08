from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AuditEvent",
    "BoundaryEvent",
    "BoundaryEventKind",
    "Candidate",
    "SawrStateMachine",
    "WFCLLMStateMachine",
    "evidence_retry_key",
]

_EXPORTS = {
    "AuditEvent": ("wfcllm.generation.state_machine", "AuditEvent"),
    "BoundaryEvent": ("wfcllm.generation.boundary", "BoundaryEvent"),
    "BoundaryEventKind": ("wfcllm.generation.boundary", "BoundaryEventKind"),
    "Candidate": ("wfcllm.generation.boundary", "Candidate"),
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
