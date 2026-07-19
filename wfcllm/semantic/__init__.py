from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EmbeddingRule",
    "HashEmbeddingRule",
    "SemanticLshEmbeddingRule",
    "SemanticWindowEvidence",
    "SemanticPreservationEvidence",
    "SemanticWindowScorer",
    "load_semantic_lsh_rule",
]

_EXPORTS = {
    "EmbeddingRule": ("wfcllm.semantic.rules", "EmbeddingRule"),
    "HashEmbeddingRule": ("wfcllm.semantic.rules", "HashEmbeddingRule"),
    "SemanticLshEmbeddingRule": (
        "wfcllm.semantic.rules",
        "SemanticLshEmbeddingRule",
    ),
    "SemanticWindowEvidence": (
        "wfcllm.semantic.window_lsh",
        "SemanticWindowEvidence",
    ),
    "SemanticPreservationEvidence": (
        "wfcllm.semantic.window_lsh",
        "SemanticPreservationEvidence",
    ),
    "SemanticWindowScorer": (
        "wfcllm.semantic.window_lsh",
        "SemanticWindowScorer",
    ),
    "load_semantic_lsh_rule": ("wfcllm.semantic.lsh", "load_semantic_lsh_rule"),
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
