"""Language-adapter registry (module-level singleton)."""
from __future__ import annotations

from wfcllm.common.registry import Registry
from wfcllm.lang.adapter import LanguageAdapter

_REGISTRY: Registry[LanguageAdapter] = Registry("language")


def register(name: str):
    """Decorator: register a LanguageAdapter subclass under `name`."""

    def deco(cls: type[LanguageAdapter]) -> type[LanguageAdapter]:
        _REGISTRY.register(name, cls)
        return cls

    return deco


def get(name: str) -> LanguageAdapter:
    return _REGISTRY.get(name)


def names() -> list[str]:
    return _REGISTRY.names()


def __contains__(name: str) -> bool:  # noqa: PLW3201 - module-level dunder is intentional
    return name in _REGISTRY
