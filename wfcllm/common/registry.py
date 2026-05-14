"""Generic typed registry used by LanguageAdapter / DatasetAdapter and other plug-in points."""
from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → factory class registry.

    Usage:
        registry: Registry[LanguageAdapter] = Registry("language")
        registry.register("python", PythonAdapter)
        adapter = registry.get("python")    # instantiates PythonAdapter()
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str, cls: type[T]) -> None:
        if name in self._entries:
            raise ValueError(
                f"{self._label} registry: '{name}' already registered "
                f"(existing={self._entries[name].__name__}, new={cls.__name__})"
            )
        self._entries[name] = cls

    def get(self, name: str) -> T:
        if name not in self._entries:
            raise KeyError(
                f"unknown {self._label}: '{name}'. "
                f"registered names: {self.names()}"
            )
        return self._entries[name]()

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
