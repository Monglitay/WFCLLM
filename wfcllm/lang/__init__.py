"""WFCLLM language-adapter layer (spec §5.1)."""
from wfcllm.lang.adapter import LanguageAdapter, StatementTypes
from wfcllm.lang.registry import get, names, register

# Importing language subpackages triggers their @register decorators.
from wfcllm.lang import cpp, java, js, python  # noqa: F401, E402

__all__ = ["LanguageAdapter", "StatementTypes", "get", "names", "register"]
