"""WFCLLM language-adapter layer (spec §5.1)."""
from wfcllm.lang.adapter import LanguageAdapter, StatementTypes
from wfcllm.lang.registry import get, names, register

# Importing the python subpackage triggers @register("python")
from wfcllm.lang import python  # noqa: F401, E402

__all__ = ["LanguageAdapter", "StatementTypes", "get", "names", "register"]
