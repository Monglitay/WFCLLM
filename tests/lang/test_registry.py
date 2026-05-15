"""Behaviours of the wfcllm.lang module-level registry."""
from __future__ import annotations

import pytest


def test_register_then_get_returns_instance():
    from wfcllm import lang
    instance = lang.get("python")
    assert instance.name == "python"
    # get() must return a fresh instance each call (per Registry.get contract).
    instance2 = lang.get("python")
    assert instance is not instance2


def test_get_unknown_raises_keyerror_with_helpful_message():
    from wfcllm import lang
    with pytest.raises(KeyError) as exc:
        lang.get("does-not-exist")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    assert "python" in msg  # registered names listed


def test_duplicate_register_raises():
    from wfcllm.lang import register
    from wfcllm.lang.adapter import LanguageAdapter, StatementTypes

    class _Dup(LanguageAdapter):
        name = "python"
        def statement_types(self): return StatementTypes(frozenset(), frozenset())
        def extract_blocks(self, source): return []
        def positive_rules(self): return []
        def negative_rules(self): return []

    with pytest.raises(ValueError, match="already registered"):
        register("python")(_Dup)


def test_names_includes_python():
    from wfcllm import lang
    assert "python" in lang.names()
