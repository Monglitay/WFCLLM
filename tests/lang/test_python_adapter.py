"""Contract tests for PythonAdapter (spec §5.1 "契约测试")."""
from __future__ import annotations


def test_python_adapter_registered():
    from wfcllm import lang
    assert "python" in lang.names()


def test_python_adapter_name_field():
    from wfcllm import lang
    assert lang.get("python").name == "python"


def test_python_adapter_statement_types_match_old_constants():
    """LanguageAdapter.statement_types must equal the legacy frozensets exactly."""
    from wfcllm.lang import get
    from wfcllm.lang.python.parser import (
        SIMPLE_STATEMENT_TYPES,
        COMPOUND_STATEMENT_TYPES,
    )
    types = get("python").statement_types()
    assert types.simple == SIMPLE_STATEMENT_TYPES
    assert types.compound == COMPOUND_STATEMENT_TYPES
    assert types.all == SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES


def test_python_adapter_extract_blocks_matches_legacy():
    """LanguageAdapter.extract_blocks must produce the same blocks as legacy extract_statement_blocks."""
    from wfcllm.lang import get
    from wfcllm.lang.python.parser import extract_statement_blocks

    source = "def f(x):\n    return x + 1\n\nclass C:\n    pass\n"
    new = get("python").extract_blocks(source)
    old = extract_statement_blocks(source)
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert (a.block_id, a.block_type, a.node_type, a.source) == (
            b.block_id, b.block_type, b.node_type, b.source,
        )


def test_unknown_language_raises_keyerror():
    from wfcllm import lang
    import pytest
    with pytest.raises(KeyError, match="unknown language"):
        lang.get("klingon")
