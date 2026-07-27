"""Contract tests for PythonAdapter (spec §5.1 "契约测试")."""
from __future__ import annotations


def test_python_adapter_registered():
    from wfcllm import lang
    assert "python" in lang.names()


def test_python_adapter_name_field():
    from wfcllm import lang
    assert lang.get("python").name == "python"


def test_python_adapter_statement_types_match_parser_constants():
    """LanguageAdapter.statement_types uses the parser's canonical sets."""
    from wfcllm.lang import get
    from wfcllm.lang.python.parser import (
        SIMPLE_STATEMENT_TYPES,
        COMPOUND_STATEMENT_TYPES,
    )
    types = get("python").statement_types()
    assert types.simple == SIMPLE_STATEMENT_TYPES
    assert types.compound == COMPOUND_STATEMENT_TYPES
    assert types.all == SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES


def test_python_adapter_extract_blocks_matches_parser():
    """LanguageAdapter.extract_blocks delegates to the current parser."""
    from wfcllm.lang import get
    from wfcllm.lang.python.parser import extract_statement_blocks

    source = "def f(x):\n    return x + 1\n\nclass C:\n    pass\n"
    adapted = get("python").extract_blocks(source)
    parsed = extract_statement_blocks(source)
    assert len(adapted) == len(parsed)
    for a, b in zip(adapted, parsed):
        assert (a.block_id, a.block_type, a.node_type, a.source) == (
            b.block_id, b.block_type, b.node_type, b.source,
        )


def test_unknown_language_raises_keyerror():
    from wfcllm import lang
    import pytest
    with pytest.raises(KeyError, match="unknown language"):
        lang.get("klingon")
