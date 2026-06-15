from __future__ import annotations

import pytest

from wfcllm.sawr.detect.proxy_windows import (
    extract_proxy_windows,
    extract_structure_contexts,
    select_target_function_name,
)


def test_select_target_function_prefers_prompt_name() -> None:
    prompt = "def target(x):\n"
    code = (
        "def helper():\n"
        "    return 0\n\n"
        "def target(x):\n"
        "    return x\n"
    )

    assert select_target_function_name(code, prompt=prompt) == "target"


def test_extracts_direct_windows_without_child_layer_statements() -> None:
    code = (
        "def target(x):\n"
        "    a = 0\n"
        "    if x:\n"
        "        b = 1\n"
        "        c = 2\n"
        "    return a\n"
    )

    contexts = extract_structure_contexts(
        code,
        prompt="def target(x):\n",
        max_group_statements=2,
    )

    function_context = next(
        context for context in contexts if context.structure_type == "function_body"
    )
    if_context = next(
        context for context in contexts if context.structure_type == "if_statement"
    )

    assert [statement.normalized_text for statement in function_context.direct_statements] == [
        "a = 0",
        "return a",
    ]
    assert [window.normalized_text for window in function_context.proxy_windows] == [
        "a = 0",
        "return a",
        "a = 0\nreturn a",
    ]
    assert [statement.normalized_text for statement in if_context.direct_statements] == [
        "b = 1",
        "c = 2",
    ]
    assert [window.normalized_text for window in if_context.proxy_windows] == [
        "b = 1",
        "c = 2",
        "b = 1\nc = 2",
    ]


def test_extract_proxy_windows_flattens_context_windows() -> None:
    code = "def target():\n    x = 1\n    y = 2\n"

    windows = extract_proxy_windows(
        code,
        prompt="def target():\n",
        max_group_statements=2,
    )

    assert [window.normalized_text for window in windows] == [
        "x = 1",
        "y = 2",
        "x = 1\ny = 2",
    ]
    assert all(window.parent_node_type == "function_definition" for window in windows)
    assert {window.window_length for window in windows} == {1, 2}


def test_no_top_level_function_returns_no_contexts() -> None:
    assert extract_structure_contexts("x = 1\n", prompt=None, max_group_statements=2) == []


def test_rejects_invalid_max_group_statements() -> None:
    with pytest.raises(ValueError, match="max_group_statements must be positive"):
        extract_proxy_windows("def f():\n    return 1\n", max_group_statements=0)
