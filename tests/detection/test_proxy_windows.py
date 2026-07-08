from __future__ import annotations

import pytest

import wfcllm.detection as detect
from wfcllm.detection.proxy_windows import (
    ProxyWindow,
    StructureContext,
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


def test_select_target_function_without_prompt_prefers_last_top_level_function() -> None:
    code = (
        "def helper():\n"
        "    return 0\n\n"
        "def target(x):\n"
        "    return x\n"
    )

    assert select_target_function_name(code, prompt=None) == "target"


def test_extracts_decorated_prompt_target_function() -> None:
    code = (
        "def helper():\n"
        "    return 0\n\n"
        "@cache\n"
        "def target(x):\n"
        "    y = x\n"
        "    return y\n"
    )

    contexts = extract_structure_contexts(
        code,
        prompt="def target(x):\n",
        max_group_statements=2,
    )

    assert select_target_function_name(code, prompt="def target(x):\n") == "target"
    assert [context.context_id for context in contexts] == ["module.target.body"]
    assert [window.normalized_text for window in contexts[0].proxy_windows] == [
        "y = x",
        "return y",
        "y = x\nreturn y",
    ]


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
        "if x:\n        b = 1\n        c = 2",
    ]


def test_skips_empty_function_body_context_when_child_context_is_scoreable() -> None:
    code = (
        "def target(x):\n"
        "    if x:\n"
        "        y = 1\n"
        "        return y\n"
    )

    contexts = extract_structure_contexts(
        code,
        prompt="def target(x):\n",
        max_group_statements=2,
    )

    assert [context.structure_type for context in contexts] == ["if_statement"]
    assert [window.normalized_text for window in contexts[0].proxy_windows] == [
        "y = 1",
        "return y",
        "y = 1\nreturn y",
        "if x:\n        y = 1\n        return y",
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


def test_extracts_compound_layer_proxy_windows_for_nested_structures() -> None:
    code = (
        "def target(numbers, threshold):\n"
        "    for i in range(len(numbers) - 1):\n"
        "        if abs(numbers[i] - numbers[i + 1]) < threshold:\n"
        "            return True\n"
        "    return False\n"
    )

    windows = extract_proxy_windows(
        code,
        prompt="def target(numbers, threshold):\n",
        max_group_statements=2,
    )

    compound_windows = [
        window
        for window in windows
        if window.candidates[0].candidate_type == "compound_layer_proxy_window"
    ]

    assert [window.normalized_text for window in compound_windows] == [
        (
            "for i in range(len(numbers) - 1):\n"
            "        if abs(numbers[i] - numbers[i + 1]) < threshold:\n"
            "            return True"
        ),
        "if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True",
    ]
    assert [window.parent_node_type for window in compound_windows] == [
        "function_definition",
        "for_statement",
    ]
    assert [window.structure_type for window in compound_windows] == [
        "for_statement",
        "if_statement",
    ]
    assert [window.candidates[0].node_type for window in compound_windows] == [
        "for_statement",
        "if_statement",
    ]


def test_detector_package_exports_proxy_window_api() -> None:
    assert detect.extract_proxy_windows is extract_proxy_windows
    assert detect.StructureContext is StructureContext
    assert detect.ProxyWindow is ProxyWindow


def test_two_statement_proxy_window_preserves_candidate_metadata() -> None:
    code = "def target():\n    x = 1\n    y = 2\n"

    window = next(
        window
        for window in extract_proxy_windows(
            code,
            prompt="def target():\n",
            max_group_statements=2,
        )
        if window.window_length == 2
    )

    first_candidate, second_candidate = window.candidates
    first_start = code.index("x = 1")
    second_start = code.index("y = 2")

    assert [candidate.text for candidate in window.candidates] == ["x = 1", "y = 2"]
    assert [candidate.candidate_type for candidate in window.candidates] == [
        "proxy_window_statement",
        "proxy_window_statement",
    ]
    assert [candidate.position_id for candidate in window.candidates] == [
        window.context_id,
        window.context_id,
    ]
    assert [candidate.layer_path for candidate in window.candidates] == [
        (window.context_id,),
        (window.context_id,),
    ]
    assert [candidate.ordinal for candidate in window.candidates] == [0, 1]
    assert (first_candidate.start_byte, first_candidate.end_byte) == (
        first_start,
        first_start + len("x = 1"),
    )
    assert (second_candidate.start_byte, second_candidate.end_byte) == (
        second_start,
        second_start + len("y = 2"),
    )


def test_no_top_level_function_returns_no_contexts() -> None:
    assert extract_structure_contexts("x = 1\n", prompt=None, max_group_statements=2) == []


def test_rejects_invalid_max_group_statements() -> None:
    with pytest.raises(ValueError, match="max_group_statements must be positive"):
        extract_proxy_windows("def f():\n    return 1\n", max_group_statements=0)
