from __future__ import annotations

import ast

import pytest

from wfcllm.generation.completion_finalizer import (
    finalize_humaneval_program,
    finalize_mbpp_program,
    finalize_mbpp_program_with_interface_wrapper,
)


PROMPT = 'def add(a, b):\n    """Return the sum."""\n'


def test_trims_incomplete_tests_after_complete_target() -> None:
    source = PROMPT + "    return a + b\n\nprint(add(1, 2"

    result = finalize_humaneval_program(PROMPT, source)

    assert result.applied is True
    assert result.reason == "target_function_complete"
    assert result.code == (
        'def add(a, b):\n    """Return the sum."""\n    return a + b\n'
    )
    compile(result.code, "<finalized>", "exec")


def test_trims_markdown_and_explanation_after_complete_target() -> None:
    source = (
        PROMPT
        + "    helper = lambda value: value\n"
        + "    return helper(a + b)\n"
        + "```\n这个实现返回两数之和。\n"
    )

    result = finalize_humaneval_program(PROMPT, source)

    assert result.applied is True
    assert "```" not in result.code
    assert "这个实现" not in result.code
    tree = ast.parse(result.code)
    target = tree.body[-1]
    assert isinstance(target, ast.FunctionDef)
    assert target.name == "add"
    assert any(isinstance(node, ast.Lambda) for node in ast.walk(target))


def test_removes_complete_top_level_tests_and_following_function() -> None:
    source = (
        "import math\n\n"
        + PROMPT
        + "    return a + b\n\n"
        + "print(add(1, 2))\n\n"
        + "def unrelated():\n    return math.pi\n"
    )
    prompt = "import math\n\n" + PROMPT

    result = finalize_humaneval_program(prompt, source)

    assert result.applied is True
    assert "print(" not in result.code
    assert "unrelated" not in result.code
    tree = ast.parse(result.code)
    assert [type(node) for node in tree.body] == [ast.Import, ast.FunctionDef]


def test_does_not_accept_prompt_only_stub() -> None:
    source = PROMPT + "print("

    result = finalize_humaneval_program(PROMPT, source)

    assert result.applied is False
    assert result.code == source
    assert result.reason == "no_complete_target_implementation"


def test_rejects_source_that_does_not_start_with_prompt() -> None:
    with pytest.raises(ValueError, match="start with the exact prompt"):
        finalize_humaneval_program(PROMPT, "def other():\n    return 1\n")


def test_preserves_decorated_target_and_nested_helper() -> None:
    prompt = (
        "def identity(decorator):\n"
        "    return decorator\n\n"
        "@identity\n"
        "def solve(value):\n"
        "    \"\"\"Return value plus one.\"\"\"\n"
    )
    source = (
        prompt
        + "    def helper(item):\n"
        + "        return item + 1\n"
        + "    return helper(value)\n\n"
        + "assert solve(1) == 2\n"
    )

    result = finalize_humaneval_program(prompt, source)

    assert result.applied is True
    tree = ast.parse(result.code)
    solve = tree.body[-1]
    assert isinstance(solve, ast.FunctionDef)
    assert solve.name == "solve"
    assert solve.decorator_list
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "helper"
        for node in ast.walk(solve)
    )


def test_mbpp_finalizer_keeps_target_without_exact_docstring_prefix() -> None:
    prompt = (
        "def count_items(items):\n"
        '    """Count the supplied items."""\n'
    )
    source = (
        "def count_items(items):\n"
        "    total = len(items)\n"
        "    return total\n\n"
        "print(count_items([1, 2, 3]))\n"
    )

    result = finalize_mbpp_program(prompt, source)

    assert result.applied is True
    assert result.reason == "mbpp_target_function_complete"
    assert result.code == (
        "def count_items(items):\n"
        "    total = len(items)\n"
        "    return total\n"
    )
    assert "print(" not in result.code


def test_mbpp_finalizer_rejects_wrong_interface() -> None:
    prompt = (
        "def count_items(items):\n"
        '    """Count the supplied items."""\n'
    )
    source = "def count_items(items, extra):\n    return len(items)\n"

    result = finalize_mbpp_program(prompt, source)

    assert result.applied is False
    assert result.code == source
    assert result.reason == "no_complete_mbpp_target_implementation"


def test_mbpp_interface_wrapper_preserves_calls_and_renames_recursion() -> None:
    prompt = (
        "def factorial(value):\n"
        '    """Return the factorial."""\n'
    )
    source = (
        "def factorial(value):\n"
        "    if value <= 1:\n"
        "        return 1\n"
        "    return value * factorial(value - 1)\n\n"
        "assert factorial(4) == 24\n"
    )

    result = finalize_mbpp_program_with_interface_wrapper(prompt, source)

    assert result.applied is True
    assert result.reason == "mbpp_target_function_interface_wrapped"
    tree = ast.parse(result.code)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    assert [node.name for node in functions] == [
        "_wfcllm_impl_factorial",
        "factorial",
    ]
    assert len(functions[-1].body) == 6
    namespace: dict[str, object] = {}
    exec(result.code, namespace)
    assert namespace["factorial"](5) == 120
    assert "assert " not in result.code


def test_mbpp_interface_wrapper_preserves_defaults_and_keyword_calls() -> None:
    prompt = (
        "def add(value, increment=2):\n"
        '    """Return the adjusted value."""\n'
    )
    source = "def add(value, increment=2):\n    return value + increment\n"

    result = finalize_mbpp_program_with_interface_wrapper(prompt, source)

    namespace: dict[str, object] = {}
    exec(result.code, namespace)
    assert namespace["add"](3) == 5
    assert namespace["add"](3, increment=4) == 7
