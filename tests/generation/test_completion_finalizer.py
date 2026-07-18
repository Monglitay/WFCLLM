from __future__ import annotations

import ast

import pytest

from wfcllm.generation.completion_finalizer import finalize_humaneval_program


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
