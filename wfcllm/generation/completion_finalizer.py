"""Syntax-only finalization for one-shot benchmark completions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ProgramFinalizationResult:
    """A finalized program or an explicit fallback to the model output."""

    code: str
    applied: bool
    reason: str


def finalize_humaneval_program(
    prompt: str,
    source: str,
) -> ProgramFinalizationResult:
    """Keep the longest complete HumanEval target function without tests.

    The operation uses only the public prompt, Python syntax, and source
    boundaries.  It does not execute code or inspect benchmark tests.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(source, str):
        raise ValueError("source must be a string")
    if not source.startswith(prompt):
        raise ValueError("generated source must start with the exact prompt")

    prompt_tree = _parse_prompt(prompt)
    prompt_target = _last_top_level_function(prompt_tree)
    target_name = prompt_target.name

    for prefix in _longest_line_prefixes(source, minimum_chars=len(prompt)):
        try:
            tree = ast.parse(prefix)
        except SyntaxError:
            continue
        target_index, target = _find_top_level_function(tree, target_name)
        if target is None or target_index is None:
            continue
        if not _has_completion_statement(target, prompt_target):
            continue
        finalized_tree = ast.Module(
            body=tree.body[: target_index + 1],
            type_ignores=[],
        )
        ast.fix_missing_locations(finalized_tree)
        code = ast.unparse(finalized_tree).rstrip() + "\n"
        return ProgramFinalizationResult(
            code=code,
            applied=True,
            reason="target_function_complete",
        )

    return ProgramFinalizationResult(
        code=source,
        applied=False,
        reason="no_complete_target_implementation",
    )


def _parse_prompt(prompt: str) -> ast.Module:
    try:
        return ast.parse(prompt)
    except SyntaxError as exc:
        raise ValueError("HumanEval prompt must be valid Python") from exc


def _last_top_level_function(
    tree: ast.Module,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise ValueError("HumanEval prompt must define a target function")
    return functions[-1]


def _find_top_level_function(
    tree: ast.Module,
    name: str,
) -> tuple[int | None, ast.FunctionDef | ast.AsyncFunctionDef | None]:
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return index, node
    return None, None


def _has_completion_statement(
    target: ast.FunctionDef | ast.AsyncFunctionDef,
    prompt_target: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    prompt_end_line = prompt_target.end_lineno or prompt_target.lineno
    return any(
        (statement.lineno or 0) > prompt_end_line
        for statement in target.body
    )


def _longest_line_prefixes(source: str, *, minimum_chars: int) -> Iterator[str]:
    lines = source.splitlines(keepends=True)
    for end in range(len(lines), 0, -1):
        prefix = "".join(lines[:end])
        if len(prefix) < minimum_chars:
            break
        yield prefix


__all__ = ["ProgramFinalizationResult", "finalize_humaneval_program"]
