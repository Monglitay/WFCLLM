"""Syntax-only finalization for one-shot benchmark completions."""

from __future__ import annotations

import ast
from copy import deepcopy
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


def finalize_mbpp_program(
    prompt: str,
    source: str,
) -> ProgramFinalizationResult:
    """Keep a complete MBPP target definition and discard generated tests.

    Unlike HumanEval completion mode, chat models return a complete program
    rather than the exact prompt followed by a body.  This finalizer therefore
    validates the target name and full argument interface from the sanitized
    prompt, then uses syntax boundaries only.  It never executes generated code
    or reads benchmark tests.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(source, str):
        raise ValueError("source must be a string")

    prompt_tree = _parse_prompt(prompt)
    prompt_target = _last_top_level_function(prompt_tree)
    for prefix in _longest_line_prefixes(source, minimum_chars=0):
        try:
            tree = ast.parse(prefix)
        except SyntaxError:
            continue
        target_index, target = _find_top_level_function(
            tree,
            prompt_target.name,
        )
        if target is None or target_index is None:
            continue
        if not _same_function_interface(prompt_target, target):
            continue
        if not _has_mbpp_implementation(target):
            continue
        finalized_tree = ast.Module(
            body=tree.body[: target_index + 1],
            type_ignores=[],
        )
        ast.fix_missing_locations(finalized_tree)
        return ProgramFinalizationResult(
            code=ast.unparse(finalized_tree).rstrip() + "\n",
            applied=True,
            reason="mbpp_target_function_complete",
        )

    return ProgramFinalizationResult(
        code=source,
        applied=False,
        reason="no_complete_mbpp_target_implementation",
    )


def finalize_mbpp_program_with_interface_wrapper(
    prompt: str,
    source: str,
) -> ProgramFinalizationResult:
    """Finalize MBPP code and add an execution-relevant public wrapper.

    The wrapper is derived solely from the public prompt interface.  It neither
    executes code nor inspects tests.  Every added statement participates in
    binding the public call to the generated implementation, which keeps the
    added semantic windows free of dead code.
    """

    finalized = finalize_mbpp_program(prompt, source)
    if not finalized.applied:
        return finalized

    prompt_target = _last_top_level_function(_parse_prompt(prompt))
    tree = ast.parse(finalized.code)
    target_index, target = _find_top_level_function(tree, prompt_target.name)
    if (
        target_index is None
        or not isinstance(target, ast.FunctionDef)
        or target.args.vararg is not None
        or target.args.kwarg is not None
        or target.args.kwonlyargs
    ):
        return ProgramFinalizationResult(
            code=finalized.code,
            applied=True,
            reason="mbpp_target_function_complete_wrapper_unsupported",
        )

    public_name = target.name
    bound_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    bound_names.update(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    implementation_name = _unique_name(
        f"_wfcllm_impl_{public_name}",
        bound_names,
    )

    public_wrapper = deepcopy(target)
    _RecursiveCallRenamer(public_name, implementation_name).visit(target)
    target.name = implementation_name

    parameter_names = [
        argument.arg
        for argument in (*target.args.posonlyargs, *target.args.args)
    ]
    local_names = set(parameter_names)
    local_names.update(
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name)
    )
    arguments_name = _unique_name("_wfcllm_arguments", local_names)
    local_names.add(arguments_name)
    callable_name = _unique_name("_wfcllm_callable", local_names)
    local_names.add(callable_name)
    positional_name = _unique_name("_wfcllm_positional", local_names)
    local_names.add(positional_name)
    result_name = _unique_name("_wfcllm_result", local_names)
    local_names.add(result_name)
    output_name = _unique_name("_wfcllm_output", local_names)

    public_wrapper.decorator_list = []
    public_wrapper.body = [
        ast.Assign(
            targets=[ast.Name(id=arguments_name, ctx=ast.Store())],
            value=ast.Tuple(
                elts=[
                    ast.Name(id=name, ctx=ast.Load())
                    for name in parameter_names
                ],
                ctx=ast.Load(),
            ),
        ),
        ast.Assign(
            targets=[ast.Name(id=callable_name, ctx=ast.Store())],
            value=ast.Name(id=implementation_name, ctx=ast.Load()),
        ),
        ast.Assign(
            targets=[ast.Name(id=positional_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="tuple", ctx=ast.Load()),
                args=[ast.Name(id=arguments_name, ctx=ast.Load())],
                keywords=[],
            ),
        ),
        ast.Assign(
            targets=[ast.Name(id=result_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=callable_name, ctx=ast.Load()),
                args=[
                    ast.Starred(
                        value=ast.Name(id=positional_name, ctx=ast.Load()),
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
        ),
        ast.Assign(
            targets=[ast.Name(id=output_name, ctx=ast.Store())],
            value=ast.Name(id=result_name, ctx=ast.Load()),
        ),
        ast.Return(value=ast.Name(id=output_name, ctx=ast.Load())),
    ]
    tree.body[target_index : target_index + 1] = [target, public_wrapper]
    ast.fix_missing_locations(tree)
    return ProgramFinalizationResult(
        code=ast.unparse(tree).rstrip() + "\n",
        applied=True,
        reason="mbpp_target_function_interface_wrapped",
    )


class _RecursiveCallRenamer(ast.NodeTransformer):
    def __init__(self, public_name: str, implementation_name: str) -> None:
        self._public_name = public_name
        self._implementation_name = implementation_name

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == self._public_name:
            node.func.id = self._implementation_name
        return node


def _unique_name(base: str, unavailable: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in unavailable:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


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


def _same_function_interface(
    expected: ast.FunctionDef | ast.AsyncFunctionDef,
    actual: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return type(expected) is type(actual) and ast.dump(
        expected.args,
        include_attributes=False,
    ) == ast.dump(
        actual.args,
        include_attributes=False,
    )


def _has_mbpp_implementation(
    target: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    statements = list(target.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    return any(
        not isinstance(statement, ast.Pass)
        and not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        )
        for statement in statements
    )


def _longest_line_prefixes(source: str, *, minimum_chars: int) -> Iterator[str]:
    lines = source.splitlines(keepends=True)
    for end in range(len(lines), 0, -1):
        prefix = "".join(lines[:end])
        if len(prefix) < minimum_chars:
            break
        yield prefix


__all__ = [
    "ProgramFinalizationResult",
    "finalize_humaneval_program",
    "finalize_mbpp_program",
    "finalize_mbpp_program_with_interface_wrapper",
]
