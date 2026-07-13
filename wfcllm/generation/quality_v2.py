from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class StaticQualityAssessment:
    eligible: bool
    quality_tier: int
    syntax_valid: bool
    target_function_name: str | None
    target_function_count: int
    signature_compatible: bool
    executable_body: bool
    has_markdown_fence: bool
    suspicious_tail: bool
    code_chars: int


def assess_static_quality(
    *,
    prompt: str,
    final_code: str,
) -> StaticQualityAssessment:
    if not isinstance(prompt, str) or not isinstance(final_code, str):
        raise ValueError("prompt and final_code must be strings")
    prompt_function = _last_top_level_function(_parse(prompt))
    target_name = prompt_function.name if prompt_function is not None else None
    has_markdown_fence = "```" in final_code
    final_tree = _parse(final_code)
    syntax_valid = final_tree is not None
    target_functions = (
        _top_level_functions_by_name(final_tree, target_name)
        if final_tree is not None and target_name is not None
        else []
    )
    target_function_count = len(target_functions)
    final_function = target_functions[0] if target_function_count == 1 else None
    signature_compatible = (
        prompt_function is not None
        and final_function is not None
        and _function_signature(prompt_function) == _function_signature(final_function)
    )
    executable_body = (
        _has_executable_body(final_function)
        if final_function is not None
        else False
    )
    suspicious_tail = _has_suspicious_tail(final_code, syntax_valid=syntax_valid)
    eligible = (
        syntax_valid
        and target_function_count == 1
        and signature_compatible
        and executable_body
        and not has_markdown_fence
        and not suspicious_tail
    )
    quality_tier = 2 if eligible else 1 if (
        syntax_valid
        and target_function_count == 1
        and signature_compatible
        and not has_markdown_fence
    ) else 0
    return StaticQualityAssessment(
        eligible=eligible,
        quality_tier=quality_tier,
        syntax_valid=syntax_valid,
        target_function_name=target_name,
        target_function_count=target_function_count,
        signature_compatible=signature_compatible,
        executable_body=executable_body,
        has_markdown_fence=has_markdown_fence,
        suspicious_tail=suspicious_tail,
        code_chars=len(final_code),
    )


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None


def _last_top_level_function(
    tree: ast.Module | None,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if tree is None:
        return None
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return functions[-1] if functions else None


def _top_level_functions_by_name(
    tree: ast.Module,
    name: str,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ast.dump(
        ast.Module(
            body=[
                ast.FunctionDef(
                    name="target",
                    args=node.args,
                    body=[ast.Pass()],
                    decorator_list=[],
                    returns=node.returns,
                    type_comment=node.type_comment,
                )
            ],
            type_ignores=[],
        ),
        include_attributes=False,
    )


def _has_executable_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    return any(not _is_placeholder(statement) for statement in body)


def _is_placeholder(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _has_suspicious_tail(final_code: str, *, syntax_valid: bool) -> bool:
    if not syntax_valid:
        return True
    tail = final_code.rstrip()
    return not tail or tail.endswith(("\\", "```"))
