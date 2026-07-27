"""Leakage-safe MBPP generation interface extraction."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


_ASSERT_WRAPPER_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
    }
)
_HELPER_CLASS_ATTRIBUTES = {"Pair": ("a", "b")}


@dataclass(frozen=True)
class MBPPInterfaceSpec:
    """Only the callable shape needed to invoke one MBPP solution."""

    function_name: str
    positional_arities: tuple[int, ...]
    keyword_names: tuple[str, ...]
    helper_classes: tuple[str, ...]


def extract_mbpp_interface(
    test_list: Sequence[str],
) -> MBPPInterfaceSpec | None:
    """Extract call shape without retaining assertions, arguments, or outputs."""
    primary_calls: list[ast.Call] = []
    all_named_calls: list[ast.Call] = []
    for test in test_list:
        if not isinstance(test, str):
            return None
        try:
            tree = ast.parse(test)
        except SyntaxError:
            return None
        all_named_calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
        for statement in tree.body:
            if isinstance(statement, ast.Assert):
                primary_calls.extend(_primary_calls(statement.test))

    name_counts = Counter(
        call.func.id
        for call in primary_calls
        if isinstance(call.func, ast.Name)
    )
    if not name_counts:
        return None
    most_common_count = max(name_counts.values())
    function_names = sorted(
        name for name, count in name_counts.items() if count == most_common_count
    )
    if len(function_names) != 1:
        return None

    function_name = function_names[0]
    target_calls = [
        call
        for call in primary_calls
        if isinstance(call.func, ast.Name) and call.func.id == function_name
    ]
    positional_arities = tuple(sorted({len(call.args) for call in target_calls}))
    if not positional_arities:
        return None
    keyword_names = tuple(
        sorted(
            {
                keyword.arg
                for call in target_calls
                for keyword in call.keywords
                if keyword.arg is not None
            }
        )
    )
    helper_classes = tuple(
        sorted(
            {
                call.func.id
                for call in all_named_calls
                if isinstance(call.func, ast.Name)
                and call.func.id != function_name
                and call.func.id[:1].isupper()
            }
        )
    )
    return MBPPInterfaceSpec(
        function_name=function_name,
        positional_arities=positional_arities,
        keyword_names=keyword_names,
        helper_classes=helper_classes,
    )


def build_mbpp_generation_prompt(
    task_text: str,
    test_list: Sequence[str],
) -> tuple[str, MBPPInterfaceSpec | None]:
    """Build a HumanEval-style code prefix from a sanitized call shape."""
    interface = extract_mbpp_interface(test_list)
    if interface is None:
        return task_text, None

    lines: list[str] = []
    for helper_class in interface.helper_classes:
        attributes = _HELPER_CLASS_ATTRIBUTES.get(helper_class)
        if attributes:
            parameters = ", ".join(attributes)
            lines.extend(
                [
                    f"class {helper_class}:",
                    f"    def __init__(self, {parameters}):",
                    *(
                        f"        self.{attribute} = {attribute}"
                        for attribute in attributes
                    ),
                    "",
                ]
            )
        else:
            lines.extend([f"class {helper_class}:", "    pass", ""])
    lines.append(_signature(interface))
    escaped_task = task_text.strip().replace('"""', r"\"\"\"")
    task_lines = escaped_task.splitlines() or [escaped_task]
    lines.append(f'    """{task_lines[0]}')
    lines.extend(f"    {line}" for line in task_lines[1:])
    lines.extend(
        [
            "    Return the result instead of printing it.",
            '    """',
        ]
    )
    return "\n".join(lines).rstrip() + "\n", interface


def _signature(interface: MBPPInterfaceSpec) -> str:
    max_arity = max(interface.positional_arities)
    parameters = [f"arg{index}" for index in range(1, max_arity + 1)]
    parameters.extend(f"{name}=None" for name in interface.keyword_names)
    return f"def {interface.function_name}({', '.join(parameters)}):"


def _primary_calls(node: ast.AST) -> list[ast.Call]:
    if isinstance(node, ast.Compare):
        return _primary_calls(node.left)
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id not in _ASSERT_WRAPPER_NAMES
        ):
            return [node]
        calls: list[ast.Call] = []
        for argument in node.args:
            calls.extend(_primary_calls(argument))
        for keyword in node.keywords:
            calls.extend(_primary_calls(keyword.value))
        return calls
    if isinstance(node, (ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Attribute, ast.Subscript)):
        calls = []
        for child in ast.iter_child_nodes(node):
            calls.extend(_primary_calls(child))
        return calls
    return []
