from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wfcllm.lang.python.parser import PythonParser, SIMPLE_STATEMENT_TYPES
from wfcllm.sawr.boundary import Candidate
from wfcllm.sawr.rules import _normalize_candidate_text

SCOREABLE_COMPOUND_TYPES = frozenset({"if_statement", "for_statement", "while_statement"})


@dataclass(frozen=True)
class DirectStatement:
    normalized_text: str
    node_type: str
    parent_node_type: str
    ordinal: int
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class ProxyWindow:
    context_id: str
    window_id: str
    normalized_text: str
    candidates: tuple[Candidate, ...]
    parent_node_type: str
    structure_type: str
    window_length: int
    context_statement_count: int
    context_window_count: int
    ordinal: int | None
    start_line: int
    end_line: int


@dataclass(frozen=True)
class StructureContext:
    context_id: str
    structure_type: str
    parent_node_type: str
    direct_statements: tuple[DirectStatement, ...]
    proxy_windows: tuple[ProxyWindow, ...]
    start_line: int
    end_line: int


def select_target_function_name(final_code: str, prompt: str | None = None) -> str | None:
    prompt_name = _last_top_level_function_name(prompt) if prompt else None
    functions = _top_level_functions(final_code)
    if not functions:
        return None

    names = [name for function in functions if (name := _function_name(function))]
    if not names:
        return None
    if prompt_name in names:
        return prompt_name
    return names[0]


def extract_structure_contexts(
    final_code: str,
    *,
    prompt: str | None = None,
    max_group_statements: int = 2,
) -> list[StructureContext]:
    if max_group_statements <= 0:
        raise ValueError("max_group_statements must be positive")

    target_name = select_target_function_name(final_code, prompt=prompt)
    if target_name is None:
        return []

    functions = _top_level_functions(final_code)
    target_function = next(
        (
            function
            for function in functions
            if _function_name(function) == target_name
        ),
        None,
    )
    if target_function is None:
        return []

    source_bytes = final_code.encode("utf-8")
    # Detector ordinals are deterministic proxy-local statement ordinals, not
    # generation-time event ordinals.
    ordinal_counter = [0]
    contexts: list[StructureContext] = []

    function_context = _build_context(
        node=target_function,
        source_bytes=source_bytes,
        context_id=f"module.{target_name}.body",
        structure_type="function_body",
        parent_node_type="function_definition",
        max_group_statements=max_group_statements,
        ordinal_counter=ordinal_counter,
    )
    if function_context is not None:
        contexts.append(function_context)

    compound_index = 0
    for compound_node in _walk_scoreable_compounds(target_function):
        compound_context = _build_context(
            node=compound_node,
            source_bytes=source_bytes,
            context_id=f"module.{target_name}.{compound_node.type}.{compound_index}",
            structure_type=compound_node.type,
            parent_node_type=compound_node.type,
            max_group_statements=max_group_statements,
            ordinal_counter=ordinal_counter,
        )
        compound_index += 1
        if compound_context is not None:
            contexts.append(compound_context)

    return contexts


def extract_proxy_windows(
    final_code: str,
    *,
    prompt: str | None = None,
    max_group_statements: int = 2,
) -> list[ProxyWindow]:
    contexts = extract_structure_contexts(
        final_code,
        prompt=prompt,
        max_group_statements=max_group_statements,
    )
    return [
        window
        for context in contexts
        for window in context.proxy_windows
    ]


def _build_context(
    *,
    node: Any,
    source_bytes: bytes,
    context_id: str,
    structure_type: str,
    parent_node_type: str,
    max_group_statements: int,
    ordinal_counter: list[int],
) -> StructureContext | None:
    if _node_has_recovery_content(node):
        return None

    direct_statements: list[DirectStatement] = []
    for block in _owned_blocks(node):
        for child in block.children:
            if child.type not in SIMPLE_STATEMENT_TYPES:
                continue
            if _node_has_recovery_content(child):
                continue

            normalized_text = _normalize_candidate_text(
                source_bytes[child.start_byte:child.end_byte].decode("utf-8")
            )
            if not normalized_text or "\n" in normalized_text:
                continue

            direct_statements.append(
                DirectStatement(
                    normalized_text=normalized_text,
                    node_type=child.type,
                    parent_node_type=parent_node_type,
                    ordinal=ordinal_counter[0],
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    start_byte=child.start_byte,
                    end_byte=child.end_byte,
                )
            )
            ordinal_counter[0] += 1

    proxy_windows = _build_windows(
        context_id=context_id,
        direct_statements=tuple(direct_statements),
        parent_node_type=parent_node_type,
        structure_type=structure_type,
        max_group_statements=max_group_statements,
    )
    if not direct_statements or not proxy_windows:
        return None

    return StructureContext(
        context_id=context_id,
        structure_type=structure_type,
        parent_node_type=parent_node_type,
        direct_statements=tuple(direct_statements),
        proxy_windows=tuple(proxy_windows),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
    )


def _build_windows(
    *,
    context_id: str,
    direct_statements: tuple[DirectStatement, ...],
    parent_node_type: str,
    structure_type: str,
    max_group_statements: int,
) -> list[ProxyWindow]:
    window_statement_groups: list[tuple[DirectStatement, ...]] = []
    statement_count = len(direct_statements)
    for window_length in range(1, min(max_group_statements, statement_count) + 1):
        for start_index in range(0, statement_count - window_length + 1):
            window_statement_groups.append(
                direct_statements[start_index:start_index + window_length]
            )

    window_count = len(window_statement_groups)
    windows: list[ProxyWindow] = []
    for window_index, statements in enumerate(window_statement_groups):
        candidates = tuple(
            Candidate(
                text=statement.normalized_text,
                candidate_type="proxy_window_statement",
                node_type=statement.node_type,
                position_id=context_id,
                token_start_idx=0,
                token_count=0,
                parent_node_type=statement.parent_node_type,
                ordinal=statement.ordinal,
                layer_path=(context_id,),
                start_byte=statement.start_byte,
                end_byte=statement.end_byte,
                depth=0,
            )
            for statement in statements
        )
        windows.append(
            ProxyWindow(
                context_id=context_id,
                window_id=f"{context_id}.window.{window_index}",
                normalized_text="\n".join(
                    statement.normalized_text for statement in statements
                ),
                candidates=candidates,
                parent_node_type=parent_node_type,
                structure_type=structure_type,
                window_length=len(statements),
                context_statement_count=statement_count,
                context_window_count=window_count,
                ordinal=statements[0].ordinal,
                start_line=statements[0].start_line,
                end_line=statements[-1].end_line,
            )
        )
    return windows


def _top_level_functions(source: str) -> list[Any]:
    tree = PythonParser().parse(source)
    return [
        function_node
        for child in tree.root_node.children
        if (function_node := _top_level_function_node(child)) is not None
    ]


def _last_top_level_function_name(source: str) -> str | None:
    names = [
        name
        for function in _top_level_functions(source)
        if (name := _function_name(function))
    ]
    return names[-1] if names else None


def _top_level_function_node(node: Any) -> Any | None:
    if _node_has_recovery_content(node):
        return None
    if node.type == "function_definition":
        return node
    if node.type != "decorated_definition":
        return None

    for child in node.children:
        if child.type == "function_definition" and not _node_has_recovery_content(child):
            return child
    return None


def _function_name(function_node: Any) -> str | None:
    name_node = function_node.child_by_field_name("name")
    if name_node is None or name_node.type == "ERROR" or name_node.is_missing:
        return None
    return name_node.text.decode("utf-8")


def _owned_blocks(node: Any) -> list[Any]:
    blocks: list[Any] = []
    for child in node.children:
        if child.type == "block" and not _node_has_recovery_content(child):
            blocks.append(child)
        elif child.type in {"elif_clause", "else_clause"}:
            blocks.extend(
                grandchild
                for grandchild in child.children
                if grandchild.type == "block"
                and not _node_has_recovery_content(grandchild)
            )
    return blocks


def _walk_scoreable_compounds(node: Any) -> list[Any]:
    compounds: list[Any] = []
    for child in node.children:
        if child.type in {"function_definition", "class_definition"}:
            continue
        if child.type in SCOREABLE_COMPOUND_TYPES:
            if not _node_has_recovery_content(child):
                compounds.append(child)
                compounds.extend(_walk_scoreable_compounds(child))
            continue
        compounds.extend(_walk_scoreable_compounds(child))
    return compounds


def _node_has_recovery_content(node: Any) -> bool:
    return node.type == "ERROR" or node.is_missing or node.has_error
