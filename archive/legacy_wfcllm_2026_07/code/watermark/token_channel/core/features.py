"""Structural features for token-channel scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ast
from typing import Any

FEATURE_VERSION = "token-channel-features/v1"
PYTHON_LANGUAGE = "python"
FORMAL_EXCLUSION_NODE_TYPES = frozenset(
    {
        "import_statement",
        "import_from_statement",
        "decorator",
        "function_signature",
        "class_header",
    }
)


@dataclass(frozen=True)
class ExcludedSpan:
    """Source span that should not participate in lexical switching."""

    label: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenChannelFeatureContext:
    """Precomputed source state reused across token rows."""

    source_code: str
    tree: ast.AST
    line_starts: list[int]
    parent_map: dict[ast.AST, ast.AST]
    structure_masks: list[bool]


@dataclass(frozen=True)
class TokenChannelFeatures:
    """Stable structural features passed into the token-channel model."""

    node_type: str
    parent_node_type: str
    block_relative_offset: int
    in_code_body: bool
    structure_mask: bool
    language: str = PYTHON_LANGUAGE

    def __post_init__(self) -> None:
        _coerce_string(self.node_type, "node_type")
        _coerce_string(self.parent_node_type, "parent_node_type")
        _coerce_int(self.block_relative_offset, "block_relative_offset")
        _coerce_bool(self.in_code_body, "in_code_body")
        _coerce_bool(self.structure_mask, "structure_mask")
        _coerce_string(self.language, "language")
        if not self.node_type:
            raise ValueError("node_type must be a non-empty string")
        if not self.parent_node_type:
            raise ValueError("parent_node_type must be a non-empty string")
        if self.block_relative_offset < 0:
            raise ValueError("block_relative_offset must be >= 0")
        if self.language != PYTHON_LANGUAGE:
            raise ValueError("language must be 'python'")

    def to_dict(self) -> dict[str, object]:
        return {
            "node_type": self.node_type,
            "parent_node_type": self.parent_node_type,
            "block_relative_offset": self.block_relative_offset,
            "in_code_body": self.in_code_body,
            "structure_mask": self.structure_mask,
            "language": self.language,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TokenChannelFeatures:
        if not isinstance(payload, Mapping):
            raise ValueError("TokenChannelFeatures payload must be a mapping")
        required_keys = {
            "node_type",
            "parent_node_type",
            "block_relative_offset",
            "in_code_body",
            "structure_mask",
        }
        missing_keys = sorted(required_keys - payload.keys())
        if missing_keys:
            raise ValueError(
                "Missing required TokenChannelFeatures keys: " + ", ".join(missing_keys)
            )
        return cls(
            node_type=_coerce_string(payload["node_type"], "node_type"),
            parent_node_type=_coerce_string(payload["parent_node_type"], "parent_node_type"),
            block_relative_offset=_coerce_int(payload["block_relative_offset"], "block_relative_offset"),
            in_code_body=_coerce_bool(payload["in_code_body"], "in_code_body"),
            structure_mask=_coerce_bool(payload["structure_mask"], "structure_mask"),
            language=_coerce_string(payload.get("language", PYTHON_LANGUAGE), "language"),
        )


def _coerce_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _coerce_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _coerce_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def collect_excluded_token_spans(source_code: str) -> tuple[ExcludedSpan, ...]:
    """Collect formal exclusion spans from Python source."""

    tree = ast.parse(source_code)
    line_starts = _build_line_starts(source_code)
    return _collect_excluded_token_spans_from_tree(source_code, tree, line_starts)


def prepare_token_channel_feature_context(source_code: str) -> TokenChannelFeatureContext:
    """Precompute reusable AST and structure-mask state for one source variant."""

    tree = ast.parse(source_code)
    line_starts = _build_line_starts(source_code)
    parent_map = {
        child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }
    structure_masks = [True] * len(source_code)
    for span in _collect_excluded_token_spans_from_tree(source_code, tree, line_starts):
        for index in range(max(0, span.start), min(len(source_code), span.end)):
            structure_masks[index] = False
    return TokenChannelFeatureContext(
        source_code=source_code,
        tree=tree,
        line_starts=line_starts,
        parent_map=parent_map,
        structure_masks=structure_masks,
    )


def _collect_excluded_token_spans_from_tree(
    source_code: str,
    tree: ast.AST,
    line_starts: list[int],
) -> tuple[ExcludedSpan, ...]:
    spans: list[ExcludedSpan] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            spans.append(
                ExcludedSpan(
                    label="import_statement",
                    start=_offset(line_starts, node.lineno, node.col_offset),
                    end=_offset(line_starts, node.end_lineno, node.end_col_offset),
                )
            )
            continue
        if isinstance(node, ast.ImportFrom):
            spans.append(
                ExcludedSpan(
                    label="import_from_statement",
                    start=_offset(line_starts, node.lineno, node.col_offset),
                    end=_offset(line_starts, node.end_lineno, node.end_col_offset),
                )
            )
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            spans.extend(_collect_decorator_spans(source_code, line_starts, node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature_end = _find_header_end(source_code, line_starts, node)
            spans.append(
                ExcludedSpan(
                    label="function_signature",
                    start=_offset(line_starts, node.lineno, node.col_offset),
                    end=signature_end,
                )
            )
            continue
        if isinstance(node, ast.ClassDef):
            header_end = _find_header_end(source_code, line_starts, node)
            spans.append(
                ExcludedSpan(
                    label="class_header",
                    start=_offset(line_starts, node.lineno, node.col_offset),
                    end=header_end,
                )
            )

    return tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label)))


def build_structure_masks(source_code: str) -> list[bool]:
    """Return a per-character structure mask for source reconstruction."""

    return prepare_token_channel_feature_context(source_code).structure_masks.copy()


def build_token_channel_features(
    source_code: str,
    token_start: int,
    token_end: int,
) -> TokenChannelFeatures:
    """Build structural token features for one token span."""

    context = prepare_token_channel_feature_context(source_code)
    return build_token_channel_features_from_context(
        context,
        token_start=token_start,
        token_end=token_end,
    )


def build_token_channel_features_from_context(
    context: TokenChannelFeatureContext,
    token_start: int,
    token_end: int,
) -> TokenChannelFeatures:
    """Build structural token features from precomputed source state."""

    if token_start < 0 or token_end < token_start or token_end > len(context.source_code):
        raise ValueError("token span is out of range")

    span_nodes = _find_nodes_covering_span(context.tree, context.line_starts, token_start, token_end)
    selected_node = _resolve_selected_node(span_nodes)
    statement_node = _nearest_statement(selected_node, context.parent_map)
    parent_node = context.parent_map.get(statement_node)
    block_relative_offset = _resolve_block_relative_offset(statement_node, parent_node)
    structure_mask = is_structure_safe_span(
        context.structure_masks,
        token_start,
        token_end,
        context.source_code,
    )

    return TokenChannelFeatures(
        node_type=_to_snake_case(type(statement_node).__name__),
        parent_node_type=_to_snake_case(type(parent_node).__name__) if parent_node is not None else "module",
        block_relative_offset=block_relative_offset,
        in_code_body=structure_mask and bool(context.source_code[token_start:token_end].strip()),
        structure_mask=structure_mask,
    )


def is_structure_safe_span(
    structure_masks: list[bool],
    start: int,
    end: int,
    source_code: str | None = None,
) -> bool:
    """Return whether all non-whitespace characters in the span are safe."""

    if start >= end:
        return False
    saw_non_whitespace = False
    whitespace_safe = True
    for index in range(start, end):
        if source_code is not None and source_code[index].isspace():
            whitespace_safe = whitespace_safe and structure_masks[index]
            continue
        saw_non_whitespace = True
        if not structure_masks[index]:
            return False
    if saw_non_whitespace:
        return True
    return whitespace_safe


def _collect_decorator_spans(
    source_code: str,
    line_starts: list[int],
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[ExcludedSpan]:
    spans: list[ExcludedSpan] = []
    for decorator in node.decorator_list:
        line_text = source_code.splitlines(keepends=True)[decorator.lineno - 1]
        at_column = line_text.rfind("@", 0, decorator.col_offset + 1)
        if at_column < 0:
            at_column = decorator.col_offset
        spans.append(
            ExcludedSpan(
                label="decorator",
                start=_offset(line_starts, decorator.lineno, at_column),
                end=_offset(line_starts, decorator.end_lineno, decorator.end_col_offset),
            )
        )
    return spans


def _build_line_starts(source_code: str) -> list[int]:
    line_starts = [0]
    for index, ch in enumerate(source_code):
        if ch == "\n":
            line_starts.append(index + 1)
    return line_starts


def _offset(line_starts: list[int], lineno: int | None, col_offset: int | None) -> int:
    if lineno is None or col_offset is None:
        raise ValueError("AST node is missing source position metadata")
    return line_starts[lineno - 1] + col_offset


def _find_header_end(
    source_code: str,
    line_starts: list[int],
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    if node.body:
        search_end = _offset(line_starts, node.body[0].lineno, node.body[0].col_offset)
    else:
        search_end = _offset(line_starts, node.end_lineno, node.end_col_offset)

    header_start = _offset(line_starts, node.lineno, node.col_offset)
    header_text = source_code[header_start:search_end]
    colon_index = header_text.rfind(":")
    if colon_index < 0:
        return search_end
    return header_start + colon_index + 1


def _find_nodes_covering_span(
    tree: ast.AST,
    line_starts: list[int],
    start: int,
    end: int,
) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        node_start = _offset(line_starts, getattr(node, "lineno"), getattr(node, "col_offset"))
        node_end = _offset(
            line_starts,
            getattr(node, "end_lineno"),
            getattr(node, "end_col_offset"),
        )
        if node_start <= start and node_end >= end:
            nodes.append(node)
    return nodes


def _resolve_selected_node(span_nodes: list[ast.AST]) -> ast.AST:
    if not span_nodes:
        return ast.Module(body=[], type_ignores=[])
    return min(
        span_nodes,
        key=lambda node: (
            getattr(node, "end_lineno", 0) - getattr(node, "lineno", 0),
            getattr(node, "end_col_offset", 0) - getattr(node, "col_offset", 0),
            len(list(ast.iter_child_nodes(node))),
        ),
    )


def _nearest_statement(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> ast.AST:
    current = node
    while current is not None and not isinstance(current, ast.stmt):
        current = parent_map.get(current)
    return current or ast.Module(body=[], type_ignores=[])


def _resolve_block_relative_offset(statement_node: ast.AST, parent_node: ast.AST | None) -> int:
    if parent_node is None:
        return 0
    for _, value in ast.iter_fields(parent_node):
        if isinstance(value, list) and statement_node in value:
            return value.index(statement_node)
    return 0


def _to_snake_case(name: str) -> str:
    result: list[str] = []
    for index, ch in enumerate(name):
        if ch.isupper() and index > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)
