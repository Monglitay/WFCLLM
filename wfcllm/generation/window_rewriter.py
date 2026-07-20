"""Key-blind causal adapter for complete semantic-window rewrites."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import io
import json
import token
import textwrap
import tokenize
from typing import Any, Protocol

from wfcllm.gate.data import RewriteCandidate, RewriteRequest
from wfcllm.windowing import (
    WINDOW_CONTRACT_VERSION,
    ParentDescriptor,
    PythonStatementUnitExtractor,
)


@dataclass(frozen=True)
class RewriteGeneration:
    """Raw output from a causal generation backend."""

    token_ids: tuple[int, ...]
    text: str
    generation_seed_id: str
    rewrite_config_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or any(
            type(value) is not int or value < 0 for value in self.token_ids
        ):
            raise ValueError("token_ids must be a tuple of non-negative integers")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.generation_seed_id, str) or not self.generation_seed_id:
            raise ValueError("generation_seed_id must be non-empty")
        if not isinstance(self.rewrite_config_id, str) or not self.rewrite_config_id:
            raise ValueError("rewrite_config_id must be non-empty")


@dataclass(frozen=True)
class ParsedRewrite:
    """Exact generated tokens plus parser facts for an online candidate."""

    token_ids: tuple[int, ...]
    text: str
    parser_spans: tuple[tuple[int, int], ...]
    parent_descriptor: str | None
    parse_status: str
    unit_count: int
    same_parent_scope: bool
    generation_seed_id: str
    rewrite_config_id: str
    semantic_validation_rule: str | None = None
    semantic_equivalence_certified: bool | None = None

    def __post_init__(self) -> None:
        if (self.semantic_validation_rule is None) != (
            self.semantic_equivalence_certified is None
        ):
            raise ValueError(
                "semantic validation rule and certificate must be present together"
            )
        if self.semantic_validation_rule is not None and (
            not isinstance(self.semantic_validation_rule, str)
            or not self.semantic_validation_rule
            or not isinstance(self.semantic_equivalence_certified, bool)
        ):
            raise ValueError("semantic validation certificate is invalid")


class CausalRewriteBackend(Protocol):
    def generate_window(
        self,
        *,
        prompt: str,
        completed_prefix: str,
        original_window: str,
        candidate_index: int,
        max_units: int,
    ) -> RewriteGeneration: ...


class KeyBlindWhitespaceWindowRewriter:
    """Create public, semantics-neutral lexical variants without a model call.

    The candidate index controls only the amount of insignificant internal
    whitespace.  The deployment key and watermark outcome are deliberately
    unavailable here, so the gated retry controller remains the only selector.
    """

    def rewrite_window(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> RewriteGeneration:
        _validate_call(request, candidate_index)
        text = _internal_whitespace_variant(
            request.original_window,
            candidate_index=candidate_index,
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return RewriteGeneration(
            token_ids=(),
            text=text,
            generation_seed_id=f"public-key-blind-whitespace:{digest}",
            rewrite_config_id="public-key-blind-whitespace/v1",
        )


class KeyBlindAstEquivalentWindowRewriter:
    """Emit fixed key-blind variants whose Python AST is byte-independently equal.

    The deployment key and LSH outcome never enter candidate construction.  A
    candidate is certified only when parsing both windows with type comments
    enabled produces identical attribute-free ASTs.  This is intentionally
    conservative: unavailable variants remain visible but uncertified.
    """

    validation_rule = "python-ast-equivalent/v1"

    def __init__(self, *, extractor: PythonStatementUnitExtractor | None = None) -> None:
        self._extractor = extractor or PythonStatementUnitExtractor()

    def rewrite_windows(
        self,
        request: RewriteRequest,
        *,
        candidate_indices: tuple[int, ...],
    ) -> tuple[ParsedRewrite, ...]:
        if candidate_indices != (1, 2, 3):
            raise ValueError(
                "candidate_indices must equal the public trajectory (1, 2, 3)"
            )
        return tuple(
            self.rewrite_window(request, candidate_index=index)
            for index in candidate_indices
        )

    def rewrite_many(
        self,
        request: RewriteRequest,
        *,
        candidate_indices: tuple[int, ...],
    ) -> tuple[RewriteCandidate, ...]:
        return tuple(
            self._candidate_from_parsed(request, parsed)
            for parsed in self.rewrite_windows(
                request,
                candidate_indices=candidate_indices,
            )
        )

    def rewrite(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> RewriteCandidate:
        return self._candidate_from_parsed(
            request,
            self.rewrite_window(request, candidate_index=candidate_index),
        )

    def rewrite_window(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> ParsedRewrite:
        _validate_call(request, candidate_index)
        if candidate_index <= 12:
            validation_rule = self.validation_rule
            text = _ast_equivalent_variant(
                request.original_window,
                candidate_index=candidate_index,
            )
            certified = (
                text != request.original_window
                and python_ast_equivalent(request.original_window, text)
            )
        else:
            validation_rule = "python-literal-equivalent/v1"
            text = _literal_equivalent_variant(
                request.original_window,
                candidate_index=candidate_index,
            )
            certified = (
                text != request.original_window
                and python_literal_equivalent(request.original_window, text)
            )
        generation = RewriteGeneration(
            token_ids=(),
            text=text,
            generation_seed_id=(
                "public-key-blind-ast:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            ),
            rewrite_config_id=f"public-key-blind-ast-equivalent/v1:{candidate_index}",
        )
        parsed = _parse_generation(
            request,
            generation,
            extractor=self._extractor,
        )
        return replace(
            parsed,
            semantic_validation_rule=validation_rule,
            semantic_equivalence_certified=certified,
        )

    @staticmethod
    def _candidate_from_parsed(
        request: RewriteRequest,
        parsed: ParsedRewrite,
    ) -> RewriteCandidate:
        prefix_bytes = len(request.completed_prefix.encode("utf-8"))
        end_byte = prefix_bytes + len(parsed.text.encode("utf-8"))
        return RewriteCandidate(
            code=parsed.text,
            parse_status=parsed.parse_status,
            unit_count=parsed.unit_count,
            same_parent_scope=parsed.same_parent_scope,
            boundary_span=(prefix_bytes, end_byte),
            generation_seed_id=parsed.generation_seed_id,
            rewrite_config_id=parsed.rewrite_config_id,
        )


class CausalWindowRewriter:
    """Generate an entire window without observing gate or LSH outcomes.

    The backend receives only the public causal prefix and original window.
    Formal collection and online generation both request the same public
    A/B/C trajectory, with no hidden pool, filtering, or replacement.
    """

    def __init__(
        self,
        backend: CausalRewriteBackend,
        *,
        extractor: PythonStatementUnitExtractor | None = None,
        generation_attempts: int = 3,
    ) -> None:
        if not callable(getattr(backend, "generate_window", None)):
            raise ValueError("backend must define generate_window")
        if type(generation_attempts) is not int or generation_attempts != 3:
            raise ValueError("generation_attempts must equal the three public candidates")
        self._backend = backend
        self._extractor = extractor or PythonStatementUnitExtractor()
        self._generation_attempts = generation_attempts

    def rewrite_generation(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> RewriteGeneration:
        _validate_call(request, candidate_index)
        value = self._backend.generate_window(
            prompt=request.prompt,
            completed_prefix=request.completed_prefix,
            original_window=request.original_window,
            candidate_index=candidate_index,
            max_units=request.window_length,
        )
        if not isinstance(value, RewriteGeneration):
            raise ValueError("causal backend must return RewriteGeneration")
        return value

    def rewrite(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> RewriteCandidate:
        parsed = self.rewrite_window(request, candidate_index=candidate_index)
        prefix_bytes = len(request.completed_prefix.encode("utf-8"))
        end_byte = prefix_bytes + len(parsed.text.encode("utf-8"))
        return RewriteCandidate(
            code=parsed.text,
            parse_status=parsed.parse_status,
            unit_count=parsed.unit_count,
            same_parent_scope=parsed.same_parent_scope,
            boundary_span=(prefix_bytes, end_byte),
            generation_seed_id=parsed.generation_seed_id,
            rewrite_config_id=parsed.rewrite_config_id,
        )

    def rewrite_many(
        self,
        request: RewriteRequest,
        *,
        candidate_indices: tuple[int, ...],
    ) -> tuple[RewriteCandidate, ...]:
        """Generate a trajectory in one backend call when batching is available."""

        parsed = self.rewrite_windows(
            request, candidate_indices=candidate_indices
        )
        return tuple(
            self._candidate_from_parsed(request, value) for value in parsed
        )

    def rewrite_windows(
        self,
        request: RewriteRequest,
        *,
        candidate_indices: tuple[int, ...],
    ) -> tuple[ParsedRewrite, ...]:
        """Generate the one public A/B/C trajectory without replacement."""

        if candidate_indices != (1, 2, 3):
            raise ValueError("candidate_indices must equal the public trajectory (1, 2, 3)")
        for candidate_index in candidate_indices:
            _validate_call(request, candidate_index)
        generate_many = getattr(self._backend, "generate_windows", None)
        if not callable(generate_many):
            generated = tuple(
                self.rewrite_generation(request, candidate_index=index)
                for index in candidate_indices
            )
        else:
            generated = generate_many(
                prompt=request.prompt,
                completed_prefix=request.completed_prefix,
                original_window=request.original_window,
                candidate_indices=candidate_indices,
                max_units=request.window_length,
            )
        if (
            not isinstance(generated, tuple)
            or len(generated) != len(candidate_indices)
            or any(not isinstance(value, RewriteGeneration) for value in generated)
        ):
            raise ValueError("causal batch backend returned an invalid trajectory")
        return tuple(
            self._parse_generation(request, generation) for generation in generated
        )

    def rewrite_window(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> ParsedRewrite:
        generation = self.rewrite_generation(request, candidate_index=candidate_index)
        return self._parse_generation(request, generation)

    def _candidate_from_parsed(
        self,
        request: RewriteRequest,
        parsed: ParsedRewrite,
    ) -> RewriteCandidate:
        prefix_bytes = len(request.completed_prefix.encode("utf-8"))
        end_byte = prefix_bytes + len(parsed.text.encode("utf-8"))
        return RewriteCandidate(
            code=parsed.text,
            parse_status=parsed.parse_status,
            unit_count=parsed.unit_count,
            same_parent_scope=parsed.same_parent_scope,
            boundary_span=(prefix_bytes, end_byte),
            generation_seed_id=parsed.generation_seed_id,
            rewrite_config_id=parsed.rewrite_config_id,
        )

    def _parse_generation(
        self,
        request: RewriteRequest,
        generation: RewriteGeneration,
    ) -> ParsedRewrite:
        return _parse_generation(
            request,
            generation,
            extractor=self._extractor,
        )


def _parse_generation(
    request: RewriteRequest,
    generation: RewriteGeneration,
    *,
    extractor: PythonStatementUnitExtractor,
) -> ParsedRewrite:
    prefix_bytes = len(request.completed_prefix.encode("utf-8"))
    complete_source = request.completed_prefix + generation.text
    units = [
        unit
        for unit in extractor.extract(complete_source)
        if unit.start_byte >= prefix_bytes
    ]
    if not generation.text or not units:
        parse_status = "parse_error"
    elif len(units) not in {1, 2, 3} or len(units) != request.window_length:
        parse_status = "unit_count_out_of_range"
    elif any(unit.hard_boundary for unit in units):
        parse_status = "parse_error"
    else:
        parse_status = "ok"

    same_parent = bool(units) and _same_parent(units)
    if same_parent:
        same_parent = _descriptor(units[0]).canonical == request.canonical_parent
    if parse_status == "ok" and not same_parent:
        parse_status = "scope_changed"

    parent_descriptor = _descriptor(units[0]).canonical if units else None
    return ParsedRewrite(
        token_ids=generation.token_ids,
        text=generation.text,
        parser_spans=tuple((unit.start_byte, unit.end_byte) for unit in units),
        parent_descriptor=parent_descriptor,
        parse_status=parse_status,
        unit_count=len(units),
        same_parent_scope=same_parent,
        generation_seed_id=generation.generation_seed_id,
        rewrite_config_id=generation.rewrite_config_id,
    )


def python_ast_equivalent(reference: str, candidate: str) -> bool:
    """Return true only for windows with identical parsed Python ASTs."""

    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise ValueError("reference and candidate must be strings")
    try:
        reference_tree = ast.parse(textwrap.dedent(reference), type_comments=True)
        candidate_tree = ast.parse(textwrap.dedent(candidate), type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return False
    return ast.dump(reference_tree, include_attributes=False) == ast.dump(
        candidate_tree,
        include_attributes=False,
    )


def python_literal_equivalent(reference: str, candidate: str) -> bool:
    """Prove equivalence after folding only closed literal identities."""

    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise ValueError("reference and candidate must be strings")
    try:
        reference_tree = ast.parse(textwrap.dedent(reference), type_comments=True)
        candidate_tree = ast.parse(textwrap.dedent(candidate), type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return False
    if _docstring_values(reference_tree) != _docstring_values(candidate_tree):
        return False
    folded = _LiteralIdentityFolder().visit(candidate_tree)
    ast.fix_missing_locations(folded)
    return ast.dump(reference_tree, include_attributes=False) == ast.dump(
        folded,
        include_attributes=False,
    )


def _ast_equivalent_variant(text: str, *, candidate_index: int) -> str:
    prefix = _common_indentation(text)
    dedented = textwrap.dedent(text)
    if not dedented.strip():
        return text

    mode = (candidate_index - 1) % 3
    if mode == 0:
        transformed = _rewrite_first_string(dedented)
    elif mode == 1:
        transformed = _rewrite_first_integer(dedented)
    else:
        transformed = _rewrite_first_string(dedented)
        transformed = _rewrite_first_integer(transformed)
    transformed = _parenthesize_first_expression(
        transformed,
        depth=candidate_index,
    )

    if not python_ast_equivalent(dedented, transformed):
        return text
    return _restore_indentation(transformed, prefix)


def _literal_equivalent_variant(text: str, *, candidate_index: int) -> str:
    prefix = _common_indentation(text)
    dedented = textwrap.dedent(text)
    try:
        tree = ast.parse(dedented, type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return text
    literal_trajectory_index = candidate_index - 13
    transformer = _LiteralVariantTransformer(
        literal_trajectory_index % 6,
        occurrence=literal_trajectory_index // 6,
    )
    transformed_tree = transformer.visit(tree)
    if not transformer.changed:
        return text
    ast.fix_missing_locations(transformed_tree)
    try:
        transformed = ast.unparse(transformed_tree)
    except (TypeError, ValueError, RecursionError):
        return text
    if text.endswith(("\n", "\r")):
        transformed += "\n"
    candidate = _restore_indentation(transformed, prefix)
    return candidate if python_literal_equivalent(text, candidate) else text


class _LiteralVariantTransformer(ast.NodeTransformer):
    def __init__(self, mode: int, *, occurrence: int) -> None:
        self.mode = mode
        self.occurrence = occurrence
        self._eligible_seen = 0
        self.changed = False
        self._skip_standalone_string = False

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node
        return self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        return node

    def visit_Constant(self, node: ast.Constant):
        if self.changed or isinstance(node.value, bool):
            return node
        value = node.value
        if type(value) is int:
            replacements = (
                ast.BinOp(node, ast.Add(), ast.Constant(0)),
                ast.BinOp(ast.Constant(0), ast.Add(), node),
                ast.BinOp(node, ast.Sub(), ast.Constant(0)),
                ast.BinOp(node, ast.Mult(), ast.Constant(1)),
                ast.BinOp(
                    ast.BinOp(node, ast.Add(), ast.Constant(1)),
                    ast.Sub(),
                    ast.Constant(1),
                ),
                ast.BinOp(node, ast.FloorDiv(), ast.Constant(1)),
            )
        elif isinstance(value, str):
            empty = ast.Constant("")
            replacements = (
                ast.BinOp(node, ast.Add(), empty),
                ast.BinOp(empty, ast.Add(), node),
                ast.BinOp(node, ast.Mult(), ast.Constant(1)),
                ast.BinOp(ast.Constant(1), ast.Mult(), node),
                ast.BinOp(ast.BinOp(node, ast.Add(), empty), ast.Add(), ast.Constant("")),
                ast.BinOp(ast.BinOp(empty, ast.Add(), ast.Constant("")), ast.Add(), node),
            )
        else:
            return node
        if self._eligible_seen != self.occurrence:
            self._eligible_seen += 1
            return node
        self._eligible_seen += 1
        self.changed = True
        return ast.copy_location(replacements[self.mode], node)


class _LiteralIdentityFolder(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        if not isinstance(node.left, ast.Constant) or not isinstance(
            node.right, ast.Constant
        ):
            return node
        left = node.left.value
        right = node.right.value
        if isinstance(left, bool) or isinstance(right, bool):
            return node
        try:
            if isinstance(node.op, ast.Add) and (
                (type(left) is int and type(right) is int)
                or (isinstance(left, str) and isinstance(right, str))
            ):
                value = left + right
            elif isinstance(node.op, ast.Sub) and type(left) is int and type(right) is int:
                value = left - right
            elif isinstance(node.op, ast.Mult) and (
                (type(left) is int and type(right) is int)
                or (isinstance(left, str) and type(right) is int)
                or (type(left) is int and isinstance(right, str))
            ):
                value = left * right
            elif (
                isinstance(node.op, ast.FloorDiv)
                and type(left) is int
                and type(right) is int
                and right != 0
            ):
                value = left // right
            else:
                return node
        except (ArithmeticError, MemoryError, OverflowError):
            return node
        return ast.copy_location(ast.Constant(value=value), node)


def _docstring_values(tree: ast.AST) -> tuple[str | None, ...]:
    containers = (
        ast.Module,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    return tuple(
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, containers)
    )


def _rewrite_first_string(text: str) -> str:
    try:
        items = tuple(tokenize.generate_tokens(io.StringIO(text).readline))
    except (IndentationError, tokenize.TokenError):
        return text
    for item in items:
        if item.type != token.STRING or item.string[:1].lower() in {"b", "f", "r", "u"}:
            continue
        try:
            value = ast.literal_eval(item.string)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(value, str):
            continue
        replacement = json.dumps(value, ensure_ascii=False)
        if replacement == item.string:
            replacement = repr(value)
        if replacement == item.string:
            continue
        return _replace_token(text, item, replacement)
    return text


def _rewrite_first_integer(text: str) -> str:
    try:
        items = tuple(tokenize.generate_tokens(io.StringIO(text).readline))
    except (IndentationError, tokenize.TokenError):
        return text
    for item in items:
        if item.type != token.NUMBER:
            continue
        try:
            value = ast.literal_eval(item.string)
        except (SyntaxError, ValueError):
            continue
        if type(value) is not int or value < 0:
            continue
        replacement = hex(value)
        if replacement == item.string:
            replacement = f"{value:_d}"
        if replacement == item.string:
            continue
        return _replace_token(text, item, replacement)
    return text


def _parenthesize_first_expression(text: str, *, depth: int) -> str:
    try:
        tree = ast.parse(text, type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return text
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.expr)
        and hasattr(node, "lineno")
        and hasattr(node, "end_lineno")
    ]
    if not candidates:
        return text
    node = min(
        candidates,
        key=lambda item: (item.lineno, item.col_offset, -(item.end_col_offset - item.col_offset)),
    )
    source = text.encode("utf-8")
    start = _byte_offset(source, node.lineno, node.col_offset)
    end = _byte_offset(source, node.end_lineno, node.end_col_offset)
    opening = b"(" * depth
    closing = b")" * depth
    return (source[:start] + opening + source[start:end] + closing + source[end:]).decode(
        "utf-8"
    )


def _replace_token(text: str, item: tokenize.TokenInfo, replacement: str) -> str:
    start = _text_offset(text, item.start)
    end = _text_offset(text, item.end)
    return text[:start] + replacement + text[end:]


def _byte_offset(source: bytes, line_number: int, byte_column: int) -> int:
    lines = source.splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise ValueError("AST position is outside source")
    return sum(len(line) for line in lines[: line_number - 1]) + byte_column


def _common_indentation(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line[: len(line) - len(line.lstrip(" \t"))]
    return ""


def _restore_indentation(text: str, prefix: str) -> str:
    trailing_newline = text.endswith(("\n", "\r"))
    lines = text.splitlines()
    restored = "\n".join(prefix + line if line.strip() else line for line in lines)
    return restored + ("\n" if trailing_newline else "")


def _validate_call(request: object, candidate_index: object) -> None:
    if not isinstance(request, RewriteRequest):
        raise ValueError("request must be a RewriteRequest")
    if type(candidate_index) is not int or candidate_index < 1:
        raise ValueError("candidate_index must be a positive integer")


def _internal_whitespace_variant(text: str, *, candidate_index: int) -> str:
    """Insert ignored internal whitespace while retaining the parsed statement."""

    try:
        items = tuple(tokenize.generate_tokens(io.StringIO(text).readline))
    except (IndentationError, tokenize.TokenError) as exc:
        raise ValueError("original window must tokenize before rewriting") from exc
    ignored = {
        token.ENCODING,
        token.INDENT,
        token.DEDENT,
        token.NEWLINE,
        tokenize.NL,
        token.ENDMARKER,
        token.COMMENT,
    }
    meaningful = tuple(item for item in items if item.type not in ignored)
    spaces = " " * candidate_index
    for left, right in zip(meaningful, meaningful[1:]):
        if left.end[0] != right.start[0]:
            continue
        offset = _text_offset(text, left.end)
        return text[:offset] + spaces + text[offset:]
    if len(meaningful) != 1:
        raise ValueError("original window has no safe lexical rewrite point")
    only = meaningful[0]
    start = _text_offset(text, only.start)
    end = _text_offset(text, only.end)
    if only.string in {"return", "yield"}:
        return text[:end] + spaces + "None" + text[end:]
    return (
        text[:start]
        + "("
        + spaces
        + text[start:end]
        + spaces
        + ")"
        + text[end:]
    )


def _text_offset(text: str, point: tuple[int, int]) -> int:
    line_number, column = point
    lines = text.splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise ValueError("token position is outside original window")
    return sum(len(line) for line in lines[: line_number - 1]) + column


def _same_parent(units: list[Any]) -> bool:
    first = units[0]
    return all(
        unit.depth == first.depth
        and unit.parent_path == first.parent_path
        and unit.direct_parent_type == first.direct_parent_type
        for unit in units
    )


def _descriptor(unit: Any) -> ParentDescriptor:
    return ParentDescriptor(
        contract_version=WINDOW_CONTRACT_VERSION,
        ancestor_node_types=unit.parent_path[:-1],
        direct_parent_type=unit.direct_parent_type,
        first_unit_ordinal=unit.direct_child_ordinal,
        compound_header_role="header" if unit.compound_header else "body",
    )


__all__ = [
    "CausalWindowRewriter",
    "KeyBlindAstEquivalentWindowRewriter",
    "KeyBlindWhitespaceWindowRewriter",
    "ParsedRewrite",
    "RewriteGeneration",
    "python_ast_equivalent",
    "python_literal_equivalent",
]
