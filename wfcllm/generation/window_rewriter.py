"""Key-blind causal adapter for complete semantic-window rewrites."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import io
import json
import re
import token
import textwrap
import tokenize
from typing import Any, Callable, Protocol

from wfcllm.gate.data import RewriteCandidate, RewriteRequest
from wfcllm.windowing import (
    WINDOW_CONTRACT_VERSION,
    ParentDescriptor,
    PythonStatementUnitExtractor,
)


class StatementUnitExtractor(Protocol):
    def extract(self, source: str) -> list[Any]: ...


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

    def __init__(
        self,
        *,
        extractor: PythonStatementUnitExtractor | None = None,
        ast_variant_budget: int = 12,
        comprehension_alpha: bool = True,
    ) -> None:
        # ast_variant_budget=12 with alpha reproduces the nocarrier trajectory;
        # 48 without alpha reproduces the opencoder dense48 trajectory.
        if type(ast_variant_budget) is not int or ast_variant_budget < 1:
            raise ValueError("ast_variant_budget must be a positive integer")
        if not isinstance(comprehension_alpha, bool):
            raise ValueError("comprehension_alpha must be a bool")
        self._extractor = extractor or PythonStatementUnitExtractor()
        self.ast_variant_budget = ast_variant_budget
        self.comprehension_alpha = comprehension_alpha

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
        if candidate_index <= self.ast_variant_budget:
            validation_rule = self.validation_rule
            text = _ast_equivalent_variant(
                request.original_window,
                candidate_index=candidate_index,
            )
            if self.comprehension_alpha and candidate_index >= 4:
                alpha_variant = _comprehension_alpha_variant(
                    request.original_window,
                    candidate_index=candidate_index,
                )
                if alpha_variant != request.original_window:
                    validation_rule = (
                        "python-comprehension-alpha-equivalent/v1"
                    )
                    text = alpha_variant
        else:
            validation_rule = "python-literal-equivalent/v1"
            text = _literal_equivalent_variant(
                request.original_window,
                candidate_index=candidate_index,
                first_literal_index=self.ast_variant_budget + 1,
            )
        text = _preserve_trailing_layout(request.original_window, text)
        if validation_rule == self.validation_rule:
            certified = (
                text != request.original_window
                and python_ast_equivalent(request.original_window, text)
            )
        elif validation_rule == "python-comprehension-alpha-equivalent/v1":
            certified = (
                text != request.original_window
                and python_comprehension_alpha_equivalent(
                    request.original_window,
                    text,
                )
            )
        else:
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


class KeyBlindCppEquivalentWindowRewriter:
    """Emit conservative C++ variants without consulting keys or LSH outcomes."""

    validation_rule = "cpp-keyblind-equivalent/v1"

    def __init__(
        self,
        *,
        extractor: StatementUnitExtractor,
        window_contract_version: str,
    ) -> None:
        self._extractor = extractor
        self._window_contract_version = window_contract_version

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
        text = _cpp_equivalent_variant(
            request.original_window,
            candidate_index=candidate_index,
        )
        generation = RewriteGeneration(
            token_ids=(),
            text=text,
            generation_seed_id=(
                "public-key-blind-cpp:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            ),
            rewrite_config_id=f"public-key-blind-cpp-equivalent/v1:{candidate_index}",
        )
        parsed = _parse_generation(
            request,
            generation,
            extractor=self._extractor,
            window_contract_version=self._window_contract_version,
        )
        return replace(
            parsed,
            semantic_validation_rule=self.validation_rule,
            semantic_equivalence_certified=text != request.original_window,
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


class KeyBlindJavaEquivalentWindowRewriter:
    """Emit conservative Java variants without consulting keys or LSH outcomes."""

    validation_rule = "java-keyblind-equivalent/v1"

    def __init__(
        self,
        *,
        extractor: StatementUnitExtractor,
        window_contract_version: str,
    ) -> None:
        self._extractor = extractor
        self._window_contract_version = window_contract_version

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
        text = _java_equivalent_variant(
            request.original_window,
            candidate_index=candidate_index,
        )
        generation = RewriteGeneration(
            token_ids=(),
            text=text,
            generation_seed_id=(
                "public-key-blind-java:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            ),
            rewrite_config_id=f"public-key-blind-java-equivalent/v1:{candidate_index}",
        )
        parsed = _parse_generation(
            request,
            generation,
            extractor=self._extractor,
            window_contract_version=self._window_contract_version,
        )
        return replace(
            parsed,
            semantic_validation_rule=self.validation_rule,
            semantic_equivalence_certified=text != request.original_window,
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
        extractor: StatementUnitExtractor | None = None,
        window_contract_version: str = WINDOW_CONTRACT_VERSION,
        generation_attempts: int = 3,
    ) -> None:
        if not callable(getattr(backend, "generate_window", None)):
            raise ValueError("backend must define generate_window")
        if type(generation_attempts) is not int or generation_attempts != 3:
            raise ValueError("generation_attempts must equal the three public candidates")
        self._backend = backend
        self._extractor = extractor or PythonStatementUnitExtractor()
        self._generation_attempts = generation_attempts
        self._window_contract_version = window_contract_version

    def for_extractor(
        self,
        extractor: StatementUnitExtractor,
        *,
        window_contract_version: str,
    ) -> CausalWindowRewriter:
        return CausalWindowRewriter(
            self._backend,
            extractor=extractor,
            generation_attempts=self._generation_attempts,
            window_contract_version=window_contract_version,
        )

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
            window_contract_version=self._window_contract_version,
        )


def _parse_generation(
    request: RewriteRequest,
    generation: RewriteGeneration,
    *,
    extractor: StatementUnitExtractor,
    window_contract_version: str = WINDOW_CONTRACT_VERSION,
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
        same_parent = (
            _descriptor(units[0], window_contract_version).canonical
            == request.canonical_parent
        )
    if parse_status == "ok" and not same_parent:
        parse_status = "scope_changed"

    parent_descriptor = (
        _descriptor(units[0], window_contract_version).canonical if units else None
    )
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


_COMPOUND_HEADER_AST_TYPES = (
    ast.AsyncFor,
    ast.AsyncWith,
    ast.For,
    ast.If,
    ast.While,
    ast.With,
)


def _compound_header_scaffold(source: str) -> str | None:
    """Complete an expression-bearing header solely for local AST validation."""

    stripped = source.lstrip()
    if not stripped or not source.rstrip().endswith(":"):
        return None
    supported_prefixes = (
        "async for ",
        "async with ",
        "for ",
        "if ",
        "while ",
        "with ",
    )
    if stripped.startswith("elif "):
        offset = len(source) - len(stripped)
        source = source[:offset] + "if  " + source[offset + len("elif") :]
    elif not stripped.startswith(supported_prefixes):
        return None
    return source.rstrip() + "\n    pass\n"


def _parse_ast_window(source: str) -> ast.Module:
    """Parse a complete window or a conservatively scaffolded compound header."""

    dedented = textwrap.dedent(source)
    try:
        return ast.parse(dedented, type_comments=True)
    except (SyntaxError, ValueError, TypeError) as original_error:
        scaffold = _compound_header_scaffold(dedented)
        if scaffold is None:
            raise original_error
        tree = ast.parse(scaffold, type_comments=True)
        if (
            len(tree.body) != 1
            or not isinstance(tree.body[0], _COMPOUND_HEADER_AST_TYPES)
            or len(tree.body[0].body) != 1
            or not isinstance(tree.body[0].body[0], ast.Pass)
            or getattr(tree.body[0], "orelse", [])
        ):
            raise original_error
        return tree


def _window_tree(text: str) -> ast.Module:
    # Synthetic-function wrapping (nocarrier) handles exact parser slices;
    # the elif scaffold (opencoder) covers headers the wrapper cannot parse.
    try:
        return _wrap_window_source(text).tree
    except (SyntaxError, ValueError, TypeError):
        return _parse_ast_window(text)


def python_ast_equivalent(reference: str, candidate: str) -> bool:
    """Return true only for windows with identical parsed Python ASTs."""

    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise ValueError("reference and candidate must be strings")
    try:
        reference_tree = _window_tree(reference)
        candidate_tree = _window_tree(candidate)
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
        reference_tree = _wrap_window_source(reference).tree
        candidate_tree = _wrap_window_source(candidate).tree
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


def _java_equivalent_variant(text: str, *, candidate_index: int) -> str:
    body = text.rstrip()
    trailing = text[len(body):]
    variants = (
        _java_parenthesize_return,
        _java_ternary_return,
        lambda value: _java_boolean_return(value, operator="&&"),
        lambda value: _java_boolean_return(value, operator="||"),
        _java_boolean_object_ternary_return,
        _java_parenthesize_condition,
        lambda value: _java_boolean_condition(value, operator="&&"),
        lambda value: _java_boolean_condition(value, operator="||"),
        _java_parenthesize_initializer_or_assignment,
        lambda value: _java_boolean_initializer_or_assignment(value, operator="&&"),
        lambda value: _java_boolean_initializer_or_assignment(value, operator="||"),
    )
    transform = variants[(candidate_index - 1) % len(variants)]
    if "\n" not in body:
        transformed = transform(body)
        return transformed + trailing
    return _java_transform_first_matching_line(text, transform)


def _java_transform_first_matching_line(
    text: str,
    transform: Callable[[str], str],
) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        body = line.rstrip()
        trailing = line[len(body):]
        transformed = transform(body) + trailing
        if transformed != line:
            return "".join((*lines[:index], transformed, *lines[index + 1:]))
    return text


def _java_parenthesize_return(text: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not _java_safe_expression(expression):
        return text
    return f"{match.group(1)}({expression}){match.group(3)}"


def _java_ternary_return(text: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not _java_safe_expression(expression):
        return text
    return f"{match.group(1)}(true ? ({expression}) : ({expression})){match.group(3)}"


def _java_boolean_object_ternary_return(text: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not _java_safe_expression(expression):
        return text
    return (
        f"{match.group(1)}(Boolean.TRUE ? ({expression}) : ({expression}))"
        f"{match.group(3)}"
    )


def _java_boolean_return(text: str, *, operator: str) -> str:
    if operator not in {"&&", "||"}:
        raise ValueError("unsupported boolean return operator")
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not _java_boolean_expression(expression):
        return text
    identity = "true" if operator == "&&" else "false"
    return f"{match.group(1)}(({expression}) {operator} {identity}){match.group(3)}"


def _java_parenthesize_condition(text: str) -> str:
    match = re.match(
        r"^(\s*(?:if|while)\s*\()(.+?)(\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if match:
        condition = match.group(2).strip()
        if not condition:
            return text
        return f"{match.group(1)}({condition}){match.group(3)}"
    match = re.match(
        r"^(\s*for\s*\()(.*?;)(.*?)(;.*?\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    condition = match.group(3).strip()
    if not condition:
        return text
    return f"{match.group(1)}{match.group(2)} ({condition}){match.group(4)}"


def _java_boolean_condition(text: str, *, operator: str) -> str:
    if operator not in {"&&", "||"}:
        raise ValueError("unsupported boolean condition operator")
    identity = "true" if operator == "&&" else "false"
    match = re.match(
        r"^(\s*(?:if|while)\s*\()(.+?)(\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if match:
        condition = match.group(2).strip()
        if not condition:
            return text
        return f"{match.group(1)}(({condition}) {operator} {identity}){match.group(3)}"
    match = re.match(
        r"^(\s*for\s*\()(.*?;)(.*?)(;.*?\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    condition = match.group(3).strip()
    if not condition:
        return text
    return f"{match.group(1)}{match.group(2)} (({condition}) {operator} {identity}){match.group(4)}"


def _java_parenthesize_initializer_or_assignment(text: str) -> str:
    match = _java_initializer_or_assignment(text)
    if not match:
        return text
    expression = match.group(2).strip()
    if not _java_safe_expression(expression):
        return text
    return f"{match.group(1)}({expression}){match.group(3)}"


def _java_boolean_initializer_or_assignment(text: str, *, operator: str) -> str:
    if operator not in {"&&", "||"}:
        raise ValueError("unsupported boolean initializer operator")
    match = re.match(
        r"^(\s*(?:(?:final\s+)?boolean\s+\w+|[A-Za-z_]\w*)\s*=\s*)"
        r"(.+?)(\s*;\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    expression = match.group(2).strip()
    if not _java_boolean_expression(expression):
        return text
    identity = "true" if operator == "&&" else "false"
    return f"{match.group(1)}(({expression}) {operator} {identity}){match.group(3)}"


def _java_initializer_or_assignment(text: str) -> re.Match[str] | None:
    declaration = re.match(
        r"^(\s*(?:final\s+)?(?:boolean|byte|short|int|long|float|double|char|"
        r"String|[A-Z]\w*(?:<[^;=]+>)?(?:\[\])?)\s+\w+\s*=\s*)"
        r"(.+?)(\s*;\s*)$",
        text,
        re.DOTALL,
    )
    if declaration:
        return declaration
    return re.match(
        r"^(\s*[A-Za-z_]\w*(?:(?:\.[A-Za-z_]\w*)|(?:\[[^\]]+\]))*\s*=\s*)"
        r"(.+?)(\s*;\s*)$",
        text,
        re.DOTALL,
    )


def _java_safe_expression(expression: str) -> bool:
    value = expression.strip()
    if not value or value.startswith("{") or "->" in value:
        return False
    if re.search(r"\bnew\b.*\{", value):
        return False
    return True


def _java_boolean_expression(expression: str) -> bool:
    value = expression.strip()
    if not _java_safe_expression(value):
        return False
    if value in {"true", "false"}:
        return True
    return bool(
        re.search(r"(?:==|!=|<=|>=|&&|\|\||[<>])", value)
        or re.match(r"!\s*[A-Za-z_(]", value)
    )


def _cpp_equivalent_variant(text: str, *, candidate_index: int) -> str:
    body = text.rstrip()
    trailing = text[len(body):]
    variants = (
        _cpp_parenthesize_return,
        _cpp_parenthesize_condition,
        _cpp_parenthesize_initializer_or_assignment,
        lambda value: _cpp_comment_return(value, tag="r1"),
        lambda value: _cpp_comment_condition(value, tag="c1"),
        lambda value: _cpp_comment_initializer_or_assignment(value, tag="a1"),
        lambda value: _cpp_ternary_return(value),
        lambda value: _cpp_boolean_condition(value, operator="&&"),
        lambda value: _cpp_ternary_initializer_or_assignment(value),
        lambda value: _cpp_comment_return(value, tag="r2"),
        lambda value: _cpp_boolean_condition(value, operator="||"),
        lambda value: _cpp_comment_initializer_or_assignment(value, tag="a2"),
        _cpp_comma_return,
        _cpp_comma_condition,
        _cpp_comma_initializer_or_assignment,
        _cpp_sizeof_return,
        _cpp_not_not_condition,
        _cpp_sizeof_initializer_or_assignment,
    )
    transform = variants[(candidate_index - 1) % len(variants)]
    if "\n" not in body:
        transformed = transform(body)
        return transformed + trailing
    return _cpp_transform_first_matching_line(text, transform)


def _cpp_transform_first_matching_line(
    text: str,
    transform: Callable[[str], str],
) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        body = line.rstrip()
        trailing = line[len(body):]
        transformed = transform(body) + trailing
        if transformed != line:
            return "".join((*lines[:index], transformed, *lines[index + 1:]))
    return text


def _cpp_parenthesize_return(text: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression:
        return text
    return f"{match.group(1)}({expression}){match.group(3)}"


def _cpp_comment_return(text: str, *, tag: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression:
        return text
    return f"{match.group(1)}/* wfcllm:{tag} */ {expression}{match.group(3)}"


def _cpp_ternary_return(text: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression:
        return text
    return f"{match.group(1)}(true ? ({expression}) : ({expression})){match.group(3)}"


def _cpp_comma_return(text: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression:
        return text
    return f"{match.group(1)}((void)0, {expression}){match.group(3)}"


def _cpp_sizeof_return(text: str) -> str:
    match = re.match(r"^(\s*return\s+)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression:
        return text
    return (
        f"{match.group(1)}((void)sizeof({expression}), {expression})"
        f"{match.group(3)}"
    )


def _cpp_parenthesize_condition(text: str) -> str:
    match = re.match(
        r"^(\s*(?:if|while|switch)\s*\()(.+?)(\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if match:
        condition = match.group(2).strip()
        if not condition:
            return text
        return f"{match.group(1)}({condition}){match.group(3)}"
    match = re.match(
        r"^(\s*for\s*\()(.*?;)(.*?)(;.*?\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    condition = match.group(3).strip()
    if not condition:
        return text
    return f"{match.group(1)}{match.group(2)} ({condition}){match.group(4)}"


def _cpp_comment_condition(text: str, *, tag: str) -> str:
    match = re.match(
        r"^(\s*(?:if|while|switch)\s*\()(.+?)(\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if match:
        condition = match.group(2).strip()
        if not condition:
            return text
        return f"{match.group(1)}/* wfcllm:{tag} */ {condition}{match.group(3)}"
    match = re.match(
        r"^(\s*for\s*\()(.*?;)(.*?)(;.*?\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    condition = match.group(3).strip()
    if not condition:
        return text
    return f"{match.group(1)}{match.group(2)} /* wfcllm:{tag} */ {condition}{match.group(4)}"


def _cpp_boolean_condition(text: str, *, operator: str) -> str:
    if operator not in {"&&", "||"}:
        raise ValueError("unsupported boolean condition operator")
    match = re.match(
        r"^(\s*(?:if|while)\s*\()(.+?)(\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if match:
        condition = match.group(2).strip()
        if not condition:
            return text
        identity = "true" if operator == "&&" else "false"
        return f"{match.group(1)}(({condition}) {operator} {identity}){match.group(3)}"
    match = re.match(
        r"^(\s*for\s*\()(.*?;)(.*?)(;.*?\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    condition = match.group(3).strip()
    if not condition:
        return text
    identity = "true" if operator == "&&" else "false"
    return f"{match.group(1)}{match.group(2)} (({condition}) {operator} {identity}){match.group(4)}"


def _cpp_comma_condition(text: str) -> str:
    match = re.match(
        r"^(\s*(?:if|while|switch)\s*\()(.+?)(\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if match:
        condition = match.group(2).strip()
        if not condition:
            return text
        return f"{match.group(1)}((void)0, {condition}){match.group(3)}"
    match = re.match(
        r"^(\s*for\s*\()(.*?;)(.*?)(;.*?\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    condition = match.group(3).strip()
    if not condition:
        return text
    return f"{match.group(1)}{match.group(2)} ((void)0, {condition}){match.group(4)}"


def _cpp_not_not_condition(text: str) -> str:
    match = re.match(
        r"^(\s*(?:if|while)\s*\()(.+?)(\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if match:
        condition = match.group(2).strip()
        if not condition:
            return text
        return f"{match.group(1)}(!!({condition})){match.group(3)}"
    match = re.match(
        r"^(\s*for\s*\()(.*?;)(.*?)(;.*?\)(?:\s*\{)?\s*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return text
    condition = match.group(3).strip()
    if not condition:
        return text
    return f"{match.group(1)}{match.group(2)} (!!({condition})){match.group(4)}"


def _cpp_parenthesize_initializer_or_assignment(text: str) -> str:
    match = re.match(r"^(\s*[^;=<>!]+?\s=\s*)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression or expression.startswith("{"):
        return text
    return f"{match.group(1)}({expression}){match.group(3)}"


def _cpp_comment_initializer_or_assignment(text: str, *, tag: str) -> str:
    match = re.match(r"^(\s*[^;=<>!]+?\s=\s*)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression or expression.startswith("{"):
        return text
    return f"{match.group(1)}/* wfcllm:{tag} */ {expression}{match.group(3)}"


def _cpp_ternary_initializer_or_assignment(text: str) -> str:
    match = re.match(r"^(\s*[^;=<>!]+?\s=\s*)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression or expression.startswith("{"):
        return text
    return f"{match.group(1)}(true ? ({expression}) : ({expression})){match.group(3)}"


def _cpp_comma_initializer_or_assignment(text: str) -> str:
    match = re.match(r"^(\s*[^;=<>!]+?\s=\s*)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression or expression.startswith("{"):
        return text
    return f"{match.group(1)}((void)0, {expression}){match.group(3)}"


def _cpp_sizeof_initializer_or_assignment(text: str) -> str:
    match = re.match(r"^(\s*[^;=<>!]+?\s=\s*)(.+?)(\s*;?\s*)$", text, re.DOTALL)
    if not match:
        return text
    expression = match.group(2).strip()
    if not expression or expression.startswith("{"):
        return text
    return (
        f"{match.group(1)}((void)sizeof({expression}), {expression})"
        f"{match.group(3)}"
    )


def python_comprehension_alpha_equivalent(
    reference: str,
    candidate: str,
) -> bool:
    """Prove equivalence modulo simple, comprehension-local target renaming."""

    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise ValueError("reference and candidate must be strings")
    try:
        reference_tree = ast.parse(textwrap.dedent(reference), type_comments=True)
        candidate_tree = ast.parse(textwrap.dedent(candidate), type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return False
    reference_normalizer = _ComprehensionAlphaNormalizer()
    candidate_normalizer = _ComprehensionAlphaNormalizer()
    reference_tree = reference_normalizer.visit(reference_tree)
    candidate_tree = candidate_normalizer.visit(candidate_tree)
    if (
        not reference_normalizer.supported
        or not candidate_normalizer.supported
        or _docstring_values(reference_tree) != _docstring_values(candidate_tree)
    ):
        return False
    ast.fix_missing_locations(reference_tree)
    ast.fix_missing_locations(candidate_tree)
    return ast.dump(reference_tree, include_attributes=False) == ast.dump(
        candidate_tree,
        include_attributes=False,
    )


def _ast_equivalent_variant(text: str, *, candidate_index: int) -> str:
    if not text.strip():
        return text

    mode = (candidate_index - 1) % 3
    if mode == 0:
        transformed = _rewrite_first_string(text)
    elif mode == 1:
        transformed = _rewrite_first_integer(text)
    else:
        transformed = _rewrite_first_string(text)
        transformed = _rewrite_first_integer(transformed)
    transformed = _parenthesize_first_window_expression(
        transformed,
        depth=candidate_index,
    )

    if not python_ast_equivalent(text, transformed):
        return text
    return transformed


@dataclass(frozen=True)
class _WrappedWindowSource:
    """A parseable synthetic function around an exact parser window slice."""

    header: str
    prepared: str
    injected_indent: str
    injection_mode: str
    suffix: str
    tree: ast.Module

    @property
    def source(self) -> str:
        return self.header + self.prepared + self.suffix

    def unwrap(self, transformed_source: str) -> str:
        if not transformed_source.startswith(self.header):
            raise ValueError("transformed window lost its synthetic header")
        prepared = transformed_source[len(self.header) :]
        if self.suffix:
            if not prepared.endswith(self.suffix):
                raise ValueError("transformed window lost its synthetic body")
            prepared = prepared[: -len(self.suffix)]
        if self.injection_mode == "none":
            return prepared
        if self.injection_mode == "first":
            if not prepared.startswith(self.injected_indent):
                raise ValueError("transformed window lost its first-line indent")
            return prepared[len(self.injected_indent) :]
        if self.injection_mode == "all":
            output: list[str] = []
            for line in prepared.splitlines(keepends=True):
                if line.strip():
                    if not line.startswith(self.injected_indent):
                        raise ValueError(
                            "transformed window lost its synthetic indentation"
                        )
                    line = line[len(self.injected_indent) :]
                output.append(line)
            return "".join(output)
        raise ValueError("unknown synthetic window injection mode")


def _wrap_window_source(text: str) -> _WrappedWindowSource:
    """Make a parser window independently parseable without changing its AST.

    Parser byte spans begin at the first statement token, so the first line of
    a function-body window has no leading indentation while later source lines
    retain it.  A synthetic function restores only the indentation omitted by
    that byte-span convention.  An isolated compound header also receives a
    synthetic ``pass`` solely outside the returned candidate text.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("window text must be non-empty")
    lines = text.splitlines(keepends=True)
    first_line = next((line for line in lines if line.strip()), "")
    first_indent = first_line[: len(first_line) - len(first_line.lstrip(" \t"))]
    subsequent_indents = [
        line[: len(line) - len(line.lstrip(" \t"))]
        for line in lines[1:]
        if line.strip()
    ]
    header = "def __wfcllm_synthetic_window__():\n"
    if first_indent:
        injected_indent = first_indent
        injection_mode = "none"
        prepared = text
    elif any(not indent for indent in subsequent_indents):
        injected_indent = "    "
        injection_mode = "all"
        prepared = "".join(
            injected_indent + line if line.strip() else line
            for line in lines
        )
    else:
        injected_indent = (
            min(subsequent_indents, key=len)
            if subsequent_indents
            else "    "
        )
        injection_mode = "first"
        prepared = injected_indent + text

    suffix = ""
    source = header + prepared
    try:
        tree = ast.parse(source, type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        if not text.rstrip().endswith(":"):
            raise
        suffix = (
            ("" if source.endswith(("\n", "\r")) else "\n")
            + injected_indent
            + "    pass\n"
        )
        tree = ast.parse(source + suffix, type_comments=True)
    return _WrappedWindowSource(
        header=header,
        prepared=prepared,
        injected_indent=injected_indent,
        injection_mode=injection_mode,
        suffix=suffix,
        tree=tree,
    )


def _parenthesize_first_window_expression(text: str, *, depth: int) -> str:
    try:
        wrapped = _wrap_window_source(text)
    except (SyntaxError, ValueError, TypeError):
        return text
    transformed = _parenthesize_first_expression(wrapped.source, depth=depth)
    try:
        return wrapped.unwrap(transformed)
    except ValueError:
        return text


def _comprehension_alpha_variant(text: str, *, candidate_index: int) -> str:
    prefix = _common_indentation(text)
    dedented = textwrap.dedent(text)
    try:
        tree = ast.parse(dedented, type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return text
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        and _simple_comprehension_binding(node) is not None
    ]
    if not candidates:
        return text
    selected = candidates[(candidate_index - 4) % len(candidates)]
    old_name = _simple_comprehension_binding(selected)
    if old_name is None:
        return text
    used_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    new_name = f"_wfcllm_comp_{candidate_index}"
    if new_name in used_names:
        return text
    _rename_simple_comprehension(selected, old_name, new_name)
    ast.fix_missing_locations(tree)
    try:
        transformed = ast.unparse(tree)
    except (TypeError, ValueError, RecursionError):
        return text
    if text.endswith(("\n", "\r")):
        transformed += "\n"
    candidate = _restore_indentation(transformed, prefix)
    return (
        candidate
        if python_comprehension_alpha_equivalent(text, candidate)
        else text
    )


def _literal_equivalent_variant(
    text: str,
    *,
    candidate_index: int,
    first_literal_index: int = 13,
) -> str:
    prefix = _common_indentation(text)
    dedented = textwrap.dedent(text)
    try:
        tree = ast.parse(dedented, type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return text
    literal_trajectory_index = candidate_index - first_literal_index
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


_COMPREHENSION_TYPES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_NESTED_SCOPE_TYPES = (
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _simple_comprehension_binding(node: ast.AST) -> str | None:
    generators = getattr(node, "generators", None)
    if not isinstance(generators, list) or len(generators) != 1:
        return None
    generator = generators[0]
    if (
        not isinstance(generator, ast.comprehension)
        or not isinstance(generator.target, ast.Name)
        or generator.is_async
    ):
        return None
    fields = _comprehension_bound_fields(node, generator)
    if any(
        isinstance(descendant, _NESTED_SCOPE_TYPES)
        for field in fields
        for descendant in ast.walk(field)
    ):
        return None
    return generator.target.id


def _comprehension_bound_fields(
    node: ast.AST,
    generator: ast.comprehension,
) -> tuple[ast.AST, ...]:
    values: list[ast.AST] = list(generator.ifs)
    if isinstance(node, ast.DictComp):
        values.extend((node.key, node.value))
    else:
        values.append(node.elt)
    return tuple(values)


def _rename_simple_comprehension(
    node: ast.AST,
    old_name: str,
    new_name: str,
) -> None:
    generator = node.generators[0]
    generator.target.id = new_name
    renamer = _MatchingLoadNameRenamer(old_name, new_name)
    for field in _comprehension_bound_fields(node, generator):
        renamer.visit(field)


class _MatchingLoadNameRenamer(ast.NodeTransformer):
    def __init__(self, old_name: str, new_name: str) -> None:
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id == self.old_name:
            node.id = self.new_name
        return node


class _ComprehensionAlphaNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.count = 0
        self.supported = True

    def _normalize(self, node: ast.AST):
        old_name = _simple_comprehension_binding(node)
        if old_name is None:
            self.supported = False
            return self.generic_visit(node)
        placeholder = f"_wfcllm_normalized_comp_{self.count}"
        self.count += 1
        generator = node.generators[0]
        generator.iter = self.visit(generator.iter)
        _rename_simple_comprehension(node, old_name, placeholder)
        for field in _comprehension_bound_fields(node, generator):
            self.visit(field)
        return node

    def visit_ListComp(self, node: ast.ListComp):
        return self._normalize(node)

    def visit_SetComp(self, node: ast.SetComp):
        return self._normalize(node)

    def visit_DictComp(self, node: ast.DictComp):
        return self._normalize(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        return self._normalize(node)


def _preserve_trailing_layout(reference: str, candidate: str) -> str:
    trailing = reference[len(reference.rstrip(" \t\r\n")) :]
    if not trailing:
        return candidate
    return candidate.rstrip(" \t\r\n") + trailing


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
        tree = _parse_ast_window(text)
    except (SyntaxError, ValueError, TypeError):
        return text
    source_line_count = len(text.splitlines())
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.expr)
        and hasattr(node, "lineno")
        and hasattr(node, "end_lineno")
        and node.end_lineno <= source_line_count
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


def _descriptor(
    unit: Any,
    window_contract_version: str = WINDOW_CONTRACT_VERSION,
) -> ParentDescriptor:
    return ParentDescriptor(
        contract_version=window_contract_version,
        ancestor_node_types=_descriptor_ancestor_node_types(unit),
        direct_parent_type=unit.direct_parent_type,
        first_unit_ordinal=unit.direct_child_ordinal,
        compound_header_role="header" if unit.compound_header else "body",
    )


def _descriptor_ancestor_node_types(unit: Any) -> tuple[str, ...]:
    ancestor_node_types = unit.parent_path[:-1]
    while (
        ancestor_node_types
        and ancestor_node_types[-1] == unit.direct_parent_type
    ):
        ancestor_node_types = ancestor_node_types[:-1]
    return ancestor_node_types


__all__ = [
    "CausalWindowRewriter",
    "KeyBlindAstEquivalentWindowRewriter",
    "KeyBlindWhitespaceWindowRewriter",
    "KeyBlindCppEquivalentWindowRewriter",
    "KeyBlindJavaEquivalentWindowRewriter",
    "ParsedRewrite",
    "RewriteGeneration",
    "python_ast_equivalent",
    "python_literal_equivalent",
]
