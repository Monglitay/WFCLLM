"""Key-blind causal adapter for complete semantic-window rewrites."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import token
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
        prefix_bytes = len(request.completed_prefix.encode("utf-8"))
        complete_source = request.completed_prefix + generation.text
        units = [
            unit
            for unit in self._extractor.extract(complete_source)
            if unit.start_byte >= prefix_bytes
        ]
        if not generation.text or not units:
            parse_status = "parse_error"
        elif (
            len(units) not in {1, 2, 3}
            or len(units) != request.window_length
        ):
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
    "KeyBlindWhitespaceWindowRewriter",
    "ParsedRewrite",
    "RewriteGeneration",
]
