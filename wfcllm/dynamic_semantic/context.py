from __future__ import annotations

import ast
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from wfcllm.dynamic_semantic.config import ContextConfig


_CONTEXT_HEADER = "WFCLLM_DYNAMIC_SEMANTIC_CONTEXT_V3"
_BOS = "<BOS>"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalContext:
    unit_id: str
    role: str
    canonical_previous: str
    canonical_current: str
    serialized: str
    context_sha256: str
    group_index: int
    unit_position: int
    is_last_in_group: bool


@dataclass(frozen=True)
class ContextExtraction:
    parse_ok: bool
    contexts: tuple[CanonicalContext, ...]
    erasure_counts: dict[str, int]


class DynamicContextExtractor:
    """Build public, bounded, final-code-reconstructible Python contexts."""

    def __init__(
        self,
        config: ContextConfig,
        *,
        token_counter: Callable[[str], int],
    ) -> None:
        self._config = config
        self._token_counter = token_counter
        self._token_count_cache: dict[str, int] = {}

    def _count_tokens(self, text: str) -> int:
        count = self._token_count_cache.get(text)
        if count is None:
            count = self._token_counter(text)
            self._token_count_cache[text] = count
        return count

    def extract(self, code: str) -> ContextExtraction:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, TypeError):
            return ContextExtraction(
                parse_ok=False,
                contexts=(),
                erasure_counts={"parse_failure": 1},
            )

        candidates: list[tuple[int, int, int, str, str, str]] = []
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for group_index, function in enumerate(functions):
            previous = _BOS
            group_size = len(function.body)
            parent_type = type(function).__name__
            for unit_position, statement in enumerate(function.body):
                current = ast.unparse(statement).strip().replace("\r\n", "\n")
                role = f"{type(statement).__name__}|{parent_type}|body"
                unit_id = _sha256(
                    f"{self._config.schema_version}\0{role}\0{current}"
                )
                candidates.append(
                    (
                        group_index,
                        unit_position,
                        group_size,
                        role,
                        previous,
                        current,
                    )
                )
                previous = current

        unit_ids = [
            _sha256(f"{self._config.schema_version}\0{role}\0{current}")
            for _, _, _, role, _, current in candidates
        ]
        duplicate_ids = {
            unit_id for unit_id, count in Counter(unit_ids).items() if count > 1
        }
        erasures: Counter[str] = Counter()
        contexts: list[CanonicalContext] = []
        for candidate, unit_id in zip(candidates, unit_ids, strict=True):
            group_index, unit_position, group_size, role, previous, current = candidate
            if unit_id in duplicate_ids:
                erasures["duplicate_unit_id"] += 1
                continue
            if self._count_tokens(current) > self._config.max_current_unit_tokens:
                erasures["current_unit_too_long"] += 1
                continue
            serialized = self._serialize(role, previous, current)
            if self._count_tokens(serialized) > self._config.max_context_tokens:
                erasures["context_too_long"] += 1
                continue
            contexts.append(
                CanonicalContext(
                    unit_id=unit_id,
                    role=role,
                    canonical_previous=previous,
                    canonical_current=current,
                    serialized=serialized,
                    context_sha256=_sha256(serialized),
                    group_index=group_index,
                    unit_position=unit_position,
                    is_last_in_group=unit_position == group_size - 1,
                )
            )
        return ContextExtraction(
            parse_ok=True,
            contexts=tuple(contexts),
            erasure_counts=dict(sorted(erasures.items())),
        )

    @staticmethod
    def _serialize(role: str, previous: str, current: str) -> str:
        return (
            f"{_CONTEXT_HEADER}\n"
            f"role={role}\n"
            f"previous={previous}\n"
            f"current={current}"
        )


class IncrementalContextTracker:
    """Track live, stable contexts from public generation prefixes."""

    def __init__(self, extractor: DynamicContextExtractor) -> None:
        self._extractor = extractor
        self._live_contexts: tuple[CanonicalContext, ...] = ()
        self._emitted_context_hashes: set[str] = set()

    @property
    def live_contexts(self) -> tuple[CanonicalContext, ...]:
        return self._live_contexts

    def observe(self, code: str) -> tuple[CanonicalContext, ...]:
        return self._update(code, include_last=False)

    def flush(self, code: str) -> tuple[CanonicalContext, ...]:
        return self._update(code, include_last=True)

    def _update(
        self,
        code: str,
        *,
        include_last: bool,
    ) -> tuple[CanonicalContext, ...]:
        extraction = self._extractor.extract(code)
        if not extraction.parse_ok:
            return ()
        live = tuple(
            context
            for context in extraction.contexts
            if include_last or not context.is_last_in_group
        )
        self._live_contexts = live
        emitted = tuple(
            context
            for context in live
            if context.context_sha256 not in self._emitted_context_hashes
        )
        self._emitted_context_hashes.update(
            context.context_sha256 for context in emitted
        )
        return emitted
