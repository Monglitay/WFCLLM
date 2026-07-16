"""Online gated generation over complete parser-defined statement windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from wfcllm.gate.data import RewriteRequest, StructuralBoundary
from wfcllm.generation.gated_state import (
    GatedWindowRetryController,
    RetryAction,
    WindowAttempt,
)
from wfcllm.generation.generator import GeneratedToken
from wfcllm.semantic.window_lsh import SemanticWindowScorer
from wfcllm.windowing import (
    WINDOW_CONTRACT_VERSION,
    CloseReason,
    ParentDescriptor,
    PythonStatementUnitExtractor,
    SemanticWindow,
    WindowPartitioner,
)


@dataclass(frozen=True)
class ByteSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("byte span bounds must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("byte span must be non-empty")


@dataclass(frozen=True)
class RewriteTokens:
    token_ids: tuple[int, ...]
    text: str
    token_texts: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or any(
            type(value) is not int or value < 0 for value in self.token_ids
        ):
            raise ValueError("token_ids must be non-negative integers")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.token_texts is not None:
            if (
                not isinstance(self.token_texts, tuple)
                or len(self.token_texts) != len(self.token_ids)
                or any(not isinstance(value, str) for value in self.token_texts)
                or "".join(self.token_texts) != self.text
            ):
                raise ValueError("token_texts must exactly reconstruct text")


@dataclass(frozen=True)
class GatedWindowAudit:
    original_span: ByteSpan
    selected_span: ByteSpan
    rollback_anchor: int
    original_unit_count: int
    selected_unit_count: int
    selected_candidate_index: int
    original_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    parent_descriptor: str
    previous_statement_text_unchanged: bool
    close_reason: str


@dataclass(frozen=True)
class GatedGenerationResult:
    final_code: str
    audit: tuple[GatedWindowAudit, ...]


class WholeWindowRewriter(Protocol):
    def rewrite_window(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> RewriteTokens: ...


class GatedGenerator:
    """Select the first stable semantic hit for each complete gated window.

    Candidate zero is always the exact original byte slice.  Rewrites replace
    the entire window from its rollback anchor; no statement-level repair or
    quality ranking is performed.
    """

    def __init__(
        self,
        *,
        partitioner: WindowPartitioner,
        scorer: SemanticWindowScorer | Any,
        rewriter: WholeWindowRewriter | Any,
        max_rewrites: int,
        extractor: PythonStatementUnitExtractor | None = None,
        model_context: Any | None = None,
    ) -> None:
        if not isinstance(partitioner, WindowPartitioner):
            raise ValueError("partitioner must be a WindowPartitioner")
        if not callable(getattr(scorer, "score", None)):
            raise ValueError("scorer must define score")
        if not callable(getattr(rewriter, "rewrite_window", None)) and not callable(
            getattr(rewriter, "rewrite_generation", None)
        ):
            raise ValueError("rewriter must define rewrite_window or rewrite_generation")
        if type(max_rewrites) is not int or max_rewrites < 0:
            raise ValueError("max_rewrites must be a non-negative integer")
        self._partitioner = partitioner
        self._scorer = scorer
        self._rewriter = rewriter
        self._max_rewrites = max_rewrites
        self._extractor = extractor or PythonStatementUnitExtractor()
        self._model_context = model_context

    def generate(
        self,
        *,
        prompt: str,
        original: str,
        suffix: str = "",
        protected_prefix: str | None = None,
        original_token_ids: tuple[int, ...] = (),
    ) -> GatedGenerationResult:
        if not all(isinstance(value, str) for value in (prompt, original, suffix)):
            raise ValueError("prompt, original, and suffix must be strings")
        if protected_prefix is not None and not isinstance(protected_prefix, str):
            raise ValueError("protected_prefix must be a string")
        if not isinstance(original_token_ids, tuple) or any(
            type(value) is not int or value < 0 for value in original_token_ids
        ):
            raise ValueError("original_token_ids must be non-negative integers")

        units = self._extractor.extract(original)
        if not units:
            return GatedGenerationResult(original + suffix, ())
        partition = self._partitioner.partition(units)
        source = original.encode("utf-8")
        output = bytearray()
        cursor = 0
        audit: list[GatedWindowAudit] = []

        for window_index, window in enumerate(partition.windows):
            output.extend(source[cursor : window.start_byte])
            prefix = protected_prefix if window_index == 0 and protected_prefix else ""
            prefix_bytes = prefix.encode("utf-8")
            rollback_anchor = window.start_byte - len(prefix_bytes)
            if rollback_anchor < cursor or source[rollback_anchor : window.start_byte] != prefix_bytes:
                prefix = ""
                prefix_bytes = b""
                rollback_anchor = window.start_byte
            if rollback_anchor < window.start_byte:
                del output[-(window.start_byte - rollback_anchor) :]

            replacement_end = _consume_trailing_layout(source, window.end_byte)
            original_window = source[rollback_anchor:replacement_end].decode("utf-8")
            candidate_zero = RewriteTokens(original_token_ids, original_window)
            selected = candidate_zero
            selected_index = 0
            selected_units = tuple(window.units)

            if window.suitable and window.close_reason is not CloseReason.INPUT_OVERFLOW:
                original_evidence = self._score(
                    window_text=source[window.start_byte : window.end_byte].decode("utf-8"),
                    parent_descriptor=window.parent_descriptor.canonical,
                )
                controller = GatedWindowRetryController(self._max_rewrites)
                controller.begin(
                    _attempt(0, suitable=True, structure_ok=True, evidence=original_evidence)
                )
                request = _rewrite_request(
                    prompt=prompt,
                    completed_prefix=source[:rollback_anchor].decode("utf-8"),
                    original_window=original_window,
                    window=window,
                )
                while controller.next_action() is RetryAction.REWRITE:
                    candidate_index = controller.rewrite_count + 1
                    candidate = self._rewrite(request, candidate_index)
                    checked = self._validate_candidate(
                        completed_prefix=request.completed_prefix,
                        candidate=candidate,
                        protected_prefix=prefix,
                        original_window=window,
                    )
                    evidence = None
                    if checked is not None:
                        candidate_units, candidate_window = checked
                        evidence = self._score(
                            window_text="".join(unit.text for unit in candidate_units),
                            parent_descriptor=candidate_window.parent_descriptor.canonical,
                        )
                    controller.observe(
                        _attempt(
                            candidate_index,
                            suitable=checked is not None,
                            structure_ok=checked is not None,
                            evidence=evidence,
                        )
                    )
                    if controller.next_action() is RetryAction.ACCEPT_REWRITE:
                        assert checked is not None
                        selected = candidate
                        selected_index = candidate_index
                        selected_units = checked[0]
                        break

            output.extend(selected.text.encode("utf-8"))
            selected_start = len(output) - len(selected.text.encode("utf-8"))
            selected_end = len(output)
            audit.append(
                GatedWindowAudit(
                    original_span=ByteSpan(window.start_byte, window.end_byte),
                    selected_span=ByteSpan(selected_start, selected_end),
                    rollback_anchor=rollback_anchor,
                    original_unit_count=len(window.units),
                    selected_unit_count=len(selected_units),
                    selected_candidate_index=selected_index,
                    original_token_ids=candidate_zero.token_ids,
                    selected_token_ids=selected.token_ids,
                    parent_descriptor=_descriptor(selected_units[0]).canonical,
                    previous_statement_text_unchanged=(
                        bytes(output[:selected_start])
                        == source[:rollback_anchor]
                    ),
                    close_reason=window.close_reason.value,
                )
            )
            self._replay_if_available(rollback_anchor, selected)
            cursor = replacement_end

        output.extend(source[cursor:])
        output.extend(suffix.encode("utf-8"))
        final_code = output.decode("utf-8")
        # Reparse the committed prefix.  This is both the continuation cursor
        # reset and the source of ordinals/descriptors for subsequent stages.
        self._extractor.extract(final_code)
        return GatedGenerationResult(final_code, tuple(audit))

    def _rewrite(self, request: RewriteRequest, candidate_index: int) -> RewriteTokens:
        if callable(getattr(self._rewriter, "rewrite_window", None)):
            value = self._rewriter.rewrite_window(
                request, candidate_index=candidate_index
            )
        else:
            value = self._rewriter.rewrite_generation(
                request, candidate_index=candidate_index
            )
        if isinstance(value, RewriteTokens):
            return value
        token_ids = getattr(value, "token_ids", None)
        text = getattr(value, "text", None)
        if isinstance(token_ids, tuple) and isinstance(text, str):
            return RewriteTokens(token_ids, text)
        raise ValueError("rewriter must return an exact token/text candidate")

    def _validate_candidate(
        self,
        *,
        completed_prefix: str,
        candidate: RewriteTokens,
        protected_prefix: str,
        original_window: SemanticWindow,
    ) -> tuple[tuple[Any, ...], SemanticWindow] | None:
        if not candidate.text.startswith(protected_prefix):
            return None
        source = completed_prefix + candidate.text
        start_byte = len(completed_prefix.encode("utf-8")) + len(
            protected_prefix.encode("utf-8")
        )
        units = tuple(
            unit
            for unit in self._extractor.extract(source)
            if unit.start_byte >= start_byte
        )
        if not 1 <= len(units) <= 3 or any(unit.hard_boundary for unit in units):
            return None
        if any(
            unit.depth != units[0].depth
            or unit.parent_path != units[0].parent_path
            or unit.direct_parent_type != units[0].direct_parent_type
            for unit in units
        ):
            return None
        original_first = original_window.units[0]
        if (
            units[0].direct_parent_type != original_first.direct_parent_type
            or units[0].parent_path != original_first.parent_path
            or units[0].compound_header != original_first.compound_header
        ):
            return None
        if original_first.compound_header and units[0].node_type != original_first.node_type:
            return None
        partition = self._partitioner.partition(list(units))
        if len(partition.windows) != 1:
            return None
        closed = partition.windows[0]
        if tuple(closed.units) != units or not closed.suitable:
            return None
        return units, closed

    def _score(self, *, window_text: str, parent_descriptor: str) -> Any:
        evidence = self._scorer.score(
            window_text=window_text,
            parent_descriptor=parent_descriptor,
        )
        if not all(hasattr(evidence, name) for name in ("hit", "stable", "margin")):
            raise ValueError("semantic scorer returned incomplete evidence")
        return evidence

    def _replay_if_available(self, rollback_anchor: int, selected: RewriteTokens) -> None:
        if self._model_context is None or selected.token_texts is None:
            return
        rollback = self._model_context.checkpoint_for_generated_byte(rollback_anchor)
        steps = tuple(
            GeneratedToken(token_id, token_text)
            for token_id, token_text in zip(
                selected.token_ids, selected.token_texts, strict=True
            )
        )
        self._model_context.replay_from_checkpoint(
            rollback.checkpoint, replacement_steps=steps
        )


def _attempt(
    candidate_index: int,
    *,
    suitable: bool,
    structure_ok: bool,
    evidence: Any | None,
) -> WindowAttempt:
    return WindowAttempt(
        candidate_index=candidate_index,
        suitable=suitable,
        structure_ok=structure_ok,
        semantic_hit=bool(evidence is not None and evidence.hit),
        semantic_stable=bool(evidence is not None and evidence.stable),
        semantic_margin=float(evidence.margin if evidence is not None else 0.0),
    )


def _rewrite_request(
    *, prompt: str, completed_prefix: str, original_window: str, window: SemanticWindow
) -> RewriteRequest:
    first = window.units[0]
    return RewriteRequest(
        prompt=prompt,
        completed_prefix=completed_prefix,
        original_window=original_window,
        canonical_parent=window.parent_descriptor.canonical,
        window_start_unit_id=first.unit_id,
        window_length=len(window.units),
        structural_boundary=StructuralBoundary(
            start_byte=window.start_byte,
            end_byte=window.end_byte,
            depth=first.depth,
            direct_parent_type=first.direct_parent_type,
            unit_ids=tuple(unit.unit_id for unit in window.units),
            compound_singleton=first.compound_header,
            hard_boundary_after=False,
        ),
    )


def _descriptor(unit: Any) -> ParentDescriptor:
    return ParentDescriptor(
        contract_version=WINDOW_CONTRACT_VERSION,
        ancestor_node_types=unit.parent_path[:-1],
        direct_parent_type=unit.direct_parent_type,
        first_unit_ordinal=unit.direct_child_ordinal,
        compound_header_role="header" if unit.compound_header else "body",
    )


def _consume_trailing_layout(source: bytes, end_byte: int) -> int:
    """Consume only the newline layout that a complete rewrite reproduces."""

    cursor = end_byte
    while cursor < len(source) and source[cursor : cursor + 1] in {b"\r", b"\n"}:
        cursor += 1
    return cursor


__all__ = [
    "ByteSpan",
    "GatedGenerationResult",
    "GatedGenerator",
    "GatedWindowAudit",
    "RewriteTokens",
]
