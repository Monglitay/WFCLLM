"""Online gated generation over complete parser-defined statement windows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Protocol

from wfcllm.gate.data import (
    RewriteRequest,
    StructuralBoundary,
    compute_hard_boundary_after,
)
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

CERTIFIED_REWRITE_SEMANTIC_RULES = frozenset(
    {
        "python-ast-equivalent/v1",
        "python-comprehension-alpha-equivalent/v1",
        "python-literal-equivalent/v1",
        "cpp-keyblind-equivalent/v1",
        "java-keyblind-equivalent/v1",
    }
)
SEMANTIC_VALIDATION_RULES = frozenset(
    {"identity/v1", "encoder-cosine/v1", *CERTIFIED_REWRITE_SEMANTIC_RULES}
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
    generation_seed_id: str | None = None
    rewrite_config_id: str | None = None
    parse_status: str | None = None
    unit_count: int | None = None
    same_parent_scope: bool | None = None
    parser_spans: tuple[tuple[int, int], ...] | None = None
    semantic_validation_rule: str | None = None
    semantic_equivalence_certified: bool | None = None

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
        for name in ("generation_seed_id", "rewrite_config_id", "parse_status"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.unit_count is not None and (
            type(self.unit_count) is not int or self.unit_count < 0
        ):
            raise ValueError("unit_count must be a non-negative integer or None")
        if self.same_parent_scope is not None and not isinstance(
            self.same_parent_scope, bool
        ):
            raise ValueError("same_parent_scope must be bool or None")
        if self.parser_spans is not None and (
            not isinstance(self.parser_spans, tuple)
            or any(
                not isinstance(span, tuple)
                or len(span) != 2
                or any(type(value) is not int or value < 0 for value in span)
                or span[1] <= span[0]
                for span in self.parser_spans
            )
        ):
            raise ValueError("parser_spans must contain non-empty byte spans")
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
    semantic_reference_cosine: float
    semantic_preservation_passed: bool
    semantic_validation_rule: str = "encoder-cosine/v1"


@dataclass(frozen=True)
class GatedGenerationResult:
    final_code: str
    audit: tuple[GatedWindowAudit, ...]
    candidates: tuple[GatedCandidateAudit, ...] = ()


@dataclass(frozen=True)
class _CombinedSemanticEvidence:
    hit: bool
    stable: bool
    margin: float
    signature: tuple[int, ...]


@dataclass(frozen=True)
class GatedCandidateAudit:
    """One public candidate attempt; never used as detector input."""

    window_index: int
    candidate_index: int
    requested_unit_count: int
    text: str
    token_ids: tuple[int, ...]
    generation_seed_id: str
    rewrite_config_id: str
    parse_status: str
    unit_count: int
    same_parent_scope: bool
    parser_spans: tuple[tuple[int, int], ...]
    structure_ok: bool | None
    semantic_reference_cosine: float | None
    semantic_preservation_passed: bool | None
    keyed_lsh_scored: bool
    lsh_signature: tuple[int, ...] | None
    semantic_hit: bool | None
    semantic_stable: bool | None
    semantic_margin: float | None
    selected: bool
    evaluation_status: str
    semantic_validation_rule: str | None = None

    def __post_init__(self) -> None:
        if type(self.window_index) is not int or self.window_index < 0:
            raise ValueError("window_index must be a non-negative integer")
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise ValueError("candidate_index must be a non-negative integer")
        if type(self.requested_unit_count) is not int or not 1 <= self.requested_unit_count <= 3:
            raise ValueError("requested_unit_count must be an integer in [1, 3]")
        if not isinstance(self.text, str):
            raise ValueError("candidate text must be a string")
        if not isinstance(self.token_ids, tuple) or any(
            type(value) is not int or value < 0 for value in self.token_ids
        ):
            raise ValueError("candidate token_ids must be non-negative integers")
        for name in ("generation_seed_id", "rewrite_config_id", "parse_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"candidate {name} must be non-empty")
        if self.candidate_index == 0 and (
            self.generation_seed_id != "candidate-zero/original"
            or self.rewrite_config_id != "candidate-zero/original/v1"
        ):
            raise ValueError("candidate zero must use original provenance identities")
        if type(self.unit_count) is not int or self.unit_count < 0:
            raise ValueError("candidate unit_count must be non-negative")
        if not isinstance(self.same_parent_scope, bool):
            raise ValueError("candidate same_parent_scope must be bool")
        if (
            not isinstance(self.parser_spans, tuple)
            or any(
                not isinstance(span, tuple)
                or len(span) != 2
                or any(type(value) is not int or value < 0 for value in span)
                or span[1] <= span[0]
                for span in self.parser_spans
            )
        ):
            raise ValueError("candidate parser_spans must contain non-empty spans")
        if self.structure_ok is not None and not isinstance(self.structure_ok, bool):
            raise ValueError("structure_ok must be bool or None")
        cosine = self.semantic_reference_cosine
        passed = self.semantic_preservation_passed
        if (cosine is None) != (passed is None):
            raise ValueError("semantic cosine and decision must be present together")
        semantic_rule = self.semantic_validation_rule
        if passed is not None and semantic_rule is None:
            semantic_rule = "encoder-cosine/v1"
        if passed is None and semantic_rule is not None:
            raise ValueError("unvalidated candidate must not name a semantic rule")
        if cosine is not None:
            if (
                isinstance(cosine, bool)
                or not isinstance(cosine, (int, float))
                or not math.isfinite(float(cosine))
                or not -1.0 <= float(cosine) <= 1.0
            ):
                raise ValueError("semantic cosine must be finite and in [-1, 1]")
            if not isinstance(passed, bool):
                raise ValueError("semantic decision must be a bool")
            if semantic_rule not in SEMANTIC_VALIDATION_RULES:
                raise ValueError("unknown semantic validation rule")
            if semantic_rule == "encoder-cosine/v1" and passed is not (
                float(cosine) >= 0.90
            ):
                raise ValueError("encoder semantic decision must equal cosine >= 0.90")
            if semantic_rule == "identity/v1" and (
                passed is not True or not math.isclose(float(cosine), 1.0)
            ):
                raise ValueError("identity candidate must have unit cosine and pass")
        if not isinstance(self.keyed_lsh_scored, bool):
            raise ValueError("keyed_lsh_scored must be a bool")
        keyed_values = (
            self.lsh_signature,
            self.semantic_hit,
            self.semantic_stable,
            self.semantic_margin,
        )
        if self.keyed_lsh_scored:
            if (
                not isinstance(self.lsh_signature, tuple)
                or not self.lsh_signature
                or any(
                    type(value) is not int or value not in (0, 1)
                    for value in self.lsh_signature
                )
                or not isinstance(self.semantic_hit, bool)
                or not isinstance(self.semantic_stable, bool)
                or isinstance(self.semantic_margin, bool)
                or not isinstance(self.semantic_margin, (int, float))
                or not math.isfinite(float(self.semantic_margin))
                or float(self.semantic_margin) < 0.0
            ):
                raise ValueError("keyed LSH audit fields must be complete and valid")
        elif any(value is not None for value in keyed_values):
            raise ValueError("unscored candidates must not carry keyed LSH evidence")
        if not isinstance(self.selected, bool):
            raise ValueError("selected must be a bool")
        allowed_statuses = {
            "evaluated_original",
            "structure_rejected",
            "semantic_rejected",
            "keyed_lsh_evaluated",
            "fast_certified_rewrite_selected",
            "generated_not_evaluated_after_accept",
            "gate_skipped",
        }
        if self.evaluation_status not in allowed_statuses:
            raise ValueError("unknown candidate evaluation_status")
        expected_shape = {
            "evaluated_original": (True, True, True),
            "structure_rejected": (False, None, False),
            "semantic_rejected": (True, False, False),
            "keyed_lsh_evaluated": (True, True, True),
            "fast_certified_rewrite_selected": (True, True, True),
            "generated_not_evaluated_after_accept": (None, None, False),
            "gate_skipped": (True, True, False),
        }[self.evaluation_status]
        if (
            self.structure_ok,
            self.semantic_preservation_passed,
            self.keyed_lsh_scored,
        ) != expected_shape:
            raise ValueError("candidate fields contradict evaluation_status")
        parser_structure_ok = (
            self.parse_status == "ok"
            and self.same_parent_scope is True
            and self.unit_count == self.requested_unit_count
        )
        if self.structure_ok is True and not parser_structure_ok:
            raise ValueError("candidate parser facts contradict structure_ok")
        if self.selected and self.evaluation_status == "generated_not_evaluated_after_accept":
            raise ValueError("an unevaluated candidate cannot be selected")
        if (
            self.selected
            and self.candidate_index > 0
            and self.evaluation_status == "keyed_lsh_evaluated"
            and (self.semantic_hit is not True or self.semantic_stable is not True)
        ):
            raise ValueError("a selected rewrite must be a stable semantic hit")
        if (
            self.selected
            and self.candidate_index > 0
            and self.evaluation_status == "fast_certified_rewrite_selected"
            and (
                self.semantic_preservation_passed is not True
                or self.semantic_validation_rule not in CERTIFIED_REWRITE_SEMANTIC_RULES
            )
        ):
            raise ValueError("a fast selected rewrite must be certified")


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
        evidence_channels: int = 1,
        extractor: Any | None = None,
        model_context: Any | None = None,
        accept_certified_rewrite_without_hit: bool = False,
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
        if (
            isinstance(evidence_channels, bool)
            or not isinstance(evidence_channels, int)
            or not 1 <= evidence_channels <= 4
        ):
            raise ValueError("evidence_channels must be an integer in [1, 4]")
        if evidence_channels > 1 and not callable(
            getattr(scorer, "score_channels", None)
        ):
            raise ValueError(
                "multi-channel scorer must define score_channels"
            )
        self._partitioner = partitioner
        self._scorer = scorer
        self._rewriter = rewriter
        self._max_rewrites = max_rewrites
        self._evidence_channels = evidence_channels
        self._extractor = extractor or PythonStatementUnitExtractor()
        self._window_contract_version = partitioner.window_contract_version
        self._model_context = model_context
        self._accept_certified_rewrite_without_hit = bool(
            accept_certified_rewrite_without_hit
        )

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
        candidate_audit: list[GatedCandidateAudit] = []
        unit_index_by_id = {unit.unit_id: index for index, unit in enumerate(units)}

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
            rewrite_window = source[rollback_anchor:window.end_byte].decode("utf-8")
            candidate_zero = RewriteTokens(
                original_token_ids,
                rewrite_window,
                generation_seed_id="candidate-zero/original",
                rewrite_config_id="candidate-zero/original/v1",
                parse_status="ok",
                unit_count=len(window.units),
                same_parent_scope=True,
                parser_spans=tuple(
                    (unit.start_byte, unit.end_byte) for unit in window.units
                ),
                semantic_validation_rule="identity/v1",
                semantic_equivalence_certified=True,
            )
            selected = candidate_zero
            selected_index = 0
            selected_units = tuple(window.units)
            selected_semantic_cosine = 1.0
            selected_semantic_preserved = True
            selected_semantic_validation_rule = "identity/v1"
            window_candidate_audit: list[GatedCandidateAudit] = []

            if window.suitable and window.close_reason is not CloseReason.INPUT_OVERFLOW:
                original_semantic_text = source[
                    window.start_byte : window.end_byte
                ].decode("utf-8")
                original_evidence = self._score(
                    window_text=original_semantic_text,
                    parent_descriptor=window.parent_descriptor.canonical,
                )
                controller = GatedWindowRetryController(self._max_rewrites)
                controller.begin(
                    _attempt(0, suitable=True, structure_ok=True, evidence=original_evidence)
                )
                window_candidate_audit.append(
                    _candidate_audit(
                        window_index=window_index,
                        candidate_index=0,
                        requested_unit_count=len(window.units),
                        candidate=candidate_zero,
                        structure_ok=True,
                        preservation_cosine=1.0,
                        preservation_passed=True,
                        semantic_validation_rule="identity/v1",
                        evidence=original_evidence,
                        evaluation_status="evaluated_original",
                    )
                )
                request = _rewrite_request(
                    prompt=prompt,
                    completed_prefix=source[:rollback_anchor].decode("utf-8"),
                    original_window=rewrite_window,
                    window=window,
                    hard_boundary_after=compute_hard_boundary_after(
                        units,
                        start_index=unit_index_by_id[window.units[0].unit_id],
                        window=tuple(window.units),
                    ),
                )
                generated_trajectory = (
                    self._rewrite_trajectory(request)
                    if self._max_rewrites == 3
                    else None
                )
                while controller.next_action() is RetryAction.REWRITE:
                    candidate_index = controller.rewrite_count + 1
                    candidate = (
                        generated_trajectory[candidate_index - 1]
                        if generated_trajectory is not None
                        else self._rewrite(request, candidate_index)
                    )
                    checked = self._validate_candidate(
                        completed_prefix=request.completed_prefix,
                        candidate=candidate,
                        protected_prefix=prefix,
                        original_window=window,
                    )
                    evidence = None
                    semantic_preservation_cosine = None
                    semantic_preservation_passed = None
                    semantic_validation_rule = None
                    if checked is not None:
                        candidate_units, candidate_window = checked
                        candidate_semantic_text = "".join(
                            unit.text for unit in candidate_units
                        )
                        semantic_comparison = self._compare_semantics(
                            reference_text=original_semantic_text,
                            candidate_text=candidate_semantic_text,
                        )
                        semantic_preservation_cosine = float(
                            semantic_comparison.cosine
                        )
                        candidate_rule = candidate.semantic_validation_rule
                        if candidate_rule is None:
                            semantic_validation_rule = "encoder-cosine/v1"
                            semantic_preservation_passed = bool(
                                semantic_comparison.passed
                            )
                        elif candidate_rule in CERTIFIED_REWRITE_SEMANTIC_RULES:
                            semantic_validation_rule = candidate_rule
                            semantic_preservation_passed = bool(
                                candidate.semantic_equivalence_certified
                            )
                        else:
                            raise ValueError(
                                f"unsupported rewrite semantic validation rule: {candidate_rule}"
                            )
                        if semantic_preservation_passed:
                            evidence = self._score(
                                window_text=candidate_semantic_text,
                                parent_descriptor=window.parent_descriptor.canonical,
                            )
                    force_accept = (
                        self._accept_certified_rewrite_without_hit
                        and checked is not None
                        and semantic_preservation_passed is True
                        and candidate.text != rewrite_window
                    )
                    controller.observe(
                        _attempt(
                            candidate_index,
                            suitable=checked is not None,
                            structure_ok=checked is not None,
                            evidence=(
                                _forced_hit_evidence(evidence)
                                if force_accept
                                else evidence
                            ),
                        )
                    )
                    window_candidate_audit.append(
                        _candidate_audit(
                            window_index=window_index,
                            candidate_index=candidate_index,
                            requested_unit_count=len(window.units),
                            candidate=candidate,
                            structure_ok=checked is not None,
                            preservation_cosine=(
                                semantic_preservation_cosine
                            ),
                            preservation_passed=(
                                semantic_preservation_passed
                            ),
                            semantic_validation_rule=semantic_validation_rule,
                            evidence=evidence,
                            evaluation_status=(
                                "structure_rejected"
                                if checked is None
                                else "semantic_rejected"
                                if semantic_preservation_passed is False
                                else "fast_certified_rewrite_selected"
                                if force_accept
                                else "keyed_lsh_evaluated"
                            ),
                        )
                    )
                    if controller.next_action() is RetryAction.ACCEPT_REWRITE:
                        assert checked is not None
                        selected = candidate
                        selected_index = candidate_index
                        selected_units = checked[0]
                        assert semantic_preservation_cosine is not None
                        selected_semantic_cosine = float(
                            semantic_preservation_cosine
                        )
                        selected_semantic_preserved = True
                        assert semantic_validation_rule is not None
                        selected_semantic_validation_rule = semantic_validation_rule
                        break
                if generated_trajectory is not None:
                    observed_indices = {
                        item.candidate_index for item in window_candidate_audit
                    }
                    for candidate_index, candidate in enumerate(
                        generated_trajectory, 1
                    ):
                        if candidate_index in observed_indices:
                            continue
                        window_candidate_audit.append(
                            _candidate_audit(
                                window_index=window_index,
                                candidate_index=candidate_index,
                                requested_unit_count=len(window.units),
                                candidate=candidate,
                                structure_ok=None,
                                preservation_cosine=None,
                                preservation_passed=None,
                                semantic_validation_rule=None,
                                evidence=None,
                                evaluation_status=(
                                    "generated_not_evaluated_after_accept"
                                ),
                            )
                        )

            if not window_candidate_audit:
                window_candidate_audit.append(
                    _candidate_audit(
                        window_index=window_index,
                        candidate_index=0,
                        requested_unit_count=len(window.units),
                        candidate=candidate_zero,
                        structure_ok=True,
                        preservation_cosine=1.0,
                        preservation_passed=True,
                        semantic_validation_rule="identity/v1",
                        evidence=None,
                        evaluation_status="gate_skipped",
                    )
                )
            candidate_audit.extend(
                replace(item, selected=item.candidate_index == selected_index)
                for item in sorted(
                    window_candidate_audit,
                    key=lambda value: value.candidate_index,
                )
            )

            trailing_layout = source[window.end_byte:replacement_end]
            selected_text = (
                selected.text.rstrip(" \t\r\n") if trailing_layout else selected.text
            )
            selected_bytes = selected_text.encode("utf-8")
            output.extend(selected_bytes)
            selected_start = len(output) - len(selected_bytes)
            selected_end = len(output)
            output.extend(trailing_layout)
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
                    parent_descriptor=_descriptor(
                        selected_units[0], self._window_contract_version
                    ).canonical,
                    previous_statement_text_unchanged=(
                        bytes(output[:selected_start])
                        == source[:rollback_anchor]
                    ),
                    close_reason=window.close_reason.value,
                    semantic_reference_cosine=selected_semantic_cosine,
                    semantic_preservation_passed=selected_semantic_preserved,
                    semantic_validation_rule=selected_semantic_validation_rule,
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
        return GatedGenerationResult(
            final_code, tuple(audit), tuple(candidate_audit)
        )

    def _rewrite_trajectory(
        self, request: RewriteRequest
    ) -> tuple[RewriteTokens, ...] | None:
        rewrite_many = getattr(self._rewriter, "rewrite_windows", None)
        if not callable(rewrite_many):
            return None
        values = rewrite_many(request, candidate_indices=(1, 2, 3))
        if not isinstance(values, tuple) or len(values) != 3:
            raise ValueError("rewriter must return the exact A/B/C trajectory")
        output: list[RewriteTokens] = []
        for value in values:
            if isinstance(value, RewriteTokens):
                output.append(value)
                continue
            token_ids = getattr(value, "token_ids", None)
            text = getattr(value, "text", None)
            if not isinstance(token_ids, tuple) or not isinstance(text, str):
                raise ValueError("rewriter trajectory contains an invalid candidate")
            output.append(
                RewriteTokens(
                    token_ids,
                    text,
                    generation_seed_id=getattr(value, "generation_seed_id", None),
                    rewrite_config_id=getattr(value, "rewrite_config_id", None),
                    parse_status=getattr(value, "parse_status", None),
                    unit_count=getattr(value, "unit_count", None),
                    same_parent_scope=getattr(value, "same_parent_scope", None),
                    parser_spans=getattr(value, "parser_spans", None),
                    semantic_validation_rule=getattr(
                        value, "semantic_validation_rule", None
                    ),
                    semantic_equivalence_certified=getattr(
                        value, "semantic_equivalence_certified", None
                    ),
                )
            )
        return tuple(output)

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
            return RewriteTokens(
                token_ids,
                text,
                generation_seed_id=getattr(value, "generation_seed_id", None),
                rewrite_config_id=getattr(value, "rewrite_config_id", None),
                parse_status=getattr(value, "parse_status", None),
                unit_count=getattr(value, "unit_count", None),
                same_parent_scope=getattr(value, "same_parent_scope", None),
                parser_spans=getattr(value, "parser_spans", None),
                semantic_validation_rule=getattr(
                    value, "semantic_validation_rule", None
                ),
                semantic_equivalence_certified=getattr(
                    value, "semantic_equivalence_certified", None
                ),
            )
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
        if (
            len(units) != len(original_window.units)
            or any(unit.hard_boundary for unit in units)
        ):
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
        if self._evidence_channels == 1:
            evidence = self._scorer.score(
                window_text=window_text,
                parent_descriptor=parent_descriptor,
            )
        else:
            channels = self._scorer.score_channels(
                window_text=window_text,
                parent_descriptor=parent_descriptor,
                channel_count=self._evidence_channels,
            )
            if (
                not isinstance(channels, tuple)
                or len(channels) != self._evidence_channels
                or any(
                    not all(
                        hasattr(item, name)
                        for name in ("hit", "stable", "margin", "signature")
                    )
                    for item in channels
                )
            ):
                raise ValueError(
                    "semantic channel evidence count does not match config"
                )
            stable = all(bool(item.stable) for item in channels)
            evidence = _CombinedSemanticEvidence(
                hit=stable and all(bool(item.hit) for item in channels),
                stable=stable,
                margin=min(float(item.margin) for item in channels),
                signature=tuple(channels[0].signature),
            )
        if not all(hasattr(evidence, name) for name in ("hit", "stable", "margin")):
            raise ValueError("semantic scorer returned incomplete evidence")
        return evidence

    def _compare_semantics(
        self, *, reference_text: str, candidate_text: str
    ) -> Any:
        compare = getattr(self._scorer, "compare_semantics", None)
        if not callable(compare):
            raise ValueError("semantic scorer must define compare_semantics")
        evidence = compare(
            reference_text=reference_text,
            candidate_text=candidate_text,
        )
        if not hasattr(evidence, "cosine") or not isinstance(
            getattr(evidence, "passed", None), bool
        ):
            raise ValueError(
                "semantic scorer returned incomplete preservation evidence"
            )
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


def _forced_hit_evidence(evidence: Any | None) -> Any:
    if evidence is None:
        return type(
            "ForcedEvidence",
            (),
            {
                "hit": True,
                "stable": True,
                "margin": 1.0,
            },
        )()
    return type(
        "ForcedEvidence",
        (),
        {
            "hit": True,
            "stable": bool(getattr(evidence, "stable", True)),
            "margin": float(getattr(evidence, "margin", 1.0)),
        },
    )()


def _candidate_audit(
    *,
    window_index: int,
    candidate_index: int,
    requested_unit_count: int,
    candidate: RewriteTokens,
    structure_ok: bool | None,
    preservation_cosine: float | None,
    preservation_passed: bool | None,
    semantic_validation_rule: str | None,
    evidence: Any | None,
    evaluation_status: str,
) -> GatedCandidateAudit:
    signature = getattr(evidence, "signature", None)
    if signature is not None:
        signature = tuple(signature)
    return GatedCandidateAudit(
        window_index=window_index,
        candidate_index=candidate_index,
        requested_unit_count=requested_unit_count,
        text=candidate.text,
        token_ids=candidate.token_ids,
        generation_seed_id=candidate.generation_seed_id,
        rewrite_config_id=candidate.rewrite_config_id,
        parse_status=candidate.parse_status,
        unit_count=candidate.unit_count,
        same_parent_scope=candidate.same_parent_scope,
        parser_spans=candidate.parser_spans,
        structure_ok=structure_ok,
        semantic_reference_cosine=preservation_cosine,
        semantic_preservation_passed=preservation_passed,
        semantic_validation_rule=semantic_validation_rule,
        keyed_lsh_scored=evidence is not None,
        lsh_signature=signature,
        semantic_hit=(None if evidence is None else bool(evidence.hit)),
        semantic_stable=(None if evidence is None else bool(evidence.stable)),
        semantic_margin=(None if evidence is None else float(evidence.margin)),
        selected=False,
        evaluation_status=evaluation_status,
    )


def _rewrite_request(
    *,
    prompt: str,
    completed_prefix: str,
    original_window: str,
    window: SemanticWindow,
    hard_boundary_after: bool,
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
            hard_boundary_after=hard_boundary_after,
        ),
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


def _consume_trailing_layout(source: bytes, end_byte: int) -> int:
    """Consume only the newline layout that a complete rewrite reproduces."""

    cursor = end_byte
    while cursor < len(source) and source[cursor : cursor + 1] in {b"\r", b"\n"}:
        cursor += 1
    return cursor


__all__ = [
    "ByteSpan",
    "GatedGenerationResult",
    "GatedCandidateAudit",
    "GatedGenerator",
    "GatedWindowAudit",
    "RewriteTokens",
]
