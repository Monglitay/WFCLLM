"""Deterministic semantic-window partitioning shared by both pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Callable, Protocol

from wfcllm.windowing.contracts import (
    WINDOW_CONTRACT_VERSION,
    GateScores,
    ParentDescriptor,
    StatementUnit,
)
from wfcllm.windowing.normalization import (
    WINDOW_NORMALIZATION_VERSION,
    normalize_unit_text,
)

if TYPE_CHECKING:
    from wfcllm.gate.input import GateInput


_SCORE_NOT_PROVIDED = object()


class GatePredictor(Protocol):
    """The inference boundary used by the pure partitioning state machine."""

    def predict(self, serialized_input: str) -> GateScores: ...


@dataclass(frozen=True)
class GateThresholds:
    close_low: float
    close_high: float
    suitable_accept: float
    max_units: int = 3
    max_input_tokens: int = 512

    def __post_init__(self) -> None:
        for name in ("close_low", "close_high", "suitable_accept"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if self.close_low >= self.close_high:
            raise ValueError("close_low must be less than close_high")
        if (
            isinstance(self.max_units, bool)
            or not isinstance(self.max_units, int)
            or not 1 <= self.max_units <= 3
        ):
            raise ValueError("max_units must be between 1 and 3")
        if (
            isinstance(self.max_input_tokens, bool)
            or not isinstance(self.max_input_tokens, int)
            or self.max_input_tokens <= 0
        ):
            raise ValueError("max_input_tokens must be positive")


class GateDecision(str, Enum):
    CONTINUE = "continue"
    CLOSE = "close"
    SKIP = "skip"


class CloseReason(str, Enum):
    GATE = "gate"
    UNCERTAIN = "uncertain"
    UNRELIABLE_GATE = "unreliable_gate"
    INPUT_OVERFLOW = "input_overflow"
    MAX_UNITS = "max_units"
    BEFORE_COMPOUND_HEADER = "before_compound_header"
    COMPOUND_HEADER = "compound_header"
    PARENT_CHANGE = "parent_change"
    HARD_BOUNDARY = "hard_boundary"
    EOF = "eof"


class SkipReason(str, Enum):
    HARD_BOUNDARY = "hard_boundary"
    INPUT_OVERFLOW = "input_overflow"
    UNCERTAIN_CLOSE = "uncertain_close"
    UNSTABLE_SCORES = "unstable_scores"
    DECISION_DISAGREEMENT = "decision_disagreement"


@dataclass(frozen=True)
class SkippedContext:
    """A hard boundary completely excluded from semantic windows.

    It remains available as previous parser context for later gate inputs.
    """

    unit: StatementUnit
    reason: SkipReason

    def __post_init__(self) -> None:
        if not isinstance(self.unit, StatementUnit):
            raise ValueError("unit must be a StatementUnit")
        if self.reason is not SkipReason.HARD_BOUNDARY:
            raise ValueError("reason must be HARD_BOUNDARY")
        if not self.unit.hard_boundary:
            raise ValueError("unit must be a hard boundary")


@dataclass(frozen=True)
class SemanticWindow:
    units: tuple[StatementUnit, ...]
    parent_descriptor: ParentDescriptor
    start_byte: int
    end_byte: int
    gate_scores: GateScores | None
    gate_decision: GateDecision
    close_reason: CloseReason
    suitable: bool
    skip_reason: SkipReason | None
    previous_context: tuple[StatementUnit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.units, tuple) or not self.units:
            raise ValueError("units must be a non-empty tuple")
        if any(not isinstance(unit, StatementUnit) for unit in self.units):
            raise ValueError("units must contain only StatementUnit instances")
        if len(self.units) > 3:
            raise ValueError("units must contain at most 3 units")
        if any(unit.hard_boundary for unit in self.units):
            raise ValueError("units must not contain hard boundaries")
        if len(self.units) > 1 and any(
            unit.compound_header for unit in self.units
        ):
            raise ValueError("compound header must be a singleton window")
        if any(
            left.end_byte > right.start_byte
            for left, right in zip(self.units, self.units[1:])
        ):
            raise ValueError("units must be ordered and non-overlapping")
        if self.start_byte != self.units[0].start_byte:
            raise ValueError("start_byte must match the first unit")
        if self.end_byte != self.units[-1].end_byte:
            raise ValueError("end_byte must match the last unit")
        if not isinstance(self.previous_context, tuple):
            raise ValueError("previous_context must be a tuple")
        if len(self.previous_context) > 3:
            raise ValueError("previous_context must contain at most 3 units")
        if any(
            not isinstance(unit, StatementUnit) for unit in self.previous_context
        ):
            raise ValueError(
                "previous_context must contain only StatementUnit instances"
            )
        if self.gate_scores is not None and not isinstance(
            self.gate_scores, GateScores
        ):
            raise ValueError("gate_scores must be GateScores or None")
        if not isinstance(self.gate_decision, GateDecision):
            raise ValueError("gate_decision must be a GateDecision")
        if not isinstance(self.close_reason, CloseReason):
            raise ValueError("close_reason must be a CloseReason")
        if self.skip_reason is not None and not isinstance(
            self.skip_reason, SkipReason
        ):
            raise ValueError("skip_reason must be a SkipReason or None")
        if not isinstance(self.suitable, bool):
            raise ValueError("suitable must be a bool")
        if (self.gate_decision is GateDecision.SKIP) != (
            self.skip_reason is not None
        ):
            raise ValueError(
                "gate_decision SKIP must match presence of skip_reason"
            )
        if self.skip_reason is not None and self.suitable:
            raise ValueError("skipped window cannot be suitable")
        if self.suitable:
            if self.gate_scores is None:
                raise ValueError("suitable window requires GateScores")
            if not self.gate_scores.stable:
                raise ValueError("suitable window requires stable GateScores")
            if not self.gate_scores.decision_agreement:
                raise ValueError(
                    "suitable window requires decision-agreeing GateScores"
                )
            if any(not unit.eligible for unit in self.units):
                raise ValueError("suitable window requires eligible units")
        if not isinstance(self.parent_descriptor, ParentDescriptor):
            raise ValueError("parent_descriptor must be a ParentDescriptor")

        first = self.units[0]
        expected_role = "header" if first.compound_header else "body"
        expected_ancestors = _descriptor_ancestor_node_types(first)
        descriptor_matches = (
            self.parent_descriptor.ancestor_node_types == expected_ancestors
            and self.parent_descriptor.direct_parent_type
            == first.direct_parent_type
            and self.parent_descriptor.first_unit_ordinal
            == first.direct_child_ordinal
            and self.parent_descriptor.compound_header_role == expected_role
        )
        if not descriptor_matches:
            raise ValueError("parent_descriptor must match the first unit")
        if any(
            unit.depth != first.depth
            or unit.parent_path != first.parent_path
            or unit.direct_parent_type != first.direct_parent_type
            for unit in self.units[1:]
        ) or any(
            left.direct_child_ordinal >= right.direct_child_ordinal
            for left, right in zip(self.units, self.units[1:])
        ):
            raise ValueError("units must share one parent structure")

    @property
    def unit_count(self) -> int:
        return len(self.units)


@dataclass(frozen=True)
class PartitionResult:
    windows: tuple[SemanticWindow, ...]
    skipped_context: tuple[SkippedContext, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.windows, tuple):
            raise ValueError("windows must be a tuple")
        if any(not isinstance(window, SemanticWindow) for window in self.windows):
            raise ValueError("windows must contain only SemanticWindow instances")
        if not isinstance(self.skipped_context, tuple):
            raise ValueError("skipped_context must be a tuple")
        if any(
            not isinstance(item, SkippedContext) for item in self.skipped_context
        ):
            raise ValueError(
                "skipped_context must contain only SkippedContext instances"
            )
        if any(
            left.end_byte > right.start_byte
            for left, right in zip(self.windows, self.windows[1:])
        ):
            raise ValueError("windows must be ordered and non-overlapping")


class WindowPartitioner:
    """Partition parser units without overlap or pipeline-specific behavior.

    The partitioner retains no per-call state. Repeatable results require
    deterministic collaborators: both the predictor and tokenizer counter.
    """

    def __init__(
        self,
        predictor: GatePredictor,
        thresholds: GateThresholds,
        tokenizer_counter: Callable[[str], int],
        window_contract_version: str = WINDOW_CONTRACT_VERSION,
        defer_unreliable_until_max_units: bool = False,
    ) -> None:
        from wfcllm.windowing.contracts import is_supported_window_contract

        if not is_supported_window_contract(window_contract_version):
            raise ValueError("unsupported window contract version")
        if type(defer_unreliable_until_max_units) is not bool:
            raise ValueError("defer_unreliable_until_max_units must be a bool")
        self._predictor = predictor
        self._thresholds = thresholds
        self._tokenizer_counter = tokenizer_counter
        self._window_contract_version = window_contract_version
        self._defer_unreliable_until_max_units = (
            defer_unreliable_until_max_units
        )

    @property
    def window_contract_version(self) -> str:
        return self._window_contract_version

    def partition(self, units: list[StatementUnit]) -> PartitionResult:
        from wfcllm.gate.input import serialize_gate_input

        windows: list[SemanticWindow] = []
        skipped_context: list[SkippedContext] = []
        consumed: list[StatementUnit] = []
        open_units: list[StatementUnit] = []
        previous_context: tuple[StatementUnit, ...] = ()
        parent_descriptor: ParentDescriptor | None = None
        last_scores: GateScores | None = None

        def close_window(
            reason: CloseReason,
            *,
            skip_reason: SkipReason | None = None,
            scores_override: GateScores | None | object = _SCORE_NOT_PROVIDED,
        ) -> None:
            nonlocal open_units, previous_context, parent_descriptor, last_scores
            if not open_units or parent_descriptor is None:
                return
            window_scores: GateScores | None
            if scores_override is _SCORE_NOT_PROVIDED:
                window_scores = last_scores
            elif scores_override is None:
                window_scores = None
            else:
                if not isinstance(scores_override, GateScores):
                    raise TypeError("scores_override must be GateScores or None")
                window_scores = scores_override
            if skip_reason is not None:
                gate_decision = GateDecision.SKIP
            elif (
                window_scores is not None
                and window_scores.close_probability
                >= self._thresholds.close_high
            ):
                gate_decision = GateDecision.CLOSE
            else:
                gate_decision = GateDecision.CONTINUE
            suitable = (
                skip_reason is None
                and window_scores is not None
                and window_scores.stable
                and window_scores.decision_agreement
                and all(unit.eligible for unit in open_units)
                and window_scores.suitable_probability
                >= self._thresholds.suitable_accept
            )
            windows.append(
                SemanticWindow(
                    units=tuple(open_units),
                    parent_descriptor=parent_descriptor,
                    start_byte=open_units[0].start_byte,
                    end_byte=open_units[-1].end_byte,
                    gate_scores=window_scores,
                    gate_decision=gate_decision,
                    close_reason=reason,
                    suitable=suitable,
                    skip_reason=skip_reason,
                    previous_context=previous_context,
                )
            )
            open_units = []
            previous_context = ()
            parent_descriptor = None
            last_scores = None

        for unit in units:
            if unit.hard_boundary:
                close_window(CloseReason.HARD_BOUNDARY)
                consumed.append(unit)
                skipped_context.append(
                    SkippedContext(unit=unit, reason=SkipReason.HARD_BOUNDARY)
                )
                continue

            if unit.compound_header and open_units:
                close_window(CloseReason.BEFORE_COMPOUND_HEADER)

            if open_units and not _has_same_parent(open_units, unit):
                close_window(CloseReason.PARENT_CHANGE)

            if not open_units:
                previous_context = tuple(consumed[-3:])
                parent_descriptor = _parent_descriptor(
                    unit, self._window_contract_version
                )

            open_units.append(unit)
            consumed.append(unit)

            gate_input, token_count = self._gate_input(
                open_units,
                previous_context,
                parent_descriptor,
            )
            if token_count > self._thresholds.max_input_tokens:
                close_window(
                    CloseReason.INPUT_OVERFLOW,
                    skip_reason=SkipReason.INPUT_OVERFLOW,
                    scores_override=None,
                )
                continue

            predicted_scores = self._predictor.predict(
                serialize_gate_input(gate_input)
            )
            if not isinstance(predicted_scores, GateScores):
                raise TypeError(
                    "GatePredictor.predict must return GateScores; "
                    f"got {type(predicted_scores).__name__} while partitioning "
                    f"{len(open_units)} current unit(s)"
                )
            last_scores = predicted_scores
            if not last_scores.stable:
                if (
                    self._defer_unreliable_until_max_units
                    and len(open_units) < self._thresholds.max_units
                    and not unit.compound_header
                ):
                    continue
                close_window(
                    CloseReason.UNRELIABLE_GATE,
                    skip_reason=SkipReason.UNSTABLE_SCORES,
                )
                continue
            if not last_scores.decision_agreement:
                if (
                    self._defer_unreliable_until_max_units
                    and len(open_units) < self._thresholds.max_units
                    and not unit.compound_header
                ):
                    continue
                close_window(
                    CloseReason.UNRELIABLE_GATE,
                    skip_reason=SkipReason.DECISION_DISAGREEMENT,
                )
                continue
            if (
                self._thresholds.close_low
                < last_scores.close_probability
                < self._thresholds.close_high
            ):
                close_window(
                    CloseReason.UNCERTAIN,
                    skip_reason=SkipReason.UNCERTAIN_CLOSE,
                )
                continue
            if unit.compound_header:
                close_window(CloseReason.COMPOUND_HEADER)
                continue
            if len(open_units) >= self._thresholds.max_units:
                close_window(CloseReason.MAX_UNITS)
                continue
            if last_scores.close_probability >= self._thresholds.close_high:
                close_window(CloseReason.GATE)

        close_window(CloseReason.EOF)
        return PartitionResult(
            windows=tuple(windows),
            skipped_context=tuple(skipped_context),
        )

    def _gate_input(
        self,
        current_units: list[StatementUnit],
        previous_context: tuple[StatementUnit, ...],
        parent_descriptor: ParentDescriptor,
    ) -> tuple[GateInput, int]:
        from wfcllm.gate.input import GateInput

        normalized_current = "\n".join(
            normalize_unit_text(unit.text) for unit in current_units
        )
        token_count = self._tokenizer_counter(normalized_current)
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("tokenizer_counter must return a non-negative integer")
        return (
            GateInput(
                normalization_version=WINDOW_NORMALIZATION_VERSION,
                parent_descriptor=parent_descriptor.canonical,
                depth=current_units[0].depth,
                previous_units=tuple(unit.text for unit in previous_context),
                previous_unit_types=tuple(
                    unit.node_type for unit in previous_context
                ),
                current_units=tuple(unit.text for unit in current_units),
                current_unit_types=tuple(unit.node_type for unit in current_units),
                current_unit_count=len(current_units),
                current_token_count=token_count,
            ),
            token_count,
        )


def _parent_descriptor(
    unit: StatementUnit,
    window_contract_version: str = WINDOW_CONTRACT_VERSION,
) -> ParentDescriptor:
    return ParentDescriptor(
        contract_version=window_contract_version,
        ancestor_node_types=_descriptor_ancestor_node_types(unit),
        direct_parent_type=unit.direct_parent_type,
        first_unit_ordinal=unit.direct_child_ordinal,
        compound_header_role="header" if unit.compound_header else "body",
    )


def _descriptor_ancestor_node_types(unit: StatementUnit) -> tuple[str, ...]:
    ancestor_node_types = unit.parent_path[:-1]
    while (
        ancestor_node_types
        and ancestor_node_types[-1] == unit.direct_parent_type
    ):
        ancestor_node_types = ancestor_node_types[:-1]
    return ancestor_node_types


def _has_same_parent(
    current_units: list[StatementUnit], candidate: StatementUnit
) -> bool:
    first = current_units[0]
    last = current_units[-1]
    return (
        candidate.depth == first.depth
        and candidate.parent_path == first.parent_path
        and candidate.direct_parent_type == first.direct_parent_type
        and candidate.direct_child_ordinal > last.direct_child_ordinal
    )
