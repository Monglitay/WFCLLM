from __future__ import annotations

from dataclasses import replace

import pytest

from wfcllm.windowing.contracts import GateScores, StatementUnit
from wfcllm.windowing.partitioner import (
    CloseReason,
    GateDecision,
    GateThresholds,
    PartitionResult,
    SemanticWindow,
    SkipReason,
    SkippedContext,
    WindowPartitioner,
)


class FakeGatePredictor:
    def __init__(self, scores: list[GateScores]) -> None:
        self._scores = iter(scores)
        self.inputs: list[str] = []

    def predict(self, serialized_input: str) -> GateScores:
        self.inputs.append(serialized_input)
        return next(self._scores)


class StatelessGatePredictor:
    def __init__(self, score: GateScores | object) -> None:
        self.score = score

    def predict(self, _serialized_input: str) -> GateScores:
        return self.score  # type: ignore[return-value]


class RaisingGatePredictor:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def predict(self, _serialized_input: str) -> GateScores:
        raise self.error


def _score(
    close: float = 0.20,
    suitable: float = 0.90,
    *,
    stable: bool = True,
    decision_agreement: bool = True,
) -> GateScores:
    return GateScores(
        close_probability=close,
        suitable_probability=suitable,
        stable=stable,
        precision_delta=0.01,
        decision_agreement=decision_agreement,
    )


def _units(count: int, *, parent: str = "module") -> list[StatementUnit]:
    return [
        StatementUnit(
            unit_id=str(index),
            node_type="expression_statement",
            text=f"value_{index} = {index}",
            start_byte=index * 20,
            end_byte=index * 20 + 12,
            start_line=index + 1,
            end_line=index + 1,
            depth=0,
            parent_path=(parent,),
            direct_parent_type=parent,
            direct_child_ordinal=index,
            eligible=True,
            hard_boundary=False,
            compound_header=False,
        )
        for index in range(count)
    ]


def _partition(
    units: list[StatementUnit],
    scores: list[GateScores],
    *,
    tokenizer_counter=lambda _text: 32,
    thresholds: GateThresholds | None = None,
):
    predictor = FakeGatePredictor(scores)
    result = WindowPartitioner(
        predictor=predictor,
        thresholds=thresholds
        or GateThresholds(
            close_low=0.45,
            close_high=0.70,
            suitable_accept=0.80,
        ),
        tokenizer_counter=tokenizer_counter,
    ).partition(units)
    return result, predictor


def _valid_window() -> SemanticWindow:
    return _partition([_units(1)[0]], [_score()])[0].windows[0]


def _planned_generation_adapter(
    partitioner: WindowPartitioner, units: list[StatementUnit]
) -> PartitionResult:
    """Exercise the shared API shape planned for generation integration."""

    return partitioner.partition(units)


def _planned_detection_adapter(
    partitioner: WindowPartitioner, units: list[StatementUnit]
) -> PartitionResult:
    """Exercise the shared API shape planned for detection integration."""

    return partitioner.partition(units)


def test_non_overlapping_windows_close_at_gate_or_three_units() -> None:
    units = _units(5)
    result, predictor = _partition(
        units,
        [
            _score(0.20),
            _score(0.85, 0.88),
            _score(0.10, 0.99),
            _score(0.10, 0.99),
            _score(0.10, 0.99),
        ],
    )

    assert [window.unit_count for window in result.windows] == [2, 3]
    assert [window.close_reason for window in result.windows] == [
        CloseReason.GATE,
        CloseReason.MAX_UNITS,
    ]
    assert result.windows[0].suitable is True
    assert result.windows[0].units[-1].end_byte < result.windows[1].units[0].start_byte
    assert len(predictor.inputs) == 5
    assert "[CURRENT_UNIT_COUNT] 3" in predictor.inputs[-1]


def test_partition_is_deterministic_with_fresh_predictors() -> None:
    units = _units(3)
    scores = [_score(0.20), _score(0.20), _score(0.20)]

    first, first_predictor = _partition(units, list(scores))
    second, second_predictor = _partition(units, list(scores))

    assert first == second
    assert first_predictor.inputs == second_predictor.inputs


def test_parent_change_closes_before_predicting_new_parent() -> None:
    first = _units(1)[0]
    second = replace(
        _units(1, parent="block")[0],
        unit_id="1",
        start_byte=20,
        end_byte=32,
        start_line=2,
        end_line=2,
        depth=1,
    )

    result, predictor = _partition([first, second], [_score(), _score()])

    assert [window.unit_count for window in result.windows] == [1, 1]
    assert [window.close_reason for window in result.windows] == [
        CloseReason.PARENT_CHANGE,
        CloseReason.EOF,
    ]
    assert result.windows[0].parent_descriptor.direct_parent_type == "module"
    assert result.windows[1].parent_descriptor.direct_parent_type == "block"
    assert result.windows[1].previous_context == (first,)
    assert len(predictor.inputs) == 2


def test_first_compound_header_is_predicted_and_closed_as_singleton() -> None:
    header = replace(
        _units(1)[0],
        node_type="if_statement",
        text="if ready:",
        compound_header=True,
    )

    result, predictor = _partition([header], [_score(0.10, 0.95)])

    assert [window.units for window in result.windows] == [(header,)]
    assert result.windows[0].close_reason is CloseReason.COMPOUND_HEADER
    assert result.windows[0].parent_descriptor.compound_header_role == "header"
    assert result.windows[0].suitable is True
    assert len(predictor.inputs) == 1


def test_same_parent_compound_header_closes_body_before_singleton() -> None:
    body, header = _units(2)
    header = replace(
        header,
        node_type="if_statement",
        text="if value_0:",
        compound_header=True,
    )

    result, predictor = _partition(
        [body, header],
        [_score(0.10, 0.85), _score(0.10, 0.95)],
    )

    assert [window.units for window in result.windows] == [(body,), (header,)]
    assert [window.close_reason for window in result.windows] == [
        CloseReason.BEFORE_COMPOUND_HEADER,
        CloseReason.COMPOUND_HEADER,
    ]
    assert [window.suitable for window in result.windows] == [True, True]
    assert [
        window.parent_descriptor.compound_header_role
        for window in result.windows
    ] == ["body", "header"]
    assert len(predictor.inputs) == 2
    assert "[CURRENT_UNIT_COUNT] 1" in predictor.inputs[0]
    assert "[CURRENT_UNIT_COUNT] 1" in predictor.inputs[1]


def test_compound_header_cannot_accept_a_following_unit() -> None:
    header, following = _units(2)
    header = replace(
        header,
        node_type="if_statement",
        text="if ready:",
        compound_header=True,
    )

    result, predictor = _partition(
        [header, following],
        [_score(), _score()],
    )

    assert [window.units for window in result.windows] == [
        (header,),
        (following,),
    ]
    assert [window.close_reason for window in result.windows] == [
        CloseReason.COMPOUND_HEADER,
        CloseReason.EOF,
    ]
    assert len(predictor.inputs) == 2


def test_hard_boundary_closes_window_and_remains_in_later_context() -> None:
    first, boundary, third = _units(3)
    boundary = replace(
        boundary,
        node_type="import_statement",
        text="import os",
        eligible=False,
        hard_boundary=True,
    )

    result, predictor = _partition([first, boundary, third], [_score(), _score()])

    assert [window.close_reason for window in result.windows] == [
        CloseReason.HARD_BOUNDARY,
        CloseReason.EOF,
    ]
    assert result.windows[1].previous_context == (first, boundary)
    assert [item.unit for item in result.skipped_context] == [boundary]
    assert result.skipped_context[0].reason is SkipReason.HARD_BOUNDARY
    assert len(predictor.inputs) == 2
    assert "[S type=import_statement] import os" in predictor.inputs[-1]


def test_token_overflow_closes_and_skips_without_predicting() -> None:
    unit = _units(1)[0]

    result, predictor = _partition(
        [unit],
        [],
        tokenizer_counter=lambda _text: 513,
    )

    window = result.windows[0]
    assert window.close_reason is CloseReason.INPUT_OVERFLOW
    assert window.skip_reason is SkipReason.INPUT_OVERFLOW
    assert window.gate_scores is None
    assert window.suitable is False
    assert predictor.inputs == []


@pytest.mark.parametrize("overflow_at", [2, 3])
def test_later_token_overflow_discards_shorter_prefix_scores(
    overflow_at: int,
) -> None:
    calls = 0

    def tokenizer_counter(_text: str) -> int:
        nonlocal calls
        calls += 1
        return 513 if calls == overflow_at else 32

    units = _units(overflow_at)
    result, predictor = _partition(
        units,
        [_score() for _ in range(overflow_at - 1)],
        tokenizer_counter=tokenizer_counter,
    )

    window = result.windows[0]
    assert window.units == tuple(units)
    assert len(predictor.inputs) == overflow_at - 1
    predicted_counts = [
        f"[CURRENT_UNIT_COUNT] {index}" in serialized
        for index, serialized in enumerate(predictor.inputs, 1)
    ]
    assert predicted_counts == [True] * (overflow_at - 1)
    assert window.gate_scores is None
    assert window.gate_decision is GateDecision.SKIP
    assert window.close_reason is CloseReason.INPUT_OVERFLOW
    assert window.skip_reason is SkipReason.INPUT_OVERFLOW
    assert window.suitable is False


def test_uncertain_close_probability_closes_and_skips() -> None:
    result, _ = _partition([_units(1)[0]], [_score(0.55, 0.99)])

    window = result.windows[0]
    assert window.close_reason is CloseReason.UNCERTAIN
    assert window.skip_reason is SkipReason.UNCERTAIN_CLOSE
    assert window.suitable is False


@pytest.mark.parametrize(
    ("score", "reason"),
    [
        (_score(stable=False), SkipReason.UNSTABLE_SCORES),
        (
            _score(decision_agreement=False),
            SkipReason.DECISION_DISAGREEMENT,
        ),
    ],
)
def test_unreliable_gate_scores_close_and_skip(
    score: GateScores, reason: SkipReason
) -> None:
    result, _ = _partition([_units(1)[0]], [score])

    window = result.windows[0]
    assert window.close_reason is CloseReason.UNRELIABLE_GATE
    assert window.skip_reason is reason
    assert window.suitable is False


def test_third_unit_forces_close_even_when_gate_says_continue() -> None:
    units = _units(4)
    result, _ = _partition(
        units,
        [_score(), _score(), _score(), _score()],
    )

    assert [window.unit_count for window in result.windows] == [3, 1]
    assert [window.close_reason for window in result.windows] == [
        CloseReason.MAX_UNITS,
        CloseReason.EOF,
    ]


def test_gate_input_keeps_only_three_previous_parser_units() -> None:
    units = _units(6)
    boundaries = [
        replace(unit, eligible=False, hard_boundary=True) for unit in units[:5]
    ]

    result, predictor = _partition([*boundaries, units[5]], [_score()])

    window = result.windows[0]
    assert window.previous_context == tuple(boundaries[-3:])
    assert "value_0 = 0" not in predictor.inputs[0]
    assert "value_1 = 1" not in predictor.inputs[0]
    assert all(f"value_{index} = {index}" in predictor.inputs[0] for index in (2, 3, 4))


def test_eof_closes_open_window_using_last_valid_scores() -> None:
    result, _ = _partition(
        _units(2),
        [_score(0.20, 0.70), _score(0.20, 0.81)],
    )

    window = result.windows[0]
    assert window.close_reason is CloseReason.EOF
    assert window.gate_scores == _score(0.20, 0.81)
    assert window.suitable is True


def test_planned_thin_adapters_share_partition_api_and_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock the planned shared API shape, not future production integration."""

    units = _units(3)
    calls: list[WindowPartitioner] = []
    original_partition = WindowPartitioner.partition

    def partition_spy(
        partitioner: WindowPartitioner, input_units: list[StatementUnit]
    ) -> PartitionResult:
        calls.append(partitioner)
        return original_partition(partitioner, input_units)

    monkeypatch.setattr(WindowPartitioner, "partition", partition_spy)
    thresholds = GateThresholds(0.45, 0.70, 0.80)
    generation_partitioner = WindowPartitioner(
        StatelessGatePredictor(_score()), thresholds, lambda _text: 32
    )
    detection_partitioner = WindowPartitioner(
        StatelessGatePredictor(_score()), thresholds, lambda _text: 32
    )

    generated = _planned_generation_adapter(generation_partitioner, units)
    detected = _planned_detection_adapter(detection_partitioner, units)
    projection = lambda result: [
        (
            window.start_byte,
            window.end_byte,
            window.parent_descriptor,
            window.suitable,
        )
        for window in result.windows
    ]

    assert projection(generated) == projection(detected)
    assert calls == [generation_partitioner, detection_partitioner]


def test_same_partitioner_instance_is_repeatable_with_stateless_collaborators() -> None:
    partitioner = WindowPartitioner(
        StatelessGatePredictor(_score()),
        GateThresholds(0.45, 0.70, 0.80),
        lambda _text: 32,
    )
    units = _units(3)

    assert partitioner.partition(units) == partitioner.partition(units)


def test_empty_input_returns_empty_partition_result() -> None:
    result, predictor = _partition([], [])

    assert result == PartitionResult(windows=(), skipped_context=())
    assert predictor.inputs == []


@pytest.mark.parametrize(
    ("probability", "reason", "decision"),
    [
        (0.45, CloseReason.EOF, GateDecision.CONTINUE),
        (0.70, CloseReason.GATE, GateDecision.CLOSE),
    ],
)
def test_close_threshold_boundaries_are_exact(
    probability: float,
    reason: CloseReason,
    decision: GateDecision,
) -> None:
    result, _ = _partition([_units(1)[0]], [_score(probability)])

    assert result.windows[0].close_reason is reason
    assert result.windows[0].gate_decision is decision


@pytest.mark.parametrize("token_count", [True, 1.5, -1])
def test_tokenizer_counter_requires_non_negative_integer(token_count: object) -> None:
    with pytest.raises(
        ValueError,
        match="tokenizer_counter must return a non-negative integer",
    ):
        WindowPartitioner(
            StatelessGatePredictor(_score()),
            GateThresholds(0.45, 0.70, 0.80),
            lambda _text: token_count,  # type: ignore[return-value]
        ).partition(_units(1))


def test_predictor_must_return_gate_scores() -> None:
    with pytest.raises(
        TypeError,
        match="GatePredictor.predict must return GateScores",
    ):
        WindowPartitioner(
            StatelessGatePredictor("not scores"),
            GateThresholds(0.45, 0.70, 0.80),
            lambda _text: 32,
        ).partition(_units(1))


def test_predictor_exception_propagates_unchanged() -> None:
    error = RuntimeError("gate backend unavailable")

    with pytest.raises(RuntimeError) as raised:
        WindowPartitioner(
            RaisingGatePredictor(error),
            GateThresholds(0.45, 0.70, 0.80),
            lambda _text: 32,
        ).partition(_units(1))

    assert raised.value is error


@pytest.mark.parametrize("units", [[], ()])
def test_semantic_window_requires_non_empty_tuple(units: object) -> None:
    with pytest.raises(ValueError, match="units must be a non-empty tuple"):
        replace(_valid_window(), units=units)


def test_semantic_window_requires_statement_units() -> None:
    with pytest.raises(
        ValueError, match="units must contain only StatementUnit instances"
    ):
        replace(_valid_window(), units=("not a unit",))


def test_semantic_window_requires_ordered_non_overlapping_units() -> None:
    first, second = _units(2)
    second = replace(second, start_byte=5, end_byte=17)

    with pytest.raises(ValueError, match="units must be ordered and non-overlapping"):
        replace(_valid_window(), units=(first, second), end_byte=second.end_byte)


def test_semantic_window_contains_at_most_three_units() -> None:
    units = tuple(_units(4))

    with pytest.raises(ValueError, match="units must contain at most 3 units"):
        replace(_valid_window(), units=units, end_byte=units[-1].end_byte)


def test_semantic_window_rejects_hard_boundary_units() -> None:
    boundary = replace(
        _units(1)[0],
        eligible=False,
        hard_boundary=True,
    )

    with pytest.raises(ValueError, match="units must not contain hard boundaries"):
        replace(_valid_window(), units=(boundary,))


def test_semantic_window_rejects_compound_header_before_following_body() -> None:
    header, following = _units(2)
    header = replace(header, compound_header=True, node_type="if_statement")
    window = _valid_window()

    with pytest.raises(
        ValueError,
        match="compound header must be a singleton window",
    ):
        replace(
            window,
            units=(header, following),
            end_byte=following.end_byte,
            parent_descriptor=replace(
                window.parent_descriptor,
                compound_header_role="header",
            ),
        )


def test_semantic_window_rejects_body_before_compound_header() -> None:
    body, header = _units(2)
    header = replace(header, compound_header=True, node_type="if_statement")

    with pytest.raises(
        ValueError,
        match="compound header must be a singleton window",
    ):
        replace(
            _valid_window(),
            units=(body, header),
            end_byte=header.end_byte,
        )


def test_suitable_window_requires_gate_scores() -> None:
    with pytest.raises(ValueError, match="suitable window requires GateScores"):
        replace(_valid_window(), gate_scores=None)


@pytest.mark.parametrize(
    ("score_changes", "message"),
    [
        ({"stable": False}, "suitable window requires stable GateScores"),
        (
            {"decision_agreement": False},
            "suitable window requires decision-agreeing GateScores",
        ),
    ],
)
def test_suitable_window_requires_reliable_gate_scores(
    score_changes: dict[str, object], message: str
) -> None:
    window = _valid_window()

    with pytest.raises(ValueError, match=message):
        replace(
            window,
            gate_scores=replace(window.gate_scores, **score_changes),
        )


def test_suitable_window_requires_all_units_eligible() -> None:
    ineligible = replace(_units(1)[0], eligible=False)

    with pytest.raises(ValueError, match="suitable window requires eligible units"):
        replace(_valid_window(), units=(ineligible,))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"start_byte": 1}, "start_byte must match the first unit"),
        ({"end_byte": 11}, "end_byte must match the last unit"),
        ({"previous_context": []}, "previous_context must be a tuple"),
        (
            {"previous_context": tuple(_units(4))},
            "previous_context must contain at most 3 units",
        ),
        (
            {"previous_context": ("not a unit",)},
            "previous_context must contain only StatementUnit instances",
        ),
        ({"gate_scores": "scores"}, "gate_scores must be GateScores or None"),
        ({"gate_decision": "close"}, "gate_decision must be a GateDecision"),
        ({"close_reason": "eof"}, "close_reason must be a CloseReason"),
        ({"skip_reason": "none"}, "skip_reason must be a SkipReason or None"),
        ({"suitable": 1}, "suitable must be a bool"),
    ],
)
def test_semantic_window_rejects_invalid_public_fields(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_valid_window(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"gate_decision": GateDecision.SKIP},
        {
            "gate_decision": GateDecision.CONTINUE,
            "skip_reason": SkipReason.UNCERTAIN_CLOSE,
        },
    ],
)
def test_semantic_window_skip_decision_matches_skip_reason(
    changes: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match="gate_decision SKIP must match presence of skip_reason",
    ):
        replace(_valid_window(), **changes)


def test_semantic_window_skipped_result_cannot_be_suitable() -> None:
    with pytest.raises(ValueError, match="skipped window cannot be suitable"):
        replace(
            _valid_window(),
            gate_decision=GateDecision.SKIP,
            skip_reason=SkipReason.UNCERTAIN_CLOSE,
            suitable=True,
        )


@pytest.mark.parametrize(
    "descriptor_changes",
    [
        {"ancestor_node_types": ("function_definition",)},
        {"direct_parent_type": "block"},
        {"first_unit_ordinal": 1},
        {"compound_header_role": "header"},
    ],
)
def test_semantic_window_descriptor_matches_first_unit(
    descriptor_changes: dict[str, object],
) -> None:
    window = _valid_window()

    with pytest.raises(
        ValueError,
        match="parent_descriptor must match the first unit",
    ):
        replace(
            window,
            parent_descriptor=replace(
                window.parent_descriptor,
                **descriptor_changes,
            ),
        )


def test_semantic_window_requires_parent_descriptor() -> None:
    with pytest.raises(
        ValueError, match="parent_descriptor must be a ParentDescriptor"
    ):
        replace(_valid_window(), parent_descriptor="descriptor")


def test_semantic_window_units_share_parent_structure() -> None:
    first, second = _units(2)
    second = replace(
        second,
        depth=1,
        parent_path=("block",),
        direct_parent_type="block",
    )

    with pytest.raises(ValueError, match="units must share one parent structure"):
        replace(_valid_window(), units=(first, second), end_byte=second.end_byte)


def test_skipped_context_requires_statement_unit() -> None:
    with pytest.raises(ValueError, match="unit must be a StatementUnit"):
        SkippedContext(unit="not unit", reason=SkipReason.HARD_BOUNDARY)


def test_skipped_context_only_records_hard_boundaries() -> None:
    unit = _units(1)[0]

    with pytest.raises(ValueError, match="reason must be HARD_BOUNDARY"):
        SkippedContext(unit=unit, reason=SkipReason.INPUT_OVERFLOW)
    with pytest.raises(ValueError, match="unit must be a hard boundary"):
        SkippedContext(unit=unit, reason=SkipReason.HARD_BOUNDARY)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"windows": []}, "windows must be a tuple"),
        ({"windows": ("window",)}, "windows must contain only SemanticWindow"),
        ({"skipped_context": []}, "skipped_context must be a tuple"),
        (
            {"skipped_context": ("context",)},
            "skipped_context must contain only SkippedContext",
        ),
    ],
)
def test_partition_result_requires_typed_tuples(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(PartitionResult((), ()), **changes)


def test_partition_result_windows_are_ordered_and_non_overlapping() -> None:
    first = _valid_window()
    overlapping_unit = replace(
        _units(1)[0],
        unit_id="overlap",
        start_byte=5,
        end_byte=17,
        direct_child_ordinal=1,
    )
    second = _partition([overlapping_unit], [_score()])[0].windows[0]

    with pytest.raises(
        ValueError, match="windows must be ordered and non-overlapping"
    ):
        PartitionResult(windows=(first, second), skipped_context=())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"close_low": -0.1}, "close_low must be a probability"),
        ({"close_high": 1.1}, "close_high must be a probability"),
        ({"suitable_accept": 1.1}, "suitable_accept must be a probability"),
        (
            {"close_low": 0.7, "close_high": 0.7},
            "close_low must be less than close_high",
        ),
        ({"max_units": 0}, "max_units must be between 1 and 3"),
        ({"max_units": 1.5}, "max_units must be between 1 and 3"),
        ({"max_units": 4}, "max_units must be between 1 and 3"),
        ({"max_input_tokens": 0}, "max_input_tokens must be positive"),
    ],
)
def test_gate_thresholds_reject_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "close_low": 0.45,
        "close_high": 0.70,
        "suitable_accept": 0.80,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        GateThresholds(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"close_probability": -0.1}, "close_probability must be a probability"),
        (
            {"suitable_probability": 1.1},
            "suitable_probability must be a probability",
        ),
        ({"precision_delta": -0.1}, "precision_delta must be a probability"),
        ({"stable": 1}, "stable must be a bool"),
        ({"decision_agreement": 1}, "decision_agreement must be a bool"),
    ],
)
def test_gate_scores_reject_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "close_probability": 0.2,
        "suitable_probability": 0.9,
        "stable": True,
        "precision_delta": 0.01,
        "decision_agreement": True,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        GateScores(**values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field", ["close_low", "close_high", "suitable_accept"]
)
def test_gate_thresholds_reject_non_finite_probabilities(
    field: str, value: float
) -> None:
    values = {
        "close_low": 0.45,
        "close_high": 0.70,
        "suitable_accept": 0.80,
    }
    values[field] = value

    with pytest.raises(ValueError, match=rf"{field} must be a probability"):
        GateThresholds(**values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    ["close_probability", "suitable_probability", "precision_delta"],
)
def test_gate_scores_reject_non_finite_probabilities(
    field: str, value: float
) -> None:
    values = {
        "close_probability": 0.2,
        "suitable_probability": 0.9,
        "stable": True,
        "precision_delta": 0.01,
        "decision_agreement": True,
    }
    values[field] = value

    with pytest.raises(ValueError, match=rf"{field} must be a probability"):
        GateScores(**values)
