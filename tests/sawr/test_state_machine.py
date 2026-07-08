from __future__ import annotations

import hashlib
from dataclasses import dataclass

from wfcllm.sawr import BoundaryEvent, Candidate, RuleDecision, RuleRequest
from wfcllm.sawr.state_machine import (
    AuditEvent,
    LayerFrame,
    SawrStateMachine,
    StateMachineSnapshot,
)


@dataclass(frozen=True)
class FakeCheckpoint:
    label: str


class SequenceRule:
    rule_name = "sequence"

    def __init__(self, hits: list[bool]) -> None:
        self._hits = hits
        self.requests: list[RuleRequest] = []

    def evaluate(self, request: RuleRequest) -> RuleDecision:
        self.requests.append(request)
        hit = self._hits.pop(0)
        return RuleDecision(hit=hit, reason=f"hit={hit}", rule_name=self.rule_name)


def _candidate(
    text: str,
    position_id: str = "module.foo.body",
    layer_path: tuple[str, ...] | None = None,
) -> Candidate:
    resolved_path = layer_path or (position_id,)
    return Candidate(
        text=text,
        candidate_type="simple_statement",
        node_type="return_statement",
        position_id=position_id,
        token_start_idx=0,
        token_count=1,
        parent_node_type="if_statement" if len(resolved_path) > 1 else "function_definition",
        layer_path=resolved_path,
    )


def _compound_event(
    path: tuple[str, ...],
    checkpoint_key: str,
    node_type: str = "if_statement",
) -> BoundaryEvent:
    return BoundaryEvent(
        kind="compound_started",
        node_type=node_type,
        parent_node_type="function_definition" if len(path) == 2 else "if_statement",
        position_id=path[0],
        layer_path=path,
        token_start_idx=0,
        token_count=0,
        text=f"{node_type}:",
        start_byte=0,
        end_byte=0,
        depth=max(0, len(path) - 1),
        checkpoint_key=checkpoint_key,
    )


def _close_event(
    path: tuple[str, ...],
    final_flush: bool = False,
    closed_layer_paths: tuple[tuple[str, ...], ...] | None = None,
    text: str = "",
    node_type: str | None = None,
    parent_node_type: str = "function_definition",
) -> BoundaryEvent:
    return BoundaryEvent(
        kind="layer_closed",
        node_type=node_type or ("if_statement" if len(path) > 1 else "function_body"),
        parent_node_type=parent_node_type,
        position_id=path[0],
        layer_path=path,
        token_start_idx=0,
        token_count=0,
        text=text,
        start_byte=0,
        end_byte=0,
        depth=max(0, len(path) - 1),
        checkpoint_key=path[-1],
        closed_layer_paths=closed_layer_paths or (path,),
        final_flush=final_flush,
    )


def _simple_event(text: str, path: tuple[str, ...]) -> BoundaryEvent:
    candidate = _candidate(text, position_id=path[0], layer_path=path)
    return BoundaryEvent(
        kind="simple_candidate",
        node_type=candidate.node_type,
        parent_node_type=candidate.parent_node_type,
        position_id=candidate.position_id,
        layer_path=path,
        token_start_idx=candidate.token_start_idx,
        token_count=candidate.token_count,
        text=text,
        start_byte=0,
        end_byte=0,
        depth=max(0, len(path) - 1),
        checkpoint_key=path[-1],
        candidate=candidate,
    )


def _state_machine(
    rule: SequenceRule,
    max_group_statements: int = 2,
    retry_budget: int = 1,
    statement_retry_budget: int | None = None,
    window_retry_budget: int | None = None,
    compound_retry_budget: int | None = None,
) -> SawrStateMachine[FakeCheckpoint]:
    return SawrStateMachine(
        sample_id="HumanEval/0",
        seed=17,
        max_group_statements=max_group_statements,
        retry_budget=retry_budget,
        statement_retry_budget=statement_retry_budget,
        window_retry_budget=window_retry_budget,
        compound_retry_budget=compound_retry_budget,
        rule=rule,
    )


def _event_names(state_machine: SawrStateMachine[FakeCheckpoint]) -> list[str]:
    return [event.event for event in state_machine.drain_audit_events()]


def _candidate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_state_machine_types_are_exported_from_package_root() -> None:
    from wfcllm.sawr import (
        BoundaryEventKind,
        CheckpointT,
        DecisionAction,
        LayerFrame,
        StateMachineSnapshot,
    )

    assert BoundaryEventKind is not None
    assert CheckpointT.__name__ == "CheckpointT"
    assert DecisionAction is not None
    assert LayerFrame is not None
    assert StateMachineSnapshot is not None


def test_state_machine_clears_group_after_hit() -> None:
    rule = SequenceRule([True])
    state_machine = _state_machine(rule)

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    assert decision.action == "accept"
    assert decision.rollback_checkpoint is None
    assert state_machine.current_group_size == 0
    assert state_machine.retry_count == 0
    assert state_machine.current_retry_count == 0
    assert state_machine.accepted_hit_count == 1
    events = state_machine.drain_audit_events()
    assert [event.event for event in events] == [
        "simple_candidate_observed",
        "accepted_generation_time_window",
    ]
    assert events[0].decision is None
    assert events[0].reason is None
    assert events[0].rule_name is None
    assert events[1].decision == "hit"
    assert events[1].rule_name == "sequence"
    assert events[1].reason == "hit=True"
    assert events[1].position_id == "module.foo.body"
    assert events[1].candidate_type == "simple_statement"
    assert events[1].candidate_hash == _candidate_hash("return x")
    assert events[1].group_statement_count == 1


def test_state_machine_does_not_reuse_candidates_from_previous_hit() -> None:
    rule = SequenceRule([True, True])
    state_machine = _state_machine(rule)

    first = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))
    second = state_machine.observe_candidate(_candidate("return y"), FakeCheckpoint("c1"))

    assert first.action == "accept"
    assert second.action == "accept"
    assert [request.candidates for request in rule.requests] == [
        (_candidate("return x"),),
        (_candidate("return y"),),
    ]
    assert state_machine.accepted_hit_count == 2
    assert _event_names(state_machine) == [
        "simple_candidate_observed",
        "accepted_generation_time_window",
        "simple_candidate_observed",
        "accepted_generation_time_window",
    ]


def test_state_machine_waits_when_group_misses_below_max_size() -> None:
    rule = SequenceRule([False])
    state_machine = _state_machine(rule, max_group_statements=2)

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    assert decision.action == "continue"
    assert state_machine.current_group_size == 1
    assert state_machine.retry_count == 0
    assert state_machine.current_retry_count == 0
    assert _event_names(state_machine) == [
        "simple_candidate_observed",
        "layer_window_rule_miss",
    ]


def test_state_machine_uses_sliding_suffix_when_full_window_misses() -> None:
    rule = SequenceRule([False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)

    first = state_machine.observe_candidate(_candidate("x = 1"), FakeCheckpoint("c0"))
    second = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c1"))

    assert first.action == "continue"
    assert second.action == "continue"
    assert state_machine.current_group_size == 1
    assert state_machine.retry_count == 0
    assert state_machine.current_retry_count == 0
    assert [request.candidates for request in rule.requests] == [
        (_candidate("x = 1"),),
        (_candidate("x = 1"), _candidate("return x")),
    ]
    assert _event_names(state_machine) == [
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "simple_candidate_observed",
        "layer_window_rule_miss",
    ]


def test_statement_retry_budget_requests_rollback_on_single_statement_miss() -> None:
    rule = SequenceRule([False])
    state_machine = _state_machine(
        rule,
        max_group_statements=1,
        retry_budget=0,
        statement_retry_budget=1,
        window_retry_budget=0,
        compound_retry_budget=0,
    )

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("s0"))

    assert decision.action == "rollback"
    assert decision.rollback_checkpoint == FakeCheckpoint("s0")
    assert state_machine.retry_count == 1
    assert _event_names(state_machine) == [
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "statement_retry_requested",
    ]


def test_window_retry_budget_requests_rollback_on_full_window_miss() -> None:
    rule = SequenceRule([False, False])
    state_machine = _state_machine(
        rule,
        max_group_statements=2,
        retry_budget=0,
        statement_retry_budget=0,
        window_retry_budget=1,
        compound_retry_budget=0,
    )

    first = state_machine.observe_candidate(_candidate("x = 1"), FakeCheckpoint("s0"))
    second = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("s1"))

    assert first.action == "continue"
    assert second.action == "rollback"
    assert second.rollback_checkpoint == FakeCheckpoint("s1")
    assert state_machine.retry_count == 1
    assert [request.candidates for request in rule.requests] == [
        (_candidate("x = 1"),),
        (_candidate("x = 1"), _candidate("return x")),
    ]
    assert _event_names(state_machine) == [
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "window_retry_requested",
    ]


def test_compound_retry_budget_overrides_legacy_retry_budget() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False])
    state_machine = _state_machine(
        rule,
        max_group_statements=2,
        retry_budget=0,
        statement_retry_budget=0,
        window_retry_budget=0,
        compound_retry_budget=1,
    )

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    rollback = state_machine.observe_event(_close_event(path), None)

    assert rollback.action == "rollback"
    assert rollback.rollback_checkpoint == FakeCheckpoint("if0")
    assert state_machine.retry_count == 1
    assert _event_names(state_machine) == [
        "compound_layer_started",
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "layer_retry_requested",
    ]


def test_compound_layer_close_scores_whole_layer_window_before_retry() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False, True])
    state_machine = _state_machine(
        rule,
        max_group_statements=2,
        retry_budget=1,
        statement_retry_budget=0,
        window_retry_budget=0,
        compound_retry_budget=1,
    )

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    close = state_machine.observe_event(
        _close_event(
            path,
            text="if x:\n    return x",
            node_type="if_statement",
            parent_node_type="function_definition",
        ),
        None,
    )

    assert close.action == "close"
    assert state_machine.accepted_hit_count == 1
    assert rule.requests[-1].candidates[0].text == "if x:\n    return x"
    assert rule.requests[-1].candidates[0].candidate_type == "compound_layer"
    assert rule.requests[-1].candidates[0].node_type == "if_statement"
    assert rule.requests[-1].candidates[0].parent_node_type == "function_definition"
    assert _event_names(state_machine) == [
        "compound_layer_started",
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "compound_layer_window_observed",
        "accepted_generation_time_window",
        "layer_closed_with_direct_hit",
    ]


def test_state_machine_empties_window_after_max_one_miss() -> None:
    rule = SequenceRule([False])
    state_machine = _state_machine(rule, max_group_statements=1, retry_budget=1)

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    assert decision.action == "continue"
    assert state_machine.current_group_size == 0
    assert state_machine.retry_count == 0
    assert state_machine.current_retry_count == 0
    assert _event_names(state_machine) == [
        "simple_candidate_observed",
        "layer_window_rule_miss",
    ]


def test_state_machine_flushes_partial_group_at_generation_end_hit() -> None:
    rule = SequenceRule([False, True])
    state_machine = _state_machine(rule, max_group_statements=2)
    observed = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    decision = state_machine.flush()

    assert observed.action == "continue"
    assert decision.action == "accept"
    assert state_machine.current_group_size == 0
    assert state_machine.accepted_hit_count == 1
    assert state_machine.current_retry_count == 0
    events = state_machine.drain_audit_events()
    assert [event.event for event in events] == [
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "accepted_generation_time_window",
        "layer_closed_with_direct_hit",
    ]
    assert events[-2].final_flush is True


def test_state_machine_flushes_partial_group_at_generation_end_miss() -> None:
    rule = SequenceRule([False, False])
    state_machine = _state_machine(rule, max_group_statements=2)
    observed = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    decision = state_machine.flush()

    assert observed.action == "continue"
    assert decision.action == "close"
    assert state_machine.current_group_size == 0
    assert state_machine.closed_without_hit_count == 1
    assert state_machine.current_retry_count == 0
    events = state_machine.drain_audit_events()
    assert [event.event for event in events] == [
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "closed_without_hit",
    ]
    assert events[-1].final_flush is True
    assert events[-1].decision == "miss"


def test_child_layer_hit_bubbles_to_parent_on_close() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([True])
    state_machine = _state_machine(rule)
    state_machine.open_root_layer(path[:1], "function_body", "function_definition")
    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))

    accept = state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    close = state_machine.observe_event(_close_event(path), None)

    assert accept.action == "accept"
    assert close.action == "close"
    assert state_machine.accepted_hit_count == 1
    assert state_machine.current_group_size == 0
    assert _event_names(state_machine) == [
        "compound_layer_started",
        "simple_candidate_observed",
        "accepted_generation_time_window",
        "layer_closed_with_direct_hit",
    ]


def test_if_statement_succeeds_when_any_branch_hits() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, True])
    state_machine = _state_machine(rule, max_group_statements=3)
    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))

    miss = state_machine.observe_event(_simple_event("x = 1", path), FakeCheckpoint("s0"))
    hit = state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s1"))
    close = state_machine.observe_event(_close_event(path), None)
    root_close = state_machine.flush()

    assert miss.action == "continue"
    assert hit.action == "accept"
    assert close.action == "close"
    assert root_close.action == "close"
    assert state_machine.accepted_hit_count == 1
    assert state_machine.closed_without_hit_count == 0
    assert _event_names(state_machine) == [
        "compound_layer_started",
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "simple_candidate_observed",
        "accepted_generation_time_window",
        "layer_closed_with_direct_hit",
        "layer_closed_with_child_hit",
    ]


def test_layer_close_without_hit_requests_rollback() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)

    opened = state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    observed = state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    rollback = state_machine.observe_event(_close_event(path), None)

    assert opened.action == "continue"
    assert observed.action == "continue"
    assert rollback.action == "rollback"
    assert rollback.rollback_checkpoint == FakeCheckpoint("if0")
    assert state_machine.retry_count == 1
    assert state_machine.current_retry_count == 0
    assert _event_names(state_machine) == [
        "compound_layer_started",
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "layer_retry_requested",
    ]


def test_layer_close_without_hit_falls_back_when_rollback_disallowed() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    fallback = state_machine.observe_event(
        _close_event(path),
        None,
        allow_rollback=False,
    )

    assert fallback.action == "fallback"
    assert fallback.rollback_checkpoint is None
    assert state_machine.retry_count == 0
    assert "layer_retry_requested" not in _event_names(state_machine)


def test_final_flush_layer_close_falls_back_without_retry_request() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    fallback = state_machine.observe_event(_close_event(path, final_flush=True), None)

    assert fallback.action == "fallback"
    assert fallback.rollback_checkpoint is None
    assert state_machine.retry_count == 0
    events = state_machine.drain_audit_events()
    event_names = [event.event for event in events]
    assert "layer_retry_requested" not in event_names
    assert event_names[-1] == "layer_fallback_committed_without_hit"
    assert events[-1].final_flush is True


def test_retry_count_persists_across_rollback_and_reopen_for_same_checkpoint_key() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False, False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    rollback = state_machine.observe_event(_close_event(path), None)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return y", path), FakeCheckpoint("s1"))
    fallback = state_machine.observe_event(_close_event(path), None)

    assert rollback.action == "rollback"
    assert rollback.rollback_checkpoint == FakeCheckpoint("if0")
    assert fallback.action == "fallback"
    assert fallback.rollback_checkpoint is None
    assert state_machine.retry_count == 1
    assert state_machine.fallback_count == 1
    assert state_machine.closed_without_hit_count == 1
    event_names = _event_names(state_machine)
    assert event_names.count("layer_retry_requested") == 1
    assert event_names.count("layer_fallback_committed_without_hit") == 1


def test_parent_rollback_clears_descendant_retry_and_retirement_state() -> None:
    parent_path = ("module.foo.body", "if:0")
    child_path = (*parent_path, "while:0")
    rule = SequenceRule([False, False, False, False, False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)

    state_machine.observe_event(_compound_event(parent_path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(
        _compound_event(child_path, "while:0", node_type="while_statement"),
        FakeCheckpoint("while0"),
    )
    state_machine.observe_event(_simple_event("x = 1", child_path), FakeCheckpoint("s0"))
    child_retry = state_machine.observe_event(_close_event(child_path), None)

    state_machine.observe_event(
        _compound_event(child_path, "while:0", node_type="while_statement"),
        FakeCheckpoint("while0"),
    )
    state_machine.observe_event(_simple_event("x = 2", child_path), FakeCheckpoint("s1"))
    child_fallback = state_machine.observe_event(_close_event(child_path), None)
    parent_retry = state_machine.observe_event(_close_event(parent_path), None)

    state_machine.observe_event(_compound_event(parent_path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(
        _compound_event(child_path, "while:0", node_type="while_statement"),
        FakeCheckpoint("while0"),
    )
    state_machine.observe_event(_simple_event("x = 3", child_path), FakeCheckpoint("s2"))
    child_retry_after_parent_rollback = state_machine.observe_event(
        _close_event(child_path),
        None,
    )

    assert child_retry.action == "rollback"
    assert child_fallback.action == "fallback"
    assert parent_retry.action == "rollback"
    assert child_retry_after_parent_rollback.action == "rollback"
    assert state_machine.retry_count == 3
    assert state_machine.fallback_count == 1
    assert state_machine.closed_without_hit_count == 1


def test_duplicate_attempt_falls_back_without_consuming_additional_retries() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False, False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=5)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    rollback = state_machine.observe_event(_close_event(path), None)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s1"))
    fallback = state_machine.observe_event(_close_event(path), None)

    assert rollback.action == "rollback"
    assert fallback.action == "fallback"
    assert state_machine.retry_count == 1
    assert state_machine.fallback_count == 1
    assert state_machine.closed_without_hit_count == 1
    events = state_machine.drain_audit_events()
    event_names = [event.event for event in events]
    assert event_names.count("layer_fallback_committed_without_hit") == 1
    assert events[-1].reason == "repeated_attempt_hash"


def test_snapshot_rollback_preserves_retry_and_attempt_safety_state() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False, False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    snapshot = state_machine.checkpoint()
    rollback = state_machine.observe_event(_close_event(path), None)

    state_machine.rollback(snapshot)
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s1"))
    fallback = state_machine.observe_event(_close_event(path), None)

    assert rollback.action == "rollback"
    assert fallback.action == "fallback"
    assert state_machine.retry_count == 1
    assert state_machine.fallback_count == 1
    assert state_machine.closed_without_hit_count == 1
    events = state_machine.drain_audit_events()
    event_names = [event.event for event in events]
    assert event_names.count("layer_fallback_committed_without_hit") == 1
    assert events[-1].reason == "repeated_attempt_hash"


def test_descendant_close_bubbles_hit_before_parent_child_hit_audit() -> None:
    parent_path = ("module.foo.body", "if:0")
    child_path = (*parent_path, "while:0")
    rule = SequenceRule([True])
    state_machine = _state_machine(rule)

    state_machine.observe_event(_compound_event(parent_path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(
        _compound_event(child_path, "while:0", node_type="while_statement"),
        FakeCheckpoint("while0"),
    )
    state_machine.observe_event(_simple_event("return x", child_path), FakeCheckpoint("s0"))
    close = state_machine.observe_event(
        _close_event(
            parent_path,
            closed_layer_paths=(parent_path, child_path),
        ),
        None,
    )

    assert close.action == "close"
    assert state_machine.accepted_hit_count == 1
    assert state_machine.closed_without_hit_count == 0
    events = _event_names(state_machine)
    assert events[-2:] == [
        "layer_closed_with_direct_hit",
        "layer_closed_with_child_hit",
    ]


def test_layer_falls_back_after_retry_exhaustion() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=0)

    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s0"))
    fallback = state_machine.observe_event(_close_event(path), None)

    assert fallback.action == "fallback"
    assert fallback.rollback_checkpoint is None
    assert state_machine.fallback_count == 1
    assert state_machine.closed_without_hit_count == 1
    assert _event_names(state_machine) == [
        "compound_layer_started",
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "layer_fallback_committed_without_hit",
    ]


def test_checkpoint_and_rollback_restore_layer_stack_and_counters() -> None:
    path = ("module.foo.body", "if:0")
    rule = SequenceRule([False, True])
    state_machine = _state_machine(rule)
    state_machine.observe_event(_compound_event(path, "if:0"), FakeCheckpoint("if0"))
    snapshot = state_machine.checkpoint()
    state_machine.observe_event(_simple_event("x = 1", path), FakeCheckpoint("s0"))

    state_machine.rollback(snapshot)
    decision = state_machine.observe_event(_simple_event("return x", path), FakeCheckpoint("s1"))

    assert decision.action == "accept"
    assert state_machine.candidate_count == 1
    assert state_machine.accepted_hit_count == 1
    assert state_machine.current_group_size == 0


def test_state_machine_can_record_no_controlled_body_close() -> None:
    rule = SequenceRule([])
    state_machine = _state_machine(rule)

    state_machine.record_no_controlled_body()

    assert state_machine.closed_without_hit_count == 1
    assert state_machine.drain_audit_events() == [
        AuditEvent(
            sample_id="HumanEval/0",
            position_id=None,
            event="closed_without_hit",
            candidate_type=None,
            group_statement_count=0,
            final_flush=True,
            rule_name=None,
            decision="miss",
            reason="no_controlled_function_body",
            candidate_hash=None,
        )
    ]


def test_layer_snapshot_types_are_importable_for_annotations() -> None:
    frame = LayerFrame[FakeCheckpoint](
        layer_id=("module.foo.body",),
        node_type="function_body",
        parent_node_type="function_definition",
        position_id="module.foo.body",
        checkpoint_before_layer=None,
        checkpoint_key="root",
        window=[],
    )
    snapshot = StateMachineSnapshot[FakeCheckpoint](
        layer_stack=(frame,),
        accepted_hit_count=0,
        closed_without_hit_count=0,
        fallback_count=0,
        candidate_count=0,
        retry_count=0,
        audit_events=(),
    )

    assert snapshot.layer_stack[0].checkpoint_key == "root"
