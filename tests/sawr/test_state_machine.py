from __future__ import annotations

from dataclasses import dataclass

from wfcllm.sawr import Candidate, RuleDecision, RuleRequest
from wfcllm.sawr.state_machine import AuditEvent, SawrStateMachine


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


def _candidate(text: str, position_id: str = "module.foo.body") -> Candidate:
    return Candidate(
        text=text,
        candidate_type="simple_statement",
        node_type="return_statement",
        position_id=position_id,
        token_start_idx=0,
        token_count=1,
    )


def _state_machine(
    rule: SequenceRule,
    max_group_statements: int = 2,
    retry_budget: int = 1,
) -> SawrStateMachine[FakeCheckpoint]:
    return SawrStateMachine(
        sample_id="HumanEval/0",
        seed=17,
        max_group_statements=max_group_statements,
        retry_budget=retry_budget,
        rule=rule,
    )


def _event_names(state_machine: SawrStateMachine[FakeCheckpoint]) -> list[str]:
    return [event.event for event in state_machine.drain_audit_events()]


def test_state_machine_clears_group_after_hit() -> None:
    rule = SequenceRule([True])
    state_machine = _state_machine(rule)

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    assert decision.action == "accept"
    assert decision.rollback_checkpoint is None
    assert state_machine.current_group_size == 0
    assert state_machine.retry_count == 0
    assert state_machine.accepted_hit_count == 1
    assert _event_names(state_machine) == [
        "candidate_observed",
        "accepted_generation_time_group",
    ]


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
        "candidate_observed",
        "accepted_generation_time_group",
        "candidate_observed",
        "accepted_generation_time_group",
    ]


def test_state_machine_waits_when_group_misses_below_max_size() -> None:
    rule = SequenceRule([False])
    state_machine = _state_machine(rule, max_group_statements=2)

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    assert decision.action == "continue"
    assert state_machine.current_group_size == 1
    assert state_machine.retry_count == 0
    assert _event_names(state_machine) == [
        "candidate_observed",
        "group_rule_miss",
    ]


def test_state_machine_requests_rollback_when_full_group_misses_with_retry_left() -> None:
    rule = SequenceRule([False])
    state_machine = _state_machine(rule, max_group_statements=1, retry_budget=1)

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    assert decision.action == "rollback"
    assert decision.rollback_checkpoint == FakeCheckpoint("c0")
    assert state_machine.current_group_size == 0
    assert state_machine.retry_count == 1
    assert _event_names(state_machine) == [
        "candidate_observed",
        "group_rule_miss",
        "rollback_requested",
    ]


def test_state_machine_falls_back_after_retry_exhaustion() -> None:
    rule = SequenceRule([False])
    state_machine = _state_machine(rule, max_group_statements=1, retry_budget=0)

    decision = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    assert decision.action == "fallback"
    assert decision.rollback_checkpoint is None
    assert state_machine.current_group_size == 0
    assert state_machine.retry_count == 0
    assert state_machine.fallback_count == 1
    assert _event_names(state_machine) == [
        "candidate_observed",
        "group_rule_miss",
        "fallback_committed_without_hit",
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
    events = state_machine.drain_audit_events()
    assert [event.event for event in events] == [
        "candidate_observed",
        "group_rule_miss",
        "accepted_generation_time_group",
    ]
    assert events[-1].final_flush is True


def test_state_machine_flushes_partial_group_at_generation_end_miss() -> None:
    rule = SequenceRule([False, False])
    state_machine = _state_machine(rule, max_group_statements=2)
    observed = state_machine.observe_candidate(_candidate("return x"), FakeCheckpoint("c0"))

    decision = state_machine.flush()

    assert observed.action == "continue"
    assert decision.action == "close"
    assert state_machine.current_group_size == 0
    assert state_machine.closed_without_hit_count == 1
    events = state_machine.drain_audit_events()
    assert [event.event for event in events] == [
        "candidate_observed",
        "group_rule_miss",
        "closed_without_hit",
    ]
    assert events[-1].final_flush is True
    assert events[-1].decision == "miss"


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
