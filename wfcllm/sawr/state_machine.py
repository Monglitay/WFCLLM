from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from wfcllm.sawr.boundary import Candidate
from wfcllm.sawr.rules import EmbeddingRule, RuleRequest

CheckpointT = TypeVar("CheckpointT")

DecisionAction = Literal["accept", "continue", "rollback", "fallback", "close"]


@dataclass(frozen=True)
class AuditEvent:
    sample_id: str
    position_id: str | None
    event: str
    candidate_type: str | None
    group_statement_count: int
    final_flush: bool
    rule_name: str | None
    decision: str | None
    reason: str | None
    candidate_hash: str | None


@dataclass(frozen=True)
class StateMachineDecision(Generic[CheckpointT]):
    action: DecisionAction
    rollback_checkpoint: CheckpointT | None = None


class SawrStateMachine(Generic[CheckpointT]):
    def __init__(
        self,
        *,
        sample_id: str,
        seed: int,
        max_group_statements: int,
        retry_budget: int,
        rule: EmbeddingRule,
    ) -> None:
        if max_group_statements <= 0:
            raise ValueError("max_group_statements must be positive")
        if retry_budget < 0:
            raise ValueError("retry_budget must be non-negative")

        self._sample_id = sample_id
        self._seed = seed
        self._max_group_statements = max_group_statements
        self._retry_budget = retry_budget
        self._rule = rule
        self._current_group: list[Candidate] = []
        self._group_start_checkpoint: CheckpointT | None = None
        self._retry_count = 0
        self._audit_events: list[AuditEvent] = []

        self.accepted_hit_count = 0
        self.closed_without_hit_count = 0
        self.fallback_count = 0
        self.candidate_count = 0

    @property
    def current_group_size(self) -> int:
        return len(self._current_group)

    @property
    def retry_count(self) -> int:
        return self._retry_count

    def observe_candidate(
        self,
        candidate: Candidate,
        group_start_checkpoint: CheckpointT,
    ) -> StateMachineDecision[CheckpointT]:
        if not self._current_group:
            self._group_start_checkpoint = group_start_checkpoint

        self._current_group.append(candidate)
        self.candidate_count += 1
        self._audit_events.append(
            self._audit_event(
                event="candidate_observed",
                final_flush=False,
                decision=None,
                reason=None,
                rule_name=None,
                candidate=candidate,
            )
        )

        request = self._build_rule_request(final_flush=False)
        decision = self._rule.evaluate(request)
        if decision.hit:
            self._audit_events.append(
                self._audit_event(
                    event="accepted_generation_time_group",
                    final_flush=False,
                    decision="hit",
                    reason=decision.reason,
                    rule_name=decision.rule_name,
                    candidate=candidate,
                )
            )
            self.accepted_hit_count += 1
            self._clear_group(reset_retry=True)
            return StateMachineDecision(action="accept")

        self._audit_events.append(
            self._audit_event(
                event="group_rule_miss",
                final_flush=False,
                decision="miss",
                reason=decision.reason,
                rule_name=decision.rule_name,
                candidate=candidate,
            )
        )
        if len(self._current_group) < self._max_group_statements:
            return StateMachineDecision(action="continue")

        checkpoint = self._group_start_checkpoint
        if self._retry_count < self._retry_budget:
            self._retry_count += 1
            self._clear_group(reset_retry=False)
            self._audit_events.append(
                self._audit_event(
                    event="rollback_requested",
                    final_flush=False,
                    decision="miss",
                    reason=decision.reason,
                    rule_name=decision.rule_name,
                    candidate=candidate,
                    group_statement_count=len(request.candidates),
                )
            )
            return StateMachineDecision(
                action="rollback",
                rollback_checkpoint=checkpoint,
            )

        self.fallback_count += 1
        self._audit_events.append(
            self._audit_event(
                event="fallback_committed_without_hit",
                final_flush=False,
                decision="miss",
                reason=decision.reason,
                rule_name=decision.rule_name,
                candidate=candidate,
            )
        )
        self._clear_group(reset_retry=True)
        return StateMachineDecision(action="fallback")

    def flush(self) -> StateMachineDecision[CheckpointT]:
        if not self._current_group:
            return StateMachineDecision(action="close")

        candidate = self._current_group[-1]
        request = self._build_rule_request(final_flush=True)
        decision = self._rule.evaluate(request)
        if decision.hit:
            self._audit_events.append(
                self._audit_event(
                    event="accepted_generation_time_group",
                    final_flush=True,
                    decision="hit",
                    reason=decision.reason,
                    rule_name=decision.rule_name,
                    candidate=candidate,
                )
            )
            self.accepted_hit_count += 1
            self._clear_group(reset_retry=True)
            return StateMachineDecision(action="accept")

        self.closed_without_hit_count += 1
        self._audit_events.append(
            self._audit_event(
                event="closed_without_hit",
                final_flush=True,
                decision="miss",
                reason=decision.reason,
                rule_name=decision.rule_name,
                candidate=candidate,
            )
        )
        self._clear_group(reset_retry=True)
        return StateMachineDecision(action="close")

    def record_no_controlled_body(self) -> None:
        self.closed_without_hit_count += 1
        self._audit_events.append(
            AuditEvent(
                sample_id=self._sample_id,
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
        )

    def drain_audit_events(self) -> list[AuditEvent]:
        events = list(self._audit_events)
        self._audit_events.clear()
        return events

    def _build_rule_request(self, *, final_flush: bool) -> RuleRequest:
        if not self._current_group:
            raise ValueError("current group must not be empty")
        return RuleRequest(
            sample_id=self._sample_id,
            position_id=self._current_group[0].position_id,
            candidates=tuple(self._current_group),
            seed=self._seed,
            final_flush=final_flush,
        )

    def _audit_event(
        self,
        *,
        event: str,
        final_flush: bool,
        decision: str | None,
        reason: str | None,
        rule_name: str | None,
        candidate: Candidate,
        group_statement_count: int | None = None,
    ) -> AuditEvent:
        return AuditEvent(
            sample_id=self._sample_id,
            position_id=candidate.position_id,
            event=event,
            candidate_type=candidate.candidate_type,
            group_statement_count=(
                len(self._current_group)
                if group_statement_count is None
                else group_statement_count
            ),
            final_flush=final_flush,
            rule_name=rule_name,
            decision=decision,
            reason=reason,
            candidate_hash=self._candidate_hash(candidate),
        )

    def _clear_group(self, *, reset_retry: bool) -> None:
        self._current_group.clear()
        self._group_start_checkpoint = None
        if reset_retry:
            self._retry_count = 0

    def _candidate_hash(self, candidate: Candidate) -> str:
        return hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
