from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from wfcllm.sawr.boundary import BoundaryEvent, Candidate
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


@dataclass
class LayerFrame(Generic[CheckpointT]):
    layer_id: tuple[str, ...]
    node_type: str
    parent_node_type: str
    position_id: str
    checkpoint_before_layer: CheckpointT | None
    checkpoint_key: str
    window: list[Candidate]
    has_direct_hit: bool = False
    has_child_hit: bool = False
    retry_count: int = 0
    attempt_hashes: set[str] = field(default_factory=set)
    closed_reason: str | None = None


@dataclass(frozen=True)
class StateMachineSnapshot(Generic[CheckpointT]):
    layer_stack: tuple[LayerFrame[CheckpointT], ...]
    accepted_hit_count: int
    closed_without_hit_count: int
    fallback_count: int
    candidate_count: int
    retry_count: int
    audit_events: tuple[AuditEvent, ...]


@dataclass(frozen=True)
class StateMachineDecision(Generic[CheckpointT]):
    action: DecisionAction
    rollback_checkpoint: CheckpointT | None = None
    rollback_blocked: bool = False


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
        self._layer_stack: list[LayerFrame[CheckpointT]] = []
        self._retired_checkpoint_keys: set[tuple[str, ...]] = set()
        self._checkpoint_retry_counts: dict[tuple[str, ...], int] = {}
        self._checkpoint_attempt_hashes: dict[tuple[str, ...], set[str]] = {}
        self._retry_count = 0
        self._audit_events: list[AuditEvent] = []

        self.accepted_hit_count = 0
        self.closed_without_hit_count = 0
        self.fallback_count = 0
        self.candidate_count = 0

    @property
    def current_group_size(self) -> int:
        if not self._layer_stack:
            return 0
        return len(self._layer_stack[-1].window)

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def current_retry_count(self) -> int:
        if not self._layer_stack:
            return 0
        return self._layer_stack[-1].retry_count

    def checkpoint(self) -> StateMachineSnapshot[CheckpointT]:
        return StateMachineSnapshot(
            layer_stack=tuple(deepcopy(self._layer_stack)),
            accepted_hit_count=self.accepted_hit_count,
            closed_without_hit_count=self.closed_without_hit_count,
            fallback_count=self.fallback_count,
            candidate_count=self.candidate_count,
            retry_count=self._retry_count,
            audit_events=tuple(self._audit_events),
        )

    def rollback(self, snapshot: StateMachineSnapshot[CheckpointT]) -> None:
        post_snapshot_events = self._audit_events[len(snapshot.audit_events) :]
        self._layer_stack = list(deepcopy(snapshot.layer_stack))
        self.accepted_hit_count = snapshot.accepted_hit_count
        self.closed_without_hit_count = snapshot.closed_without_hit_count
        self.fallback_count = snapshot.fallback_count
        self.candidate_count = snapshot.candidate_count
        # Loop-prevention safety maps intentionally survive generation rollback.
        # They prevent retrying the same failed layer attempt after local state rewinds.
        self._retry_count = max(
            snapshot.retry_count,
            sum(self._checkpoint_retry_counts.values()),
        )
        self._audit_events = list(snapshot.audit_events) + post_snapshot_events

    def open_root_layer(
        self,
        layer_id: tuple[str, ...],
        node_type: str,
        parent_node_type: str,
    ) -> None:
        if self._layer_stack:
            return
        if not layer_id:
            raise ValueError("root layer_id must not be empty")
        self._layer_stack.append(
            LayerFrame(
                layer_id=layer_id,
                node_type=node_type,
                parent_node_type=parent_node_type,
                position_id=layer_id[0],
                checkpoint_before_layer=None,
                checkpoint_key="root",
                window=[],
                retry_count=self._checkpoint_retry_counts.get(layer_id, 0),
            )
        )

    def observe_event(
        self,
        event: BoundaryEvent,
        checkpoint: CheckpointT | None,
        *,
        allow_rollback: bool = True,
    ) -> StateMachineDecision[CheckpointT]:
        if event.kind == "compound_started":
            return self._open_layer(event, checkpoint)
        if event.kind == "simple_candidate":
            if event.candidate is None:
                raise ValueError("simple_candidate event must include a candidate")
            return self._observe_direct_candidate(event.candidate, checkpoint)
        if event.kind == "layer_closed":
            return self._close_layer_paths(
                event.closed_layer_paths,
                final_flush=event.final_flush,
                allow_rollback=allow_rollback,
            )
        if event.kind == "final_flush":
            return self.flush()
        raise ValueError(f"unsupported boundary event kind: {event.kind}")

    def observe_candidate(
        self,
        candidate: Candidate,
        group_start_checkpoint: CheckpointT,
    ) -> StateMachineDecision[CheckpointT]:
        if isinstance(candidate, BoundaryEvent):
            return self.observe_event(candidate, group_start_checkpoint)
        layer_path = candidate.layer_path or (candidate.position_id,)
        event = BoundaryEvent(
            kind="simple_candidate",
            node_type=candidate.node_type,
            parent_node_type=candidate.parent_node_type,
            position_id=candidate.position_id,
            layer_path=layer_path,
            token_start_idx=candidate.token_start_idx,
            token_count=candidate.token_count,
            text=candidate.text,
            start_byte=candidate.start_byte,
            end_byte=candidate.end_byte,
            depth=candidate.depth,
            checkpoint_key=layer_path[-1],
            candidate=candidate,
        )
        return self.observe_event(event, group_start_checkpoint)

    def observe_final_candidate(
        self,
        candidate: Candidate,
    ) -> StateMachineDecision[CheckpointT]:
        if isinstance(candidate, BoundaryEvent):
            return self.observe_event(candidate, None)
        layer_path = candidate.layer_path or (candidate.position_id,)
        event = BoundaryEvent(
            kind="simple_candidate",
            node_type=candidate.node_type,
            parent_node_type=candidate.parent_node_type,
            position_id=candidate.position_id,
            layer_path=layer_path,
            token_start_idx=candidate.token_start_idx,
            token_count=candidate.token_count,
            text=candidate.text,
            start_byte=candidate.start_byte,
            end_byte=candidate.end_byte,
            depth=candidate.depth,
            checkpoint_key=layer_path[-1],
            candidate=candidate,
            final_flush=True,
        )
        return self.observe_event(event, None)

    def flush(self) -> StateMachineDecision[CheckpointT]:
        final_decision = StateMachineDecision[CheckpointT](action="close")
        while self._layer_stack:
            frame = self._layer_stack.pop()
            accepted_before = self.accepted_hit_count
            decision = self._close_one_layer(
                frame,
                final_flush=True,
                allow_rollback=False,
            )
            if decision.action == "rollback":
                return decision
            if decision.action == "fallback":
                final_decision = decision
            elif self.accepted_hit_count > accepted_before:
                final_decision = StateMachineDecision(action="accept")
        return final_decision

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

    def record_budget_exhausted(self, name: str) -> None:
        self._audit_events.append(
            AuditEvent(
                sample_id=self._sample_id,
                position_id=None,
                event=name,
                candidate_type=None,
                group_statement_count=0,
                final_flush=True,
                rule_name=None,
                decision="stop",
                reason=name,
                candidate_hash=None,
            )
        )

    def drain_audit_events(self) -> list[AuditEvent]:
        events = list(self._audit_events)
        self._audit_events.clear()
        return events

    def _open_layer(
        self,
        event: BoundaryEvent,
        checkpoint: CheckpointT | None,
    ) -> StateMachineDecision[CheckpointT]:
        if not event.layer_path:
            raise ValueError("compound_started event must include a layer_path")

        parent_path = event.layer_path[:-1]
        if parent_path and not self._layer_stack:
            self.open_root_layer(
                parent_path,
                node_type=event.parent_node_type,
                parent_node_type="module",
            )

        if self._layer_stack and self._layer_stack[-1].layer_id == event.layer_path:
            return StateMachineDecision(action="continue")

        checkpoint_key = event.checkpoint_key or event.layer_path[-1]
        frame = LayerFrame(
            layer_id=event.layer_path,
            node_type=event.node_type,
            parent_node_type=event.parent_node_type,
            position_id=event.position_id,
            checkpoint_before_layer=checkpoint,
            checkpoint_key=checkpoint_key,
            window=[],
            retry_count=self._checkpoint_retry_counts.get(event.layer_path, 0),
        )
        self._layer_stack.append(frame)
        self._audit_events.append(
            self._audit_event(
                event="compound_layer_started",
                frame=frame,
                final_flush=event.final_flush,
                decision=None,
                reason=None,
                rule_name=None,
                group_statement_count=0,
            )
        )
        return StateMachineDecision(action="continue")

    def _observe_direct_candidate(
        self,
        candidate: Candidate,
        checkpoint: CheckpointT | None,
    ) -> StateMachineDecision[CheckpointT]:
        frame = self._ensure_candidate_layer(candidate, checkpoint)
        frame.window.append(candidate)
        frame.attempt_hashes.add(self._candidate_hash(candidate))
        self.candidate_count += 1
        self._audit_events.append(
            self._audit_event(
                event="simple_candidate_observed",
                candidate=candidate,
                frame=frame,
                final_flush=False,
                decision=None,
                reason=None,
                rule_name=None,
            )
        )

        request = self._build_rule_request(frame, final_flush=False)
        decision = self._rule.evaluate(request)
        if decision.hit:
            frame.has_direct_hit = True
            self.accepted_hit_count += 1
            self._audit_events.append(
                self._audit_event(
                    event="accepted_generation_time_window",
                    candidate=candidate,
                    frame=frame,
                    final_flush=False,
                    decision="hit",
                    reason=decision.reason,
                    rule_name=decision.rule_name,
                    group_statement_count=len(request.candidates),
                )
            )
            frame.window.clear()
            return StateMachineDecision(action="accept")

        self._audit_events.append(
            self._audit_event(
                event="layer_window_rule_miss",
                candidate=candidate,
                frame=frame,
                final_flush=False,
                decision="miss",
                reason=decision.reason,
                rule_name=decision.rule_name,
                group_statement_count=len(request.candidates),
            )
        )
        if len(frame.window) >= self._max_group_statements:
            suffix_size = self._max_group_statements - 1
            frame.window = frame.window[-suffix_size:] if suffix_size > 0 else []
        return StateMachineDecision(action="continue")

    def _ensure_candidate_layer(
        self,
        candidate: Candidate,
        checkpoint: CheckpointT | None,
    ) -> LayerFrame[CheckpointT]:
        layer_path = candidate.layer_path or (candidate.position_id,)
        if not layer_path:
            raise ValueError("candidate layer_path must not be empty")
        if self._layer_stack and self._layer_stack[-1].layer_id == layer_path:
            return self._layer_stack[-1]

        checkpoint_before_layer = (
            None if not self._layer_stack and len(layer_path) == 1 else checkpoint
        )
        frame = LayerFrame(
            layer_id=layer_path,
            node_type=candidate.parent_node_type,
            parent_node_type="module" if len(layer_path) == 1 else candidate.parent_node_type,
            position_id=candidate.position_id,
            checkpoint_before_layer=checkpoint_before_layer,
            checkpoint_key=layer_path[-1],
            window=[],
            retry_count=self._checkpoint_retry_counts.get(layer_path, 0),
        )
        self._layer_stack.append(frame)
        return frame

    def _close_layer_paths(
        self,
        closed_layer_paths: tuple[tuple[str, ...], ...],
        *,
        final_flush: bool,
        allow_rollback: bool,
    ) -> StateMachineDecision[CheckpointT]:
        if not closed_layer_paths:
            return StateMachineDecision(action="close")

        final_decision = StateMachineDecision[CheckpointT](action="close")
        for path in sorted(closed_layer_paths, key=len, reverse=True):
            while self._layer_stack and (
                self._layer_stack[-1].layer_id == path
                or self._is_descendant_path(self._layer_stack[-1].layer_id, path)
            ):
                frame = self._layer_stack.pop()
                decision = self._close_one_layer(
                    frame,
                    final_flush=final_flush,
                    allow_rollback=allow_rollback,
                )
                if decision.action == "rollback":
                    return decision
                if decision.action == "fallback":
                    final_decision = decision
        return final_decision

    def _close_one_layer(
        self,
        frame: LayerFrame[CheckpointT],
        *,
        final_flush: bool,
        allow_rollback: bool,
    ) -> StateMachineDecision[CheckpointT]:
        flush_decision = self._flush_frame_window(frame, final_flush=True)
        if flush_decision.action == "accept":
            frame.has_direct_hit = True

        if frame.has_direct_hit or frame.has_child_hit:
            if self._layer_stack:
                self._layer_stack[-1].has_child_hit = True
            self._audit_events.append(
                self._audit_event(
                    event=(
                        "layer_closed_with_child_hit"
                        if frame.has_child_hit
                        else "layer_closed_with_direct_hit"
                    ),
                    frame=frame,
                    final_flush=final_flush,
                    decision="hit",
                    reason=None,
                    rule_name=None,
                    group_statement_count=0,
                )
            )
            frame.closed_reason = "hit"
            return StateMachineDecision(action="close")

        if frame.checkpoint_before_layer is None:
            self.closed_without_hit_count += 1
            frame.closed_reason = "closed_without_hit"
            self._audit_events.append(
                self._audit_event(
                    event="closed_without_hit",
                    frame=frame,
                    final_flush=final_flush,
                    decision="miss",
                    reason=None,
                    rule_name=None,
                    group_statement_count=0,
                )
            )
            return StateMachineDecision(action="close")

        safety_key = frame.layer_id
        attempt_hash = self._attempt_hash(frame)
        seen_attempt_hashes = self._checkpoint_attempt_hashes.setdefault(
            safety_key,
            set(),
        )
        if attempt_hash in seen_attempt_hashes:
            return self._commit_fallback(
                frame,
                final_flush=final_flush,
                reason="repeated_attempt_hash",
            )

        seen_attempt_hashes.add(attempt_hash)
        if (
            frame.retry_count < self._retry_budget
            and safety_key not in self._retired_checkpoint_keys
        ):
            if final_flush:
                return self._commit_fallback(
                    frame,
                    final_flush=final_flush,
                    reason="final_flush",
                )
            if not allow_rollback:
                return self._commit_fallback(
                    frame,
                    final_flush=final_flush,
                    reason="rollback_disallowed",
                    rollback_blocked=True,
                )
            frame.retry_count += 1
            self._checkpoint_retry_counts[safety_key] = frame.retry_count
            self._retry_count += 1
            frame.closed_reason = "retry"
            self._audit_events.append(
                self._audit_event(
                    event="layer_retry_requested",
                    frame=frame,
                    final_flush=final_flush,
                    decision="miss",
                    reason=None,
                    rule_name=None,
                    group_statement_count=0,
                )
            )
            self._clear_descendant_safety_state(safety_key)
            return StateMachineDecision(
                action="rollback",
                rollback_checkpoint=frame.checkpoint_before_layer,
            )

        return self._commit_fallback(frame, final_flush=final_flush, reason=None)

    def _commit_fallback(
        self,
        frame: LayerFrame[CheckpointT],
        *,
        final_flush: bool,
        reason: str | None,
        rollback_blocked: bool = False,
    ) -> StateMachineDecision[CheckpointT]:
        self.closed_without_hit_count += 1
        self.fallback_count += 1
        self._retired_checkpoint_keys.add(frame.layer_id)
        frame.closed_reason = "fallback"
        self._audit_events.append(
            self._audit_event(
                event="layer_fallback_committed_without_hit",
                frame=frame,
                final_flush=final_flush,
                decision="miss",
                reason=reason,
                rule_name=None,
                group_statement_count=0,
            )
        )
        return StateMachineDecision(
            action="fallback",
            rollback_blocked=rollback_blocked,
        )

    def _flush_frame_window(
        self,
        frame: LayerFrame[CheckpointT],
        *,
        final_flush: bool,
    ) -> StateMachineDecision[CheckpointT]:
        if not frame.window:
            return StateMachineDecision(action="close")

        candidate = frame.window[-1]
        request = self._build_rule_request(frame, final_flush=final_flush)
        decision = self._rule.evaluate(request)
        if decision.hit:
            self.accepted_hit_count += 1
            self._audit_events.append(
                self._audit_event(
                    event="accepted_generation_time_window",
                    candidate=candidate,
                    frame=frame,
                    final_flush=final_flush,
                    decision="hit",
                    reason=decision.reason,
                    rule_name=decision.rule_name,
                    group_statement_count=len(request.candidates),
                )
            )
            frame.window.clear()
            return StateMachineDecision(action="accept")

        frame.window.clear()
        return StateMachineDecision(action="close")

    def _build_rule_request(
        self,
        frame: LayerFrame[CheckpointT],
        *,
        final_flush: bool,
    ) -> RuleRequest:
        if not frame.window:
            raise ValueError("layer window must not be empty")
        return RuleRequest(
            sample_id=self._sample_id,
            position_id=frame.window[0].position_id,
            candidates=tuple(frame.window),
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
        candidate: Candidate | None = None,
        frame: LayerFrame[CheckpointT] | None = None,
        group_statement_count: int | None = None,
    ) -> AuditEvent:
        return AuditEvent(
            sample_id=self._sample_id,
            position_id=(
                candidate.position_id
                if candidate is not None
                else frame.position_id
                if frame is not None
                else None
            ),
            event=event,
            candidate_type=candidate.candidate_type if candidate is not None else None,
            group_statement_count=(
                group_statement_count
                if group_statement_count is not None
                else len(frame.window)
                if frame is not None
                else 0
            ),
            final_flush=final_flush,
            rule_name=rule_name,
            decision=decision,
            reason=reason,
            candidate_hash=(
                self._candidate_hash(candidate) if candidate is not None else None
            ),
        )

    def _candidate_hash(self, candidate: Candidate) -> str:
        return hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()

    def _attempt_hash(self, frame: LayerFrame[CheckpointT]) -> str:
        payload = "\n".join(sorted(frame.attempt_hashes))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _clear_descendant_safety_state(self, parent_path: tuple[str, ...]) -> None:
        self._retired_checkpoint_keys = {
            path
            for path in self._retired_checkpoint_keys
            if not self._is_descendant_path(path, parent_path)
        }
        self._checkpoint_retry_counts = {
            path: count
            for path, count in self._checkpoint_retry_counts.items()
            if not self._is_descendant_path(path, parent_path)
        }
        self._checkpoint_attempt_hashes = {
            path: hashes
            for path, hashes in self._checkpoint_attempt_hashes.items()
            if not self._is_descendant_path(path, parent_path)
        }

    def _is_descendant_path(
        self,
        candidate_path: tuple[str, ...],
        parent_path: tuple[str, ...],
    ) -> bool:
        return (
            len(candidate_path) > len(parent_path)
            and candidate_path[: len(parent_path)] == parent_path
        )
