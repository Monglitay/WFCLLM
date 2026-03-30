"""Diagnostics primitives and helpers for route-one observability."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, TypedDict


class FailureReason(str, Enum):
    signature_miss = "signature_miss"
    margin_miss = "margin_miss"
    signature_and_margin_miss = "signature_and_margin_miss"
    no_block_generated = "no_block_generated"
    cascade_replaced = "cascade_replaced"
    unknown = "unknown"


_FAILURE_REASON_VALUES = {reason.value for reason in FailureReason}


def hash_block_text(text: str) -> str:
    """Return a SHA-256 text digest to keep block text storage compact."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_failure_reason(reason: Any) -> str:
    if isinstance(reason, FailureReason):
        return reason.value
    if isinstance(reason, str) and reason in _FAILURE_REASON_VALUES:
        return reason
    return FailureReason.unknown.value


def _normalize_value(value: Any) -> Any:
    if isinstance(value, FailureReason):
        return value.value
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _freeze_normalized_value(value: Any) -> Any:
    value = _normalize_value(value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_normalized_value(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_normalized_value(item) for item in value)
    return value


def _cascade_event_identity(event: Dict[str, Any]) -> Any:
    rollback_id = event.get("rollback_id")
    if rollback_id is not None:
        return ("rollback_id", _freeze_normalized_value(rollback_id))
    return ("event", _freeze_normalized_value(event))


class InitialVerifyRecord(TypedDict, total=False):
    passed: bool
    failure_reason: FailureReason


class RetryAttemptRecord(TypedDict, total=False):
    attempt_index: int
    produced_block: bool
    failure_reason: FailureReason


class CascadeEventRecord(TypedDict, total=False):
    triggered: bool


class FinalOutcomeRecord(TypedDict, total=False):
    embedded: bool
    rescued_by_retry: bool
    rescued_by_cascade: bool
    exhausted_retries: bool
    failure_reason: FailureReason


@dataclass
class BlockLifecycleRecord:
    sample_id: str
    block_ordinal: int
    initial_verify: InitialVerifyRecord = field(default_factory=dict)
    retry_attempts: List[RetryAttemptRecord] = field(default_factory=list)
    cascade_events: List[CascadeEventRecord] = field(default_factory=list)
    final_outcome: FinalOutcomeRecord = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "block_ordinal": self.block_ordinal,
            "initial_verify": _normalize_value(self.initial_verify),
            "retry_attempts": _normalize_value(self.retry_attempts),
            "cascade_events": _normalize_value(self.cascade_events),
            "final_outcome": _normalize_value(self.final_outcome),
        }


def summarize_sample_diagnostics(records: Iterable[BlockLifecycleRecord]) -> Dict[str, Any]:
    failure_reason_counts: Dict[str, int] = {
        reason.value: 0 for reason in FailureReason
    }
    retry_summary = {
        "blocks_with_retry": 0,
        "attempts_total": 0,
        "attempts_no_block": 0,
        "retry_rescued_blocks": 0,
        "retry_exhausted_blocks": 0,
    }
    cascade_summary = {
        "cascade_triggers": 0,
        "cascade_rollbacks": 0,
        "cascade_rescued_blocks": 0,
    }
    rescued_blocks = 0
    unrescued_blocks = 0
    seen_cascade_rollbacks = set()

    for record in records:
        initial_reason = record.initial_verify.get("failure_reason")
        if initial_reason is not None:
            failure_reason_counts[_normalize_failure_reason(initial_reason)] += 1

        attempts = record.retry_attempts or []
        if attempts:
            retry_summary["blocks_with_retry"] += 1
        retry_summary["attempts_total"] += len(attempts)
        for attempt in attempts:
            if not attempt.get("produced_block", False):
                retry_summary["attempts_no_block"] += 1
            reason = attempt.get("failure_reason")
            if reason is not None:
                failure_reason_counts[_normalize_failure_reason(reason)] += 1

        for event in record.cascade_events or []:
            event_identity = _cascade_event_identity(event)
            if event_identity in seen_cascade_rollbacks:
                continue
            seen_cascade_rollbacks.add(event_identity)
            triggered = event.get("triggered", False)
            if triggered:
                cascade_summary["cascade_triggers"] += 1
            cascade_summary["cascade_rollbacks"] += 1

        outcome = record.final_outcome or {}
        record_rescued = False
        if outcome.get("rescued_by_retry"):
            retry_summary["retry_rescued_blocks"] += 1
            record_rescued = True
        if outcome.get("rescued_by_cascade"):
            cascade_summary["cascade_rescued_blocks"] += 1
            record_rescued = True
        if record_rescued:
            rescued_blocks += 1
        if not outcome.get("embedded", False):
            unrescued_blocks += 1
        if outcome.get("exhausted_retries"):
            retry_summary["retry_exhausted_blocks"] += 1
        terminal_reason = outcome.get("failure_reason")
        if terminal_reason is not None:
            failure_reason_counts[_normalize_failure_reason(terminal_reason)] += 1

    return {
        "diagnostics_version": 1,
        "retry_summary": retry_summary,
        "cascade_summary": cascade_summary,
        "failure_reason_counts": failure_reason_counts,
        "rescued_blocks": rescued_blocks,
        "unrescued_blocks": unrescued_blocks,
    }
