from __future__ import annotations

import math
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from typing import Any

import torch
import torch.nn.functional as F

from wfcllm.generation.boundary import (
    BoundaryDetectorState,
    BoundaryEvent,
    Candidate,
    PromptAwareBoundaryDetector,
)
from wfcllm.generation.kv_cache import CacheSnapshot
from wfcllm.method.config import SawrGenerationConfig
from wfcllm.semantic.rules import EmbeddingRule
from wfcllm.generation.state_machine import (
    AuditEvent,
    LayerFrame,
    SawrStateMachine,
    StateMachineSnapshot,
)


class _LazyAutoTokenizer:
    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        from transformers import AutoTokenizer as _AutoTokenizer

        return _AutoTokenizer.from_pretrained(*args, **kwargs)


class _LazyAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

        return _AutoModelForCausalLM.from_pretrained(*args, **kwargs)


def BitsAndBytesConfig(*args: Any, **kwargs: Any) -> Any:
    from transformers import BitsAndBytesConfig as _BitsAndBytesConfig

    return _BitsAndBytesConfig(*args, **kwargs)


AutoTokenizer = _LazyAutoTokenizer
AutoModelForCausalLM = _LazyAutoModelForCausalLM


@dataclass(frozen=True)
class SawrGenerateResult:
    final_code: str
    accepted_hit_count: int
    closed_without_hit_count: int
    fallback_count: int
    candidate_count: int
    audit_events: list[AuditEvent]


@dataclass(frozen=True)
class SawrCheckpoint:
    generated_ids: list[int]
    generated_text: str
    kv_snapshot: Any
    boundary_state: BoundaryDetectorState | None
    next_logits: torch.Tensor | None
    state_machine_state: StateMachineSnapshot[SawrCheckpoint] | None
    _context_id: str = field(repr=False, compare=False)
    _generation_id: int = field(repr=False, compare=False)
    _checkpoint_id: int = field(repr=False, compare=False)


@dataclass(frozen=True)
class GeneratedToken:
    token_id: int
    token_text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_id, bool)
            or not isinstance(self.token_id, int)
            or self.token_id < 0
        ):
            raise ValueError("token_id must be a non-negative integer")
        if not isinstance(self.token_text, str):
            raise ValueError("token_text must be a string")


@dataclass(frozen=True)
class GeneratedByteCheckpoint:
    checkpoint: SawrCheckpoint
    protected_prefix: str
    exact_start_byte: int


@dataclass(frozen=True)
class _CheckpointRecord:
    generated_ids: tuple[int, ...]
    generated_text: str
    kv_seq_len: int
    boundary_state: BoundaryDetectorState | None
    boundary_id: object | None
    next_logits: torch.Tensor | None
    state_machine_state: StateMachineSnapshot[SawrCheckpoint] | None
    state_machine_fingerprint: str | None
    public_state_object: StateMachineSnapshot[SawrCheckpoint] | None
    intern_key: tuple[Any, ...]


@dataclass(frozen=True)
class GeneratedToken:
    token_id: int
    token_text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_id, bool)
            or not isinstance(self.token_id, int)
            or self.token_id < 0
        ):
            raise ValueError("token_id must be a non-negative integer")
        if not isinstance(self.token_text, str):
            raise ValueError("token_text must be a string")


@dataclass(frozen=True)
class _RetryPenaltySequence:
    base_ids: tuple[int, ...]
    failed_ids: tuple[int, ...]


class SawrModelContext:
    """Small causal-LM generation context with rollback support."""

    _MAX_RETRY_PENALTY_SEQUENCES = 128
    _CHECKPOINT_REGISTRY_HEADROOM = 512
    _SUPPORTED_STATE_DATACLASSES = frozenset(
        {StateMachineSnapshot, LayerFrame, AuditEvent, Candidate}
    )

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: SawrGenerationConfig,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        from wfcllm.generation.kv_cache import KVCacheManager

        self._cache_mgr = KVCacheManager()
        self._device = _model_device(model, config.device)
        self.generated_ids: list[int] = []
        self.generated_text = ""
        self.past_kv: tuple | None = None
        self.boundary: PromptAwareBoundaryDetector | None = None
        self._next_logits: torch.Tensor | None = None
        self._step_history: list[SawrCheckpoint] = []
        self._boundary_checkpoints: list[SawrCheckpoint] = []
        self._retry_penalty_sequences: list[_RetryPenaltySequence] = []
        self._context_id = uuid.uuid4().hex
        self._generation_id = 0
        self._next_checkpoint_id = 0
        self._checkpoint_records: dict[int, _CheckpointRecord] = {}
        self._interned_checkpoints: dict[tuple[Any, ...], SawrCheckpoint] = {}
        self._max_checkpoint_records = (
            2 * self._config.max_new_tokens + self._CHECKPOINT_REGISTRY_HEADROOM
        )

    @property
    def eos_id(self) -> int | None:
        if self._config.eos_token_id is not None:
            return self._config.eos_token_id
        return getattr(self._tokenizer, "eos_token_id", None)

    def prefill(self, prompt: str) -> None:
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_ids = input_ids.to(self._device)
        with torch.no_grad():
            output = self._model(input_ids=input_ids, use_cache=True)
        self.past_kv = output.past_key_values
        self._next_logits = output.logits[:, -1, :].detach().clone()
        self.generated_ids = []
        self.generated_text = ""
        self._step_history = []
        self._boundary_checkpoints = []
        self._retry_penalty_sequences = []
        self._generation_id += 1
        self._next_checkpoint_id = 0
        self._checkpoint_records = {}
        self._interned_checkpoints = {}

    def checkpoint(
        self,
        state_machine_state: StateMachineSnapshot[SawrCheckpoint] | None = None,
    ) -> SawrCheckpoint:
        if self.past_kv is None:
            raise ValueError("cannot checkpoint before prefill")
        boundary_state = (
            self.boundary.checkpoint() if self.boundary is not None else None
        )
        kv_seq_len = self._cache_mgr.snapshot(self.past_kv).seq_len
        intern_key = self._checkpoint_intern_key(
            generated_ids=tuple(self.generated_ids),
            generated_text=self.generated_text,
            kv_seq_len=kv_seq_len,
            boundary_state=boundary_state,
            boundary_id=self.boundary,
            next_logits=self._next_logits,
            state_machine_fingerprint=None,
        )
        existing = self._interned_checkpoints.get(intern_key)
        if existing is not None:
            self._validate_checkpoint(existing)
            base = existing
        else:
            base = self._register_checkpoint(
                generated_ids=tuple(self.generated_ids),
                generated_text=self.generated_text,
                kv_seq_len=kv_seq_len,
                boundary_state=boundary_state,
                boundary_id=self.boundary,
                next_logits=self._next_logits,
                state_machine_state=None,
                state_machine_fingerprint=None,
                public_state_object=None,
                intern_key=intern_key,
            )
        if state_machine_state is not None:
            return self.attach_state_machine_state(base, state_machine_state)
        return base

    def attach_state_machine_state(
        self,
        checkpoint: SawrCheckpoint,
        state_machine_state: StateMachineSnapshot[SawrCheckpoint],
    ) -> SawrCheckpoint:
        self._validate_checkpoint(checkpoint)
        if not isinstance(state_machine_state, StateMachineSnapshot):
            raise ValueError("state_machine_state must be a StateMachineSnapshot")
        fingerprint = repr(self._canonical_state_value(state_machine_state, set()))
        try:
            stored_state = deepcopy(state_machine_state)
            public_state = deepcopy(stored_state)
        except Exception as exc:
            raise ValueError("state_machine_state must support safe deepcopy") from exc
        base_record = self._checkpoint_records[checkpoint._checkpoint_id]
        intern_key = self._checkpoint_intern_key(
            generated_ids=base_record.generated_ids,
            generated_text=base_record.generated_text,
            kv_seq_len=base_record.kv_seq_len,
            boundary_state=base_record.boundary_state,
            boundary_id=base_record.boundary_id,
            next_logits=base_record.next_logits,
            state_machine_fingerprint=fingerprint,
        )
        existing = self._interned_checkpoints.get(intern_key)
        if existing is not None:
            self._validate_checkpoint(existing)
            return existing
        return self._register_checkpoint(
            generated_ids=base_record.generated_ids,
            generated_text=base_record.generated_text,
            kv_seq_len=base_record.kv_seq_len,
            boundary_state=base_record.boundary_state,
            boundary_id=base_record.boundary_id,
            next_logits=base_record.next_logits,
            state_machine_state=stored_state,
            state_machine_fingerprint=fingerprint,
            public_state_object=public_state,
            intern_key=intern_key,
        )

    def _canonical_state_value(
        self,
        value: Any,
        active_ids: set[int],
    ) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("state_machine_state contains a non-finite float")
            return value
        if isinstance(value, SawrCheckpoint):
            return (
                "SawrCheckpoint",
                value._context_id,
                value._generation_id,
                value._checkpoint_id,
            )
        value_id = id(value)
        if value_id in active_ids:
            raise ValueError("state_machine_state must not contain cycles")
        if isinstance(value, (tuple, list)):
            active_ids.add(value_id)
            result = tuple(
                self._canonical_state_value(item, active_ids) for item in value
            )
            active_ids.remove(value_id)
            return (type(value).__name__, result)
        if isinstance(value, (set, frozenset)):
            active_ids.add(value_id)
            items = [self._canonical_state_value(item, active_ids) for item in value]
            active_ids.remove(value_id)
            return (type(value).__name__, tuple(sorted(items, key=repr)))
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("state_machine_state dict keys must be strings")
            active_ids.add(value_id)
            items = tuple(
                (key, self._canonical_state_value(value[key], active_ids))
                for key in sorted(value)
            )
            active_ids.remove(value_id)
            return ("dict", items)
        if is_dataclass(value) and type(value) in self._SUPPORTED_STATE_DATACLASSES:
            active_ids.add(value_id)
            result = (
                type(value).__name__,
                tuple(
                    (
                        item.name,
                        self._canonical_state_value(getattr(value, item.name), active_ids),
                    )
                    for item in dataclass_fields(value)
                ),
            )
            active_ids.remove(value_id)
            return result
        raise ValueError(
            "state_machine_state contains an unsupported mutable or runtime value"
        )

    def _checkpoint_intern_key(
        self,
        *,
        generated_ids: tuple[int, ...],
        generated_text: str,
        kv_seq_len: int,
        boundary_state: BoundaryDetectorState | None,
        boundary_id: object | None,
        next_logits: torch.Tensor | None,
        state_machine_fingerprint: str | None,
    ) -> tuple[Any, ...]:
        return (
            self._generation_id,
            generated_ids,
            generated_text,
            kv_seq_len,
            boundary_state,
            id(boundary_id),
            id(next_logits),
            state_machine_fingerprint,
        )

    def _register_checkpoint(
        self,
        *,
        generated_ids: tuple[int, ...],
        generated_text: str,
        kv_seq_len: int,
        boundary_state: BoundaryDetectorState | None,
        boundary_id: object | None,
        next_logits: torch.Tensor | None,
        state_machine_state: StateMachineSnapshot[SawrCheckpoint] | None,
        state_machine_fingerprint: str | None,
        public_state_object: StateMachineSnapshot[SawrCheckpoint] | None,
        intern_key: tuple[Any, ...],
    ) -> SawrCheckpoint:
        if len(self._checkpoint_records) >= self._max_checkpoint_records:
            raise ValueError("checkpoint registry capacity exceeded")
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        checkpoint = SawrCheckpoint(
            generated_ids=list(generated_ids),
            generated_text=generated_text,
            kv_snapshot=CacheSnapshot(seq_len=kv_seq_len),
            boundary_state=boundary_state,
            next_logits=(
                next_logits.detach().clone()
                if next_logits is not None
                else None
            ),
            state_machine_state=public_state_object,
            _context_id=self._context_id,
            _generation_id=self._generation_id,
            _checkpoint_id=checkpoint_id,
        )
        self._checkpoint_records[checkpoint_id] = _CheckpointRecord(
            generated_ids=generated_ids,
            generated_text=generated_text,
            kv_seq_len=kv_seq_len,
            boundary_state=boundary_state,
            boundary_id=boundary_id,
            next_logits=next_logits,
            state_machine_state=state_machine_state,
            state_machine_fingerprint=state_machine_fingerprint,
            public_state_object=public_state_object,
            intern_key=intern_key,
        )
        self._interned_checkpoints[intern_key] = checkpoint
        return checkpoint

    def rollback(
        self,
        checkpoint: SawrCheckpoint,
        *,
        remember_failed_sequence: bool = False,
    ) -> StateMachineSnapshot[SawrCheckpoint] | None:
        if self.past_kv is None:
            raise ValueError("cannot rollback before prefill")
        self._validate_checkpoint(checkpoint)
        record = self._checkpoint_records[checkpoint._checkpoint_id]
        if remember_failed_sequence:
            self._remember_failed_sequence(checkpoint)
        self.past_kv = self._cache_mgr.rollback(self.past_kv, checkpoint.kv_snapshot)
        self.generated_ids = list(record.generated_ids)
        self.generated_text = record.generated_text
        if self.boundary is not None and record.boundary_state is not None:
            self.boundary.rollback(record.boundary_state)
        self._next_logits = record.next_logits
        self._step_history = self._step_history[: len(self.generated_ids)]
        boundary_count = (
            len(record.boundary_state.token_boundaries)
            if record.boundary_state is not None
            else 0
        )
        self._boundary_checkpoints = self._boundary_checkpoints[:boundary_count]
        self._prune_checkpoint_registry(record, checkpoint._checkpoint_id)
        return (
            deepcopy(record.state_machine_state)
            if record.state_machine_state is not None
            else None
        )

    def _prune_checkpoint_registry(
        self,
        target: _CheckpointRecord,
        target_checkpoint_id: int,
    ) -> None:
        retained_ids = {
            checkpoint_id
            for checkpoint_id, record in self._checkpoint_records.items()
            if (
                checkpoint_id == target_checkpoint_id
                or (
                    len(record.generated_ids) <= len(target.generated_ids)
                    and target.generated_ids[: len(record.generated_ids)]
                    == record.generated_ids
                    and target.generated_text.startswith(record.generated_text)
                    and record.kv_seq_len <= target.kv_seq_len
                )
            )
        }
        self._checkpoint_records = {
            checkpoint_id: record
            for checkpoint_id, record in self._checkpoint_records.items()
            if checkpoint_id in retained_ids
        }
        self._interned_checkpoints = {
            key: value
            for key, value in self._interned_checkpoints.items()
            if value._checkpoint_id in retained_ids
        }

    def _validate_checkpoint(self, checkpoint: SawrCheckpoint) -> None:
        if not isinstance(checkpoint, SawrCheckpoint):
            raise ValueError("checkpoint must be a SawrCheckpoint")
        if checkpoint._context_id != self._context_id:
            raise ValueError("checkpoint does not belong to the current context")
        if checkpoint._generation_id != self._generation_id:
            raise ValueError("checkpoint is stale after a newer prefill")
        record = self._checkpoint_records.get(checkpoint._checkpoint_id)
        if record is None:
            raise ValueError("checkpoint is stale or unknown")
        next_logits_match = (
            record.next_logits is None
            and checkpoint.next_logits is None
        ) or (
            record.next_logits is not None
            and checkpoint.next_logits is not None
            and record.next_logits.shape == checkpoint.next_logits.shape
            and record.next_logits.dtype == checkpoint.next_logits.dtype
            and torch.equal(record.next_logits, checkpoint.next_logits)
        )
        if (
            tuple(checkpoint.generated_ids) != record.generated_ids
            or checkpoint.generated_text != record.generated_text
            or checkpoint.kv_snapshot.seq_len != record.kv_seq_len
            or checkpoint.boundary_state != record.boundary_state
            or not next_logits_match
        ):
            raise ValueError("checkpoint was modified after creation")
        try:
            observed_state_fingerprint = (
                repr(self._canonical_state_value(checkpoint.state_machine_state, set()))
                if checkpoint.state_machine_state is not None
                else None
            )
        except ValueError as exc:
            raise ValueError(
                "checkpoint state_machine_state was modified after creation"
            ) from exc
        if (
            record.state_machine_fingerprint is None
            and checkpoint.state_machine_state is not None
        ) or (
            record.state_machine_fingerprint is not None
            and observed_state_fingerprint != record.state_machine_fingerprint
        ):
            raise ValueError("checkpoint state_machine_state was modified after creation")
        if self.boundary is not record.boundary_id:
            raise ValueError("checkpoint boundary detector does not match current context")
        if len(self.generated_ids) < len(record.generated_ids):
            raise ValueError("checkpoint snapshot is stale for the current branch")
        if tuple(self.generated_ids[: len(record.generated_ids)]) != record.generated_ids:
            raise ValueError("checkpoint snapshot is stale for the current branch")
        if not self.generated_text.startswith(record.generated_text):
            raise ValueError("checkpoint snapshot is stale for the current branch")
        if self.past_kv is None:
            raise ValueError("cannot validate checkpoint before prefill")
        current_seq_len = self._cache_mgr.snapshot(self.past_kv).seq_len
        if record.kv_seq_len > current_seq_len:
            raise ValueError("checkpoint snapshot is stale for the current KV cache")

    def _remember_failed_sequence(self, checkpoint: SawrCheckpoint) -> None:
        if self._config.retry_repetition_penalty <= 1.0:
            return
        base_ids = tuple(checkpoint.generated_ids)
        if len(base_ids) > len(self.generated_ids):
            return
        if tuple(self.generated_ids[: len(base_ids)]) != base_ids:
            return
        failed_ids = tuple(self.generated_ids[len(base_ids):])
        if not failed_ids:
            return
        sequence = _RetryPenaltySequence(base_ids=base_ids, failed_ids=failed_ids)
        if sequence in self._retry_penalty_sequences:
            return
        self._retry_penalty_sequences.append(sequence)
        excess = len(self._retry_penalty_sequences) - self._MAX_RETRY_PENALTY_SEQUENCES
        if excess > 0:
            del self._retry_penalty_sequences[:excess]

    def record_boundary_checkpoint(
        self,
        state_machine: SawrStateMachine[SawrCheckpoint],
    ) -> None:
        if not self._boundary_checkpoints:
            raise ValueError("cannot attach state before a generated token checkpoint")
        self._boundary_checkpoints[-1] = self.attach_state_machine_state(
            self._boundary_checkpoints[-1],
            state_machine.checkpoint(),
        )

    def forward_and_sample(self) -> GeneratedToken:
        step_checkpoint = self.checkpoint()
        logits = self._next_token_logits()
        next_id = self._sample(logits)
        token_text = self._decode_generated_token(next_id)
        self._forward_forced_token(next_id)

        self.generated_ids.append(next_id)
        self.generated_text += token_text
        self._step_history.append(step_checkpoint)
        self._boundary_checkpoints.append(step_checkpoint)
        return GeneratedToken(token_id=next_id, token_text=token_text)

    def replay_from_checkpoint(
        self,
        checkpoint: SawrCheckpoint,
        *,
        replacement_steps: list[GeneratedToken] | tuple[GeneratedToken, ...],
    ) -> SawrCheckpoint:
        if not isinstance(replacement_steps, (list, tuple)) or not replacement_steps:
            raise ValueError("replacement_steps must be a non-empty sequence")
        if any(not isinstance(step, GeneratedToken) for step in replacement_steps):
            raise ValueError("replacement_steps must contain only GeneratedToken values")
        if not "".join(step.token_text for step in replacement_steps):
            raise ValueError("replacement token text must be non-empty in aggregate")
        self._validate_checkpoint(checkpoint)
        for index, step in enumerate(replacement_steps):
            if step.token_text != self._decode_generated_token(step.token_id):
                raise ValueError(
                    "replacement token text does not match tokenizer decode "
                    f"at token index {index}"
                )
        penalties_before = list(self._retry_penalty_sequences)
        self.rollback(checkpoint)
        start_length = len(checkpoint.generated_ids)
        try:
            for step in replacement_steps:
                step_checkpoint = self.checkpoint()
                self._forward_forced_token(step.token_id)
                self.generated_ids.append(step.token_id)
                self.generated_text += step.token_text
                self._step_history.append(step_checkpoint)
                self._boundary_checkpoints.append(step_checkpoint)
                if self.boundary is not None:
                    self.boundary.feed_text(step.token_text)
        except Exception:
            self.rollback(checkpoint)
            self._retry_penalty_sequences = penalties_before
            raise
        self._retry_penalty_sequences = [
            sequence
            for sequence in penalties_before
            if len(sequence.base_ids) < start_length
        ]
        return self.checkpoint()

    def _decode_generated_token(self, token_id: int) -> str:
        token_text = self._tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
        )
        if not isinstance(token_text, str):
            raise ValueError("tokenizer.decode must return a string")
        return token_text

    def _forward_forced_token(self, token_id: int) -> None:
        if self.past_kv is None:
            raise ValueError("cannot replay before prefill")
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self._device)
        with torch.no_grad():
            output = self._model(
                input_ids=input_ids,
                past_key_values=self.past_kv,
                use_cache=True,
            )
        self.past_kv = output.past_key_values
        self._next_logits = output.logits[:, -1, :].detach().clone()

    def checkpoint_for_generated_byte(self, start_byte: int) -> GeneratedByteCheckpoint:
        if isinstance(start_byte, bool) or not isinstance(start_byte, int):
            raise ValueError("start_byte must be an integer")
        generated_bytes = self.generated_text.encode("utf-8")
        if start_byte < 0 or start_byte > len(generated_bytes):
            raise ValueError("start_byte is outside generated text")
        try:
            generated_bytes[:start_byte].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("start_byte must be a UTF-8 codepoint boundary") from exc

        candidates = [*self._step_history, self.checkpoint()]
        eligible = [
            candidate
            for candidate in candidates
            if len(candidate.generated_text.encode("utf-8")) <= start_byte
        ]
        if not eligible:
            raise ValueError("no valid token checkpoint exists for start_byte")
        _, selected = max(
            enumerate(eligible),
            key=lambda item: (
                len(item[1].generated_text.encode("utf-8")),
                item[0],
            ),
        )
        selected_end = len(selected.generated_text.encode("utf-8"))
        protected_bytes = generated_bytes[selected_end:start_byte]
        try:
            protected_prefix = protected_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("start_byte must be a UTF-8 codepoint boundary") from exc
        return GeneratedByteCheckpoint(
            checkpoint=selected,
            protected_prefix=protected_prefix,
            exact_start_byte=start_byte,
        )

    def checkpoint_for_token_start(self, token_start_idx: int) -> SawrCheckpoint:
        if 0 <= token_start_idx < len(self._boundary_checkpoints):
            return self._boundary_checkpoints[token_start_idx]
        return self.checkpoint()

    def is_finished(self) -> bool:
        eos_id = self.eos_id
        if len(self.generated_ids) >= self._config.max_new_tokens:
            return True
        return (
            eos_id is not None
            and bool(self.generated_ids)
            and self.generated_ids[-1] == eos_id
        )

    def _next_token_logits(self) -> torch.Tensor:
        if self._next_logits is not None:
            logits = self._next_logits
            self._next_logits = None
            return logits
        if not self.generated_ids:
            raise ValueError("cannot sample without prefill logits")
        if self.past_kv is None:
            raise ValueError("cannot sample before prefill")
        last_id = self.generated_ids[-1]
        input_ids = torch.tensor([[last_id]], dtype=torch.long, device=self._device)
        with torch.no_grad():
            output = self._model(
                input_ids=input_ids,
                past_key_values=self.past_kv,
                use_cache=True,
            )
        self.past_kv = output.past_key_values
        return output.logits[:, -1, :]

    def _sample(self, logits: torch.Tensor) -> int:
        logits = logits.squeeze(0).squeeze(0).float()
        logits = self._apply_retry_repetition_penalty(logits)
        if self._config.temperature <= 0:
            return int(logits.argmax().item())

        logits = logits / self._config.temperature
        if self._config.top_k > 0:
            top_k = min(self._config.top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k).values[-1]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float("-inf")),
                logits,
            )
        if self._config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self._config.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _apply_retry_repetition_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        if self._config.retry_repetition_penalty <= 1.0:
            return logits
        token_counts = self._retry_penalty_next_token_counts()
        if not token_counts:
            return logits

        adjusted = logits.clone()
        penalty = math.log(self._config.retry_repetition_penalty)
        for token_id, count in token_counts.items():
            if 0 <= token_id < adjusted.size(-1):
                adjusted[token_id] -= penalty * count
        return adjusted

    def _retry_penalty_next_token_counts(self) -> dict[int, int]:
        current_ids = tuple(self.generated_ids)
        token_counts: dict[int, int] = {}
        for sequence in self._retry_penalty_sequences:
            base_len = len(sequence.base_ids)
            if len(current_ids) < base_len:
                continue
            if current_ids[:base_len] != sequence.base_ids:
                continue
            current_suffix = current_ids[base_len:]
            if len(current_suffix) >= len(sequence.failed_ids):
                continue
            if sequence.failed_ids[: len(current_suffix)] != current_suffix:
                continue
            token_id = sequence.failed_ids[len(current_suffix)]
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
        return token_counts


class SawrGenerator:
    """Connect local generation, boundary detection, and SAWR state decisions."""

    def __init__(
        self,
        config: SawrGenerationConfig,
        rule: EmbeddingRule,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self._rule = rule
        if model is None or tokenizer is None:
            model, tokenizer = load_sawr_model(config)
        self._model = model
        self._tokenizer = tokenizer

    def generate(
        self,
        sample_id: str,
        prompt: str,
        dataset: str,
        max_group_statements: int,
        retry_budget: int,
        global_rollback_budget: int,
        max_total_sampled_tokens: int,
        statement_retry_budget: int | None = None,
        window_retry_budget: int | None = None,
        compound_retry_budget: int | None = None,
        seed_override: int | None = None,
    ) -> SawrGenerateResult:
        active_seed = self.config.seed if seed_override is None else seed_override
        torch.manual_seed(active_seed)
        lm_prompt = build_generation_prompt(
            prompt,
            self._tokenizer,
            dataset=dataset,
            prompt_mode=self.config.prompt_mode,
        )
        context = SawrModelContext(self._model, self._tokenizer, self.config)
        context.boundary = PromptAwareBoundaryDetector(prompt=prompt, dataset=dataset)
        context.prefill(lm_prompt)
        state_machine: SawrStateMachine[SawrCheckpoint] = SawrStateMachine(
            sample_id=sample_id,
            seed=active_seed,
            max_group_statements=max_group_statements,
            retry_budget=retry_budget,
            statement_retry_budget=statement_retry_budget,
            window_retry_budget=window_retry_budget,
            compound_retry_budget=compound_retry_budget,
            rule=self._rule,
        )
        saw_controlled_body = context.boundary.saw_controlled_body
        absolute_sampled_tokens = 0
        global_rollback_count = 0
        budget_exhausted_name: str | None = None

        while not context.is_finished():
            if absolute_sampled_tokens >= max_total_sampled_tokens:
                budget_exhausted_name = "absolute_sampled_token_budget_exhausted"
                break

            step = context.forward_and_sample()
            absolute_sampled_tokens += 1

            previous_text = (
                context.generated_text
                if not step.token_text
                else context.generated_text[: -len(step.token_text)]
            )
            event_text, stop_reached = _event_text_before_stop(
                previous_text,
                step.token_text,
                self.config.stop_sequences,
            )
            if not event_text:
                context.boundary.feed_text("")
                if stop_reached:
                    break
                continue

            saw_controlled_body = (
                saw_controlled_body or context.boundary.saw_controlled_body
            )
            batch_state_snapshot = state_machine.checkpoint()
            events = context.boundary.feed_text(event_text)
            saw_controlled_body = (
                saw_controlled_body or context.boundary.saw_controlled_body
            )
            event_result = self._handle_events(
                events,
                context,
                state_machine,
                batch_state_snapshot=batch_state_snapshot,
                remaining_rollback_budget=(
                    global_rollback_budget - global_rollback_count
                ),
            )
            if event_result == "budget_exhausted":
                budget_exhausted_name = "global_rollback_budget_exhausted"
                break
            if event_result == "rolled_back":
                global_rollback_count += 1
                continue
            if stop_reached:
                break

        final_events = context.boundary.flush()
        self._handle_events(
            final_events,
            context,
            state_machine,
            allow_rollback=False,
        )
        if budget_exhausted_name is not None:
            state_machine.record_budget_exhausted(budget_exhausted_name)
        if not saw_controlled_body and not context.boundary.saw_controlled_body:
            state_machine.record_no_controlled_body()

        generated = strip_repeated_prompt_function(prompt, context.generated_text)
        generated = truncate_at_stop_sequences(generated, self.config.stop_sequences)
        return SawrGenerateResult(
            final_code=compose_final_code(prompt, generated, dataset=dataset),
            accepted_hit_count=state_machine.accepted_hit_count,
            closed_without_hit_count=state_machine.closed_without_hit_count,
            fallback_count=state_machine.fallback_count,
            candidate_count=state_machine.candidate_count,
            audit_events=state_machine.drain_audit_events(),
        )

    def _handle_events(
        self,
        events: list[BoundaryEvent],
        context: SawrModelContext,
        state_machine: SawrStateMachine[SawrCheckpoint],
        *,
        allow_rollback: bool = True,
        batch_state_snapshot: StateMachineSnapshot[SawrCheckpoint] | None = None,
        remaining_rollback_budget: int | None = None,
    ) -> str | None:
        for event in events:
            checkpoint = None
            if event.kind == "compound_started":
                base_checkpoint = context.checkpoint_for_token_start(
                    event.token_start_idx
                )
                checkpoint = (
                    base_checkpoint
                    if (
                        base_checkpoint.state_machine_state is not None
                        or batch_state_snapshot is None
                    )
                    else context.attach_state_machine_state(
                        base_checkpoint,
                        batch_state_snapshot,
                    )
                )
            elif event.kind == "simple_candidate" and event.candidate is not None:
                base_checkpoint = context.checkpoint_for_token_start(
                    event.token_start_idx
                )
                checkpoint = (
                    base_checkpoint
                    if (
                        base_checkpoint.state_machine_state is not None
                        or batch_state_snapshot is None
                    )
                    else context.attach_state_machine_state(
                        base_checkpoint,
                        batch_state_snapshot,
                    )
                )

            rollback_allowed = allow_rollback and (
                remaining_rollback_budget is None or remaining_rollback_budget > 0
            )
            decision = state_machine.observe_event(
                event,
                checkpoint,
                allow_rollback=rollback_allowed,
            )
            if decision.rollback_blocked:
                return "budget_exhausted"
            if decision.action == "rollback" and decision.rollback_checkpoint is not None:
                if not allow_rollback:
                    continue
                if (
                    remaining_rollback_budget is not None
                    and remaining_rollback_budget <= 0
                ):
                    return "budget_exhausted"
                snapshot = context.rollback(
                    decision.rollback_checkpoint,
                    remember_failed_sequence=True,
                )
                if snapshot is not None:
                    state_machine.rollback(snapshot)
                return "rolled_back"
        return None


def build_chat_prompt(prompt: str, tokenizer: Any) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return prompt
    messages = [
        {
            "role": "user",
            "content": (
                "Complete the following Python function. "
                "Output only the function body (indented), "
                "no extra function definitions, no main block.\n\n"
                + prompt
            ),
        }
    ]
    return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_generation_prompt(
    prompt: str,
    tokenizer: Any,
    *,
    dataset: str = "humaneval",
    prompt_mode: str = "completion",
) -> str:
    if prompt_mode == "completion":
        if dataset.lower() == "mbpp":
            return build_mbpp_code_prompt(prompt)
        return prompt
    if prompt_mode == "chat":
        return build_chat_prompt(prompt, tokenizer)
    raise ValueError(f"unsupported prompt_mode: {prompt_mode}")


def build_mbpp_code_prompt(prompt: str) -> str:
    return (
        "Write a correct Python function for the following programming task.\n"
        "Output only executable Python code. Do not include explanations, "
        "Markdown fences, examples, tests, print calls, or prose.\n\n"
        f"Task:\n{prompt}\n\nPython code:\n"
    )


def strip_repeated_prompt_function(prompt: str, generated: str) -> str:
    def_lines = [line for line in prompt.splitlines() if re.match(r"^def ", line)]
    if not def_lines:
        return generated
    last_def = def_lines[-1].rstrip()
    generated_lines = generated.splitlines(keepends=True)
    for index, line in enumerate(generated_lines):
        if line.rstrip() == last_def:
            return "".join(generated_lines[index + 1 :]).lstrip("\n")
    return generated


def compose_final_code(prompt: str, generated: str, *, dataset: str) -> str:
    """Build the detector-facing final code for a dataset prompt style."""

    if dataset.lower() == "humaneval":
        return prompt + generated
    if dataset.lower() == "mbpp":
        return extract_python_code_completion(strip_repeated_prompt_text(prompt, generated))
    return prompt + generated


def strip_repeated_prompt_text(prompt: str, generated: str) -> str:
    """Remove an exact natural-language prompt echo from a completion."""

    if generated.startswith(prompt):
        return generated[len(prompt) :].lstrip("\n")
    return generated


def extract_python_code_completion(text: str) -> str:
    """Extract executable Python from completion-style model output."""

    fenced = _first_python_fence(text)
    if fenced is not None:
        return _trim_to_compilable_python(fenced)

    lines = text.replace("<jupyter>", "").splitlines()
    start = _first_python_code_line(lines)
    if start is None:
        return text.strip() + ("\n" if text.strip() else "")
    return _trim_to_compilable_python("\n".join(lines[start:]))


def _first_python_fence(text: str) -> str | None:
    pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else None


def _first_python_code_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:from\s+\S+\s+import\s+|import\s+|def\s+|class\s+|@)", line):
            return index
    return None


def _trim_to_compilable_python(text: str) -> str:
    lines: list[str] = []
    for line in text.replace("<jupyter>", "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            break
        if line.startswith(
            (
                "This function",
                "You can",
                "The ",
                "In this",
                "Explanation:",
                "Output:",
                "Input:",
            )
        ):
            break
        lines.append(line)

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            compile(candidate + "\n", "<wfcllm-final-code>", "exec")
        except SyntaxError:
            continue
        return candidate + "\n"
    return ("\n".join(lines).strip() + "\n") if lines else ""


def truncate_at_stop_sequences(generated: str, stop_sequences: tuple[str, ...]) -> str:
    stop_positions = [
        position
        for stop_sequence in stop_sequences
        if (position := generated.find(stop_sequence)) != -1
    ]
    if not stop_positions:
        return generated
    return generated[: min(stop_positions)]


def _event_text_before_stop(
    previous_text: str,
    token_text: str,
    stop_sequences: tuple[str, ...],
) -> tuple[str, bool]:
    if not stop_sequences:
        return token_text, False

    combined_text = previous_text + token_text
    stop_positions = [
        position
        for stop_sequence in stop_sequences
        if (position := combined_text.find(stop_sequence)) != -1
    ]
    if not stop_positions:
        return token_text, False

    stop_position = min(stop_positions)
    token_start = len(previous_text)
    if stop_position <= token_start:
        return "", True
    return token_text[: stop_position - token_start], True


def resolve_torch_dtype(value: str) -> torch.dtype | None:
    if value == "auto":
        return None
    if value == "fp32":
        return torch.float32
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype: {value}")


def load_sawr_model(config: SawrGenerationConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    dtype = resolve_torch_dtype(config.torch_dtype)
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        kwargs: dict[str, object] = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(config.model_path, **kwargs)
        if config.device != "auto":
            model = model.to(config.device)
    model.eval()
    return model, tokenizer


def _model_device(model: Any, configured_device: str) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(configured_device)


WFCLLMCheckpoint = SawrCheckpoint
WFCLLMGenerateResult = SawrGenerateResult
WFCLLMGenerator = SawrGenerator
WFCLLMModelContext = SawrModelContext
load_wfcllm_model = load_sawr_model
