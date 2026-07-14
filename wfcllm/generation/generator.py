from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from wfcllm.generation.boundary import (
    BoundaryDetectorState,
    BoundaryEvent,
    PromptAwareBoundaryDetector,
)
from wfcllm.method.config import SawrGenerationConfig
from wfcllm.semantic.rules import EmbeddingRule
from wfcllm.generation.state_machine import AuditEvent, SawrStateMachine, StateMachineSnapshot


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


@dataclass(frozen=True)
class _GeneratedStep:
    token_id: int
    token_text: str


@dataclass(frozen=True)
class _RetryPenaltySequence:
    base_ids: tuple[int, ...]
    failed_ids: tuple[int, ...]


class SawrModelContext:
    """Small causal-LM generation context with rollback support."""

    _MAX_RETRY_PENALTY_SEQUENCES = 128

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

    def checkpoint(
        self,
        state_machine_state: StateMachineSnapshot[SawrCheckpoint] | None = None,
    ) -> SawrCheckpoint:
        if self.past_kv is None:
            raise ValueError("cannot checkpoint before prefill")
        boundary_state = (
            self.boundary.checkpoint() if self.boundary is not None else None
        )
        return SawrCheckpoint(
            generated_ids=list(self.generated_ids),
            generated_text=self.generated_text,
            kv_snapshot=self._cache_mgr.snapshot(self.past_kv),
            boundary_state=boundary_state,
            next_logits=(
                self._next_logits.detach().clone()
                if self._next_logits is not None
                else None
            ),
            state_machine_state=state_machine_state,
        )

    def rollback(
        self,
        checkpoint: SawrCheckpoint,
        *,
        remember_failed_sequence: bool = False,
    ) -> StateMachineSnapshot[SawrCheckpoint] | None:
        if self.past_kv is None:
            raise ValueError("cannot rollback before prefill")
        if remember_failed_sequence:
            self._remember_failed_sequence(checkpoint)
        self.past_kv = self._cache_mgr.rollback(self.past_kv, checkpoint.kv_snapshot)
        self.generated_ids = list(checkpoint.generated_ids)
        self.generated_text = checkpoint.generated_text
        if self.boundary is not None and checkpoint.boundary_state is not None:
            self.boundary.rollback(checkpoint.boundary_state)
        self._next_logits = (
            checkpoint.next_logits.detach().clone()
            if checkpoint.next_logits is not None
            else None
        )
        self._step_history = self._step_history[: len(self.generated_ids)]
        boundary_count = (
            len(checkpoint.boundary_state.token_boundaries)
            if checkpoint.boundary_state is not None
            else 0
        )
        self._boundary_checkpoints = self._boundary_checkpoints[:boundary_count]
        return checkpoint.state_machine_state

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
        checkpoint = self.checkpoint(state_machine.checkpoint())
        self._boundary_checkpoints.append(checkpoint)

    def forward_and_sample(self) -> _GeneratedStep:
        step_checkpoint = self.checkpoint()
        logits = self._next_token_logits()
        next_id = self._sample(logits)
        token_text = self._tokenizer.decode([next_id], skip_special_tokens=True)

        self.generated_ids.append(next_id)
        self.generated_text += token_text
        self._step_history.append(step_checkpoint)
        if token_text:
            self._boundary_checkpoints.append(step_checkpoint)
        return _GeneratedStep(token_id=next_id, token_text=token_text)

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
        prefix_observer: Any | None = None,
    ) -> SawrGenerateResult:
        active_seed = self.config.seed if seed_override is None else seed_override
        torch.manual_seed(active_seed)
        lm_prompt = build_generation_prompt(
            prompt,
            self._tokenizer,
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
            if not step.token_text:
                continue

            _notify_prefix_observer(
                prefix_observer,
                _compose_final_code(prompt, context.generated_text, self.config.stop_sequences),
            )

            previous_text = context.generated_text[: -len(step.token_text)]
            event_text, stop_reached = _event_text_before_stop(
                previous_text,
                step.token_text,
                self.config.stop_sequences,
            )
            if not event_text:
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
                _notify_prefix_observer(
                    prefix_observer,
                    _compose_final_code(
                        prompt,
                        context.generated_text,
                        self.config.stop_sequences,
                    ),
                )
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
        final_code = prompt + generated
        _flush_prefix_observer(prefix_observer, final_code)
        return SawrGenerateResult(
            final_code=final_code,
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
                    if base_checkpoint.state_machine_state is not None
                    else replace(
                        base_checkpoint,
                        state_machine_state=batch_state_snapshot,
                    )
                )
            elif event.kind == "simple_candidate" and event.candidate is not None:
                base_checkpoint = context.checkpoint_for_token_start(
                    event.token_start_idx
                )
                checkpoint = (
                    base_checkpoint
                    if base_checkpoint.state_machine_state is not None
                    else replace(
                        base_checkpoint,
                        state_machine_state=batch_state_snapshot,
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


def _compose_final_code(
    prompt: str,
    generated_text: str,
    stop_sequences: tuple[str, ...],
) -> str:
    generated = strip_repeated_prompt_function(prompt, generated_text)
    generated = truncate_at_stop_sequences(generated, stop_sequences)
    return prompt + generated


def _notify_prefix_observer(observer: Any | None, final_code_prefix: str) -> None:
    if observer is None:
        return
    observer.observe_prefix(final_code_prefix)


def _flush_prefix_observer(observer: Any | None, final_code: str) -> None:
    if observer is None:
        return
    observer.flush(final_code)


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
    prompt_mode: str = "completion",
) -> str:
    if prompt_mode == "completion":
        return prompt
    if prompt_mode == "chat":
        return build_chat_prompt(prompt, tokenizer)
    raise ValueError(f"unsupported prompt_mode: {prompt_mode}")


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
