"""Unified generation state management with atomic checkpoint/rollback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import StatementInterceptor, InterceptorState, InterceptEvent
from wfcllm.watermark.kv_cache import KVCacheManager, CacheSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Complete snapshot of all mutable generation state."""

    generated_ids: list[int]
    generated_text: str
    kv_snapshot: CacheSnapshot
    interceptor_state: InterceptorState
    next_logits: torch.Tensor | None = None


class GenerationContext:
    """Encapsulates all mutable state during code generation.

    Provides atomic checkpoint/rollback ensuring generated_ids, KV cache,
    and interceptor state are always in sync.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: WatermarkConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._cache_mgr = KVCacheManager()
        self._device = next(model.parameters()).device
        self._rollback_count = 0
        self._empty_cache_interval = config.cuda_empty_cache_interval

        # Mutable state
        self.generated_ids: list[int] = []
        self.generated_text: str = ""
        self.past_kv: tuple | None = None
        self.interceptor = StatementInterceptor()

        # Event tracking
        self.last_event: InterceptEvent | None = None
        self.last_block_checkpoint: Checkpoint | None = None

        # Per-step history for block-start checkpoint reconstruction.
        # Entry i = state just before the i-th generated token was sampled.
        # Stored as (generated_ids_snapshot, text, kv_seq_len,
        # interceptor_state, next_logits_snapshot).
        self._step_history: list[
            tuple[list[int], str, int, InterceptorState, torch.Tensor | None]
        ] = []

        self._eos_id: int | None = None
        self._prefill_logits: torch.Tensor | None = None
        self._next_logits: torch.Tensor | None = None

    @property
    def eos_id(self) -> int:
        if self._eos_id is None:
            self._eos_id = (
                self._config.eos_token_id
                if self._config.eos_token_id is not None
                else self._tokenizer.eos_token_id
            )
        return self._eos_id

    def prefill(self, prompt: str) -> None:
        """Run model forward on prompt to initialize KV cache."""
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_ids = input_ids.to(self._device)

        output = self._model(input_ids=input_ids, use_cache=True)
        self.past_kv = output.past_key_values
        self._prefill_logits = output.logits[:, -1, :].detach().clone()
        self._next_logits = self._prefill_logits.clone()

        self.interceptor.reset()

    def checkpoint(self) -> Checkpoint:
        """Atomically save current state."""
        return Checkpoint(
            generated_ids=list(self.generated_ids),
            generated_text=self.generated_text,
            kv_snapshot=self._cache_mgr.snapshot(self.past_kv),
            interceptor_state=self.interceptor.checkpoint(),
            next_logits=(self._next_logits.detach().clone() if self._next_logits is not None else None),
        )

    def rollback(self, cp: Checkpoint) -> None:
        """Atomically restore to a checkpointed state."""
        old_kv = self.past_kv
        self.past_kv = self._cache_mgr.rollback(self.past_kv, cp.kv_snapshot)

        if old_kv is not self.past_kv:
            del old_kv
            self._rollback_count += 1
            if self._rollback_count % self._empty_cache_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.generated_ids = list(cp.generated_ids)
        self.generated_text = cp.generated_text
        self.interceptor.rollback(cp.interceptor_state)
        self._next_logits = cp.next_logits.detach().clone() if cp.next_logits is not None else None

        # Trim step history to match rolled-back length
        rollback_len = len(cp.generated_ids)
        if len(self._step_history) > rollback_len:
            self._step_history = self._step_history[:rollback_len]

        # Reset event state
        self.last_event = None
        self.last_block_checkpoint = None

    def forward_and_sample(self, penalty_ids: list[int] | None = None) -> int:
        """Single-step forward + sample, atomically updating all state.

        Returns the sampled token id.
        """
        # Capture pre-forward state for lazy checkpoint
        # IMPORTANT: capture kv seq_len BEFORE model forward, so it aligns
        # with the pre-feed generated_ids length (both exclude current token)
        pre_forward_ids_len = len(self.generated_ids)
        pre_forward_text = self.generated_text
        pre_forward_kv_seq_len = (
            self.past_kv[0][0].shape[2] if self.past_kv is not None else 0
        )
        pre_forward_interceptor_state = self.interceptor.checkpoint()
        pre_forward_next_logits = (
            self._next_logits.detach().clone()
            if self._next_logits is not None
            else None
        )

        # Record this step's pre-forward state in history
        self._step_history.append((
            list(self.generated_ids),
            pre_forward_text,
            pre_forward_kv_seq_len,
            pre_forward_interceptor_state,
            pre_forward_next_logits,
        ))

        # Model forward
        if self._next_logits is not None:
            logits = self._next_logits
            self._next_logits = None
        else:
            last_id = self.generated_ids[-1] if self.generated_ids else 0
            input_ids = torch.tensor([[last_id]], dtype=torch.long, device=self._device)
            output = self._model(
                input_ids=input_ids,
                past_key_values=self.past_kv,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            self.past_kv = output.past_key_values

        # Sample
        next_id = self._sample(logits, penalty_ids)

        # Update generated state
        self.generated_ids.append(next_id)
        token_text = self._tokenizer.decode([next_id], skip_special_tokens=True)
        self.generated_text += token_text

        # Capture interceptor state AFTER model forward but BEFORE feed_token
        # This ensures the interceptor snapshot does not include the new block
        pre_feed_interceptor_state = self.interceptor.checkpoint()

        # Feed interceptor
        event = self.interceptor.feed_token(token_text)

        # Track events with lazy checkpoint materialization
        self.last_event = event
        if event is not None:
            # Build checkpoint at block START (token_start_idx), not at the
            # trigger token. This ensures retry sub-loops regenerate the entire
            # block and repetition penalty can actually influence the content.
            block_start = event.token_start_idx
            if 0 <= block_start < len(self._step_history):
                frame = self._step_history[block_start]
                self.last_block_checkpoint = Checkpoint(
                    generated_ids=frame[0],
                    generated_text=frame[1],
                    kv_snapshot=CacheSnapshot(seq_len=frame[2]),
                    interceptor_state=frame[3],
                    next_logits=frame[4].detach().clone() if frame[4] is not None else None,
                )
            else:
                # Fallback: use pre-forward state (old behaviour)
                self.last_block_checkpoint = Checkpoint(
                    generated_ids=list(self.generated_ids[:pre_forward_ids_len]),
                    generated_text=pre_forward_text,
                    kv_snapshot=CacheSnapshot(seq_len=pre_forward_kv_seq_len),
                    interceptor_state=pre_feed_interceptor_state,
                    next_logits=(
                        pre_forward_next_logits.detach().clone()
                        if pre_forward_next_logits is not None
                        else None
                    ),
                )
        else:
            self.last_block_checkpoint = None

        return next_id

    def is_finished(self) -> bool:
        """Check if generation should stop."""
        if len(self.generated_ids) >= self._config.max_new_tokens:
            return True
        if self.generated_ids and self.generated_ids[-1] == self.eos_id:
            return True
        return False

    def flush_final_event(self) -> InterceptEvent | None:
        """Emit the last pending simple block when generation stops at EOF."""
        event = self.interceptor.finalize_pending_simple_block()
        self.last_event = event
        if event is None:
            self.last_block_checkpoint = None
            return None

        block_start = event.token_start_idx
        if 0 <= block_start < len(self._step_history):
            frame = self._step_history[block_start]
            self.last_block_checkpoint = Checkpoint(
                generated_ids=frame[0],
                generated_text=frame[1],
                kv_snapshot=CacheSnapshot(seq_len=frame[2]),
                interceptor_state=frame[3],
                next_logits=frame[4].detach().clone() if frame[4] is not None else None,
            )
        else:
            self.last_block_checkpoint = self.checkpoint()
        return event

    def _sample(
        self,
        logits: torch.Tensor,
        penalty_ids: list[int] | None = None,
    ) -> int:
        """Sample a token with temperature, top-k, top-p, repetition penalty."""
        logits = logits.squeeze(0).float()

        # Repetition penalty
        if penalty_ids and self._config.repetition_penalty != 1.0:
            penalty = self._config.repetition_penalty
            for tid in penalty_ids:
                if 0 <= tid < logits.size(0):
                    if logits[tid] > 0:
                        logits[tid] /= penalty
                    else:
                        logits[tid] *= penalty

        if self._config.temperature > 0:
            logits = logits / self._config.temperature
        else:
            # Greedy: return argmax
            return logits.argmax().item()

        if self._config.top_k > 0:
            top_k = min(self._config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k).values[-1]
            logits[indices_to_remove] = float("-inf")

        if self._config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > self._config.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
