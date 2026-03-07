"""Watermark-embedded code generation using custom token-by-token loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.interceptor import StatementInterceptor
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.kv_cache import KVCacheManager
from wfcllm.watermark.verifier import ProjectionVerifier


@dataclass
class GenerateResult:
    """Result of watermark-embedded generation."""

    code: str
    total_blocks: int
    embedded_blocks: int
    failed_blocks: int
    fallback_blocks: int


class WatermarkGenerator:
    """Code generator with watermark embedding via rejection sampling."""

    def __init__(
        self,
        model,
        tokenizer,
        encoder,
        encoder_tokenizer,
        config: WatermarkConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

        self._interceptor = StatementInterceptor()
        self._entropy_est = NodeEntropyEstimator()
        self._keying = WatermarkKeying(config.secret_key, config.encoder_embed_dim)
        self._verifier = ProjectionVerifier(
            encoder, encoder_tokenizer, device=config.encoder_device
        )
        self._cache_mgr = KVCacheManager()

    def generate(self, prompt: str) -> GenerateResult:
        """Generate code with watermark embedding.

        Args:
            prompt: The input prompt for code generation.

        Returns:
            GenerateResult with generated code and embedding statistics.
        """
        device = next(self._model.parameters()).device

        # Tokenize prompt
        input_ids = torch.tensor(
            [self._tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        past_kv = None
        generated_ids: list[int] = []
        generated_text = ""

        total_blocks = 0
        embedded_blocks = 0
        failed_blocks = 0
        fallback_blocks = 0
        pending_fallbacks: list[str] = []

        self._interceptor.reset()
        eos_id = self._config.eos_token_id or self._tokenizer.eos_token_id

        for _ in range(self._config.max_new_tokens):
            # Forward pass
            output = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            past_kv = output.past_key_values

            # Sample next token
            next_id = self._sample_token(logits)

            if next_id == eos_id:
                break

            generated_ids.append(next_id)
            token_text = self._tokenizer.decode([next_id], skip_special_tokens=True)
            generated_text += token_text

            # Feed to interceptor
            event = self._interceptor.feed_token(token_text)

            if event is not None and event.block_type == "simple":
                total_blocks += 1

                block_entropy = self._entropy_est.estimate_block_entropy(
                    event.block_text
                )
                margin = self._entropy_est.compute_margin(block_entropy, self._config)

                v, t = self._keying.derive(
                    event.parent_node_type or "module", event.node_type
                )

                result = self._verifier.verify(event.block_text, v, t, margin)

                if result.passed:
                    embedded_blocks += 1
                else:
                    snapshot = self._cache_mgr.snapshot(past_kv)
                    success = False

                    for _ in range(self._config.max_retries):
                        past_kv = self._cache_mgr.rollback(past_kv, snapshot)
                        rollback_count = event.token_count
                        if rollback_count > 0 and rollback_count <= len(generated_ids):
                            generated_ids = generated_ids[:-rollback_count]
                            generated_text = self._tokenizer.decode(
                                generated_ids, skip_special_tokens=True
                            )

                        regen_ids, regen_text, past_kv = self._regenerate_block(
                            past_kv, device, rollback_count
                        )
                        generated_ids.extend(regen_ids)
                        generated_text += regen_text

                        result = self._verifier.verify(regen_text, v, t, margin)
                        if result.passed:
                            embedded_blocks += 1
                            success = True
                            break

                    if not success:
                        failed_blocks += 1
                        pending_fallbacks.append(event.block_text)

            elif event is not None and event.block_type == "compound":
                if self._config.enable_fallback and pending_fallbacks:
                    total_blocks += 1
                    block_entropy = self._entropy_est.estimate_block_entropy(
                        event.block_text
                    )
                    margin = self._entropy_est.compute_margin(block_entropy, self._config)
                    v, t = self._keying.derive(
                        event.parent_node_type or "module", event.node_type
                    )
                    result = self._verifier.verify(event.block_text, v, t, margin)
                    if result.passed:
                        fallback_blocks += 1
                        pending_fallbacks.clear()

            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)

        return GenerateResult(
            code=generated_text,
            total_blocks=total_blocks,
            embedded_blocks=embedded_blocks,
            failed_blocks=failed_blocks,
            fallback_blocks=fallback_blocks,
        )

    def _sample_token(self, logits: torch.Tensor) -> int:
        """Sample a token from logits with temperature, top-k, top-p."""
        logits = logits.squeeze(0).float()

        if self._config.temperature > 0:
            logits = logits / self._config.temperature

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

    def _regenerate_block(
        self, past_kv, device, num_tokens: int
    ) -> tuple[list[int], str, tuple]:
        """Re-generate a fixed number of tokens from the rollback point."""
        regen_ids: list[int] = []
        input_ids = torch.tensor(
            [[self._tokenizer.eos_token_id or 0]], dtype=torch.long, device=device
        )

        for _ in range(num_tokens):
            output = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            past_kv = output.past_key_values
            next_id = self._sample_token(logits)
            regen_ids.append(next_id)
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)

        regen_text = self._tokenizer.decode(regen_ids, skip_special_tokens=True)
        return regen_ids, regen_text, past_kv
