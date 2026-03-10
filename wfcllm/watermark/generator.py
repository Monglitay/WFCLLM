"""Watermark-embedded code generation using custom token-by-token loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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

    @torch.no_grad()
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

                logger.debug(
                    "[simple block #%d] node=%s parent=%s entropy=%.4f margin=%.4f "
                    "target=%+d proj=%.4f passed=%s | text=%r",
                    total_blocks, event.node_type, event.parent_node_type,
                    block_entropy, margin, result.target_sign,
                    result.projection, result.passed,
                    event.block_text[:80],
                )

                if result.passed:
                    embedded_blocks += 1
                else:
                    # -------------------------------------------------------
                    # 回滚点：语句块被检测到之前的完整状态
                    # -------------------------------------------------------
                    # 计算语句块在 generated_ids 中占用的 token 数
                    block_token_count = event.token_count
                    # 截断位置（语句块开始前）
                    rollback_idx = max(0, len(generated_ids) - block_token_count)

                    # 保存回滚点
                    rollback_generated_ids = generated_ids[:rollback_idx]
                    rollback_generated_text = self._tokenizer.decode(
                        rollback_generated_ids, skip_special_tokens=True
                    )
                    rollback_kv_snapshot = self._cache_mgr.snapshot_at(
                        past_kv,
                        rollback_idx=rollback_idx,
                        current_generated_count=len(generated_ids),
                    )
                    rollback_interceptor_state = self._interceptor.save_state()
                    # 回滚点的最后一个 token（用于子循环第一次 forward 的 input_ids）
                    if rollback_generated_ids:
                        rollback_last_token_id = rollback_generated_ids[-1]
                    else:
                        # prompt 结束后立即触发块，用 prompt 最后一个 token
                        rollback_last_token_id = (
                            self._tokenizer.encode(prompt, add_special_tokens=False)[-1]
                        )

                    success = False
                    prev_retry_ids: list[int] | None = None

                    for retry_i in range(self._config.max_retries):
                        # 恢复完整回滚点状态
                        past_kv = self._cache_mgr.rollback(
                            past_kv, rollback_kv_snapshot
                        )
                        generated_ids = list(rollback_generated_ids)
                        generated_text = rollback_generated_text
                        self._interceptor.restore_state(rollback_interceptor_state)

                        # 子主循环：自由生成，直到 interceptor 触发新语句块
                        sub_input_ids = torch.tensor(
                            [[rollback_last_token_id]], dtype=torch.long, device=device
                        )
                        sub_event = None

                        for _ in range(self._config.max_new_tokens):
                            sub_output = self._model(
                                input_ids=sub_input_ids,
                                past_key_values=past_kv,
                                use_cache=True,
                            )
                            sub_logits = sub_output.logits[:, -1, :]
                            past_kv = sub_output.past_key_values

                            sub_next_id = self._sample_token(
                                sub_logits, penalty_ids=prev_retry_ids
                            )

                            if sub_next_id == eos_id:
                                break

                            generated_ids.append(sub_next_id)
                            sub_token_text = self._tokenizer.decode(
                                [sub_next_id], skip_special_tokens=True
                            )
                            generated_text += sub_token_text
                            sub_input_ids = torch.tensor(
                                [[sub_next_id]], dtype=torch.long, device=device
                            )

                            sub_event = self._interceptor.feed_token(sub_token_text)
                            if sub_event is not None and sub_event.block_type == "simple":
                                break

                        if sub_event is None or sub_event.block_type != "simple":
                            # 子循环未触发语句块（遇到 EOS 等），放弃 retry
                            logger.debug(
                                "  [retry %d/%d] sub-loop ended without block",
                                retry_i + 1, self._config.max_retries,
                            )
                            break

                        result = self._verifier.verify(
                            sub_event.block_text, v, t, margin
                        )
                        logger.debug(
                            "  [retry %d/%d] proj=%.4f target=%+d margin=%.4f "
                            "passed=%s | text=%r",
                            retry_i + 1, self._config.max_retries,
                            result.projection, result.target_sign, margin,
                            result.passed, sub_event.block_text[:80],
                        )

                        if result.passed:
                            embedded_blocks += 1
                            success = True
                            # 子循环已更新 past_kv / generated_ids / generated_text
                            # 主循环下一步从这里继续，next_id 使用子循环最后采样的 token
                            next_id = generated_ids[-1]
                            break

                        # 记录本次子循环生成的 token IDs，下次 retry 时作为惩罚目标
                        retry_block_count = len(generated_ids) - rollback_idx
                        if retry_block_count > 0:
                            prev_retry_ids = list(generated_ids[rollback_idx:])

                    if not success:
                        logger.debug(
                            "  [FAILED] block #%d exhausted %d retries",
                            total_blocks, self._config.max_retries,
                        )
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
                    logger.debug(
                        "[compound fallback] node=%s parent=%s entropy=%.4f margin=%.4f "
                        "target=%+d proj=%.4f passed=%s",
                        event.node_type, event.parent_node_type,
                        block_entropy, margin, result.target_sign,
                        result.projection, result.passed,
                    )
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

    def _sample_token(
        self,
        logits: torch.Tensor,
        penalty_ids: list[int] | None = None,
    ) -> int:
        """Sample a token from logits with temperature, top-k, top-p, and optional repetition penalty."""
        logits = logits.squeeze(0).float()

        # Repetition penalty: applied before temperature scaling
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
