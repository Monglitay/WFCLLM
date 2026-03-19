"""Watermark-embedded code generation using rejection sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from wfcllm.watermark.cascade import CascadeManager
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.retry_loop import RetryLoop, RetryDiagnostics
from wfcllm.watermark.verifier import ProjectionVerifier

logger = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    """Watermark embedding statistics."""

    total_blocks: int = 0
    embedded_blocks: int = 0
    failed_blocks: int = 0
    fallback_blocks: int = 0
    cascade_blocks: int = 0
    retry_diagnostics: list[RetryDiagnostics] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Result of watermark-embedded generation."""

    code: str
    stats: EmbedStats

    # Backward-compatible properties
    @property
    def total_blocks(self) -> int:
        return self.stats.total_blocks

    @property
    def embedded_blocks(self) -> int:
        return self.stats.embedded_blocks

    @property
    def failed_blocks(self) -> int:
        return self.stats.failed_blocks

    @property
    def fallback_blocks(self) -> int:
        return self.stats.fallback_blocks


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

        self._entropy_est = NodeEntropyEstimator()
        self._lsh_space = LSHSpace(
            config.secret_key, config.encoder_embed_dim, config.lsh_d
        )
        self._keying = WatermarkKeying(
            config.secret_key, config.lsh_d, config.lsh_gamma
        )
        self._verifier = ProjectionVerifier(
            encoder, encoder_tokenizer,
            lsh_space=self._lsh_space,
            device=config.encoder_device,
        )

        _STRUCTURAL_KEYWORDS = [
            "import", "return", "def", "class", "if", "else", "elif",
            "for", "while", "with", "try", "except", "finally", "pass",
            "break", "continue", "raise", "yield", "lambda",
            "and", "or", "not", "in", "is", "from", "as", "assert",
            "del", "global", "nonlocal", "\n", " ", "\t",
        ]
        self._structural_token_ids: set[int] = {
            tid
            for kw in _STRUCTURAL_KEYWORDS
            for tid in self._tokenizer.encode(kw, add_special_tokens=False)
        }

    @property
    def config(self) -> WatermarkConfig:
        return self._config

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerateResult:
        """Generate code with watermark embedding."""
        ctx = GenerationContext(
            model=self._model,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        ctx.prefill(prompt)

        stats = EmbedStats()
        cascade_mgr = CascadeManager(self._config)
        retry_loop = RetryLoop(
            ctx=ctx,
            config=self._config,
            verifier=self._verifier,
            keying=self._keying,
            entropy_est=self._entropy_est,
            structural_token_ids=self._structural_token_ids,
        )
        pending_fallbacks: list[str] = []

        while not ctx.is_finished():
            next_id = ctx.forward_and_sample()

            if next_id == ctx.eos_id:
                break

            event = ctx.last_event
            if event is None:
                continue

            if event.block_type == "compound":
                cascade_mgr.on_compound_block_start(ctx, event)
                continue

            # Simple block
            stats.total_blocks += 1
            verify_result = self._verify_block(event)

            if verify_result.passed:
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                continue

            # Verification failed → retry
            block_cp = ctx.last_block_checkpoint
            if block_cp is None:
                # Shouldn't happen, but degrade gracefully
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                continue

            retry_result = retry_loop.run(block_cp, event)
            stats.retry_diagnostics.append(retry_result.diagnostics)

            if retry_result.success:
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                logger.debug(
                    "[RETRY OK] block #%d after %d attempts",
                    stats.total_blocks, retry_result.attempts,
                )
            else:
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                cascade_mgr.on_simple_block_failed(event.block_text)
                logger.debug(
                    "[RETRY FAILED] block #%d exhausted %d retries",
                    stats.total_blocks, retry_result.attempts,
                )

                if cascade_mgr.should_cascade():
                    self._try_cascade(
                        ctx, cascade_mgr, retry_loop, stats, pending_fallbacks
                    )

        return GenerateResult(code=ctx.generated_text, stats=stats)

    def _verify_block(self, event):
        """Verify a single block against LSH criteria."""
        block_entropy = self._entropy_est.estimate_block_entropy(event.block_text)
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(event.parent_node_type or "module")
        result = self._verifier.verify(event.block_text, valid_set, margin)

        logger.debug(
            "[simple block] node=%s parent=%s entropy=%.4f margin_thresh=%.4f\n"
            "  sig=%s in_valid=%s valid_set_size=%d min_margin=%.4f passed=%s\n"
            "  text=%r",
            event.node_type, event.parent_node_type,
            block_entropy, margin,
            result.lsh_signature,
            result.lsh_signature in valid_set,
            len(valid_set), result.min_margin, result.passed,
            event.block_text[:80],
        )
        return result

    def _try_cascade(self, ctx, cascade_mgr, retry_loop, stats, pending_fallbacks):
        """Active cascade: rollback to compound block start, then resume main loop.

        We intentionally do NOT verify the first regenerated compound event here.
        Incremental parsing emits compound blocks at header-complete time, so
        verifying inside this helper would use intermediate text that does not
        match the final AST seen by extraction. After rollback, the outer
        generation loop continues normally and verifies the regenerated simple
        blocks against final-text boundaries.
        """
        cascade_cp = cascade_mgr.cascade(ctx)
        if cascade_cp is None:
            return

        stats.cascade_blocks += 1
        pending_fallbacks.clear()
        logger.debug(
            "[CASCADE OK] rolled back to compound block start; resuming main loop"
        )
