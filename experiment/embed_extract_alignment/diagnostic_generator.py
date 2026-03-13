"""DiagnosticGenerator: WatermarkGenerator subclass that records EmbedEvents."""
from __future__ import annotations

import logging

import torch

from wfcllm.watermark.cascade import CascadeManager
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext
from wfcllm.watermark.generator import EmbedStats, GenerateResult, WatermarkGenerator
from wfcllm.watermark.retry_loop import RetryLoop

from experiment.embed_extract_alignment.models import EmbedEvent

logger = logging.getLogger(__name__)


class DiagnosticGenerator(WatermarkGenerator):
    """Subclass of WatermarkGenerator that records all EmbedEvents.

    Usage:
        gen = DiagnosticGenerator(model, tokenizer, encoder, enc_tokenizer, config)
        result = gen.generate(prompt)
        events = gen.embed_events   # list[EmbedEvent], populated after generate()
    """

    embed_events: list[EmbedEvent]

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerateResult:
        """Generate with watermark, recording all embed attempts into self.embed_events."""
        self.embed_events = []

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
                self._diag_try_passive_fallback(ctx, event, stats, pending_fallbacks)
                continue

            # ── Simple block ──────────────────────────────────────────────
            stats.total_blocks += 1
            verify_result = self._verify_block(event)

            if verify_result.passed:
                # Recording point 1: simple passed (no retry needed)
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=True,
                ))
                continue

            block_cp = ctx.last_block_checkpoint
            if block_cp is None:
                # Recording point 3: failed, no checkpoint for retry
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=False,
                ))
                continue

            retry_result = retry_loop.run(block_cp, event)
            stats.retry_diagnostics.append(retry_result.diagnostics)

            if retry_result.success:
                # Recording point 2: simple passed after retry
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=True,
                ))
                logger.debug("[RETRY OK] block #%d", stats.total_blocks)
            else:
                # Recording point 3: retry exhausted
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                cascade_mgr.on_simple_block_failed(event.block_text)
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=False,
                ))
                logger.debug("[RETRY FAILED] block #%d", stats.total_blocks)

                if cascade_mgr.should_cascade():
                    self._diag_try_cascade(ctx, cascade_mgr, retry_loop, stats, pending_fallbacks)

        return GenerateResult(code=ctx.generated_text, stats=stats)

    def _diag_try_passive_fallback(self, ctx, event, stats, pending_fallbacks) -> None:
        """Passive fallback with EmbedEvent recording (points 4 & 5)."""
        if not self._config.enable_fallback or not pending_fallbacks:
            return

        stats.total_blocks += 1
        block_entropy = self._entropy_est.estimate_block_entropy(event.block_text)
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(event.parent_node_type or "module")
        result = self._verifier.verify(event.block_text, valid_set, margin)

        # Recording point 4 (passed) or 5 (failed)
        self.embed_events.append(EmbedEvent(
            path="fallback",
            block_text=event.block_text,
            parent_node_type=event.parent_node_type or "module",
            node_type=event.node_type,
            passed=result.passed,
        ))

        if result.passed:
            stats.fallback_blocks += 1
            pending_fallbacks.clear()
            logger.debug("[FALLBACK OK] compound node=%s", event.node_type)
        else:
            logger.debug("[FALLBACK MISS] compound node=%s", event.node_type)

    def _diag_try_cascade(self, ctx, cascade_mgr, retry_loop, stats, pending_fallbacks) -> None:
        """Cascade regeneration with EmbedEvent recording (point 6).

        Records the NEWLY REGENERATED compound block's text/parent (not the triggering block).
        """
        cascade_cp = cascade_mgr.cascade(ctx)
        if cascade_cp is None:
            return

        compound_event = None
        for _ in range(self._config.max_new_tokens):
            next_id = ctx.forward_and_sample()
            if next_id == ctx.eos_id:
                break
            event = ctx.last_event
            if event is not None and event.block_type == "compound":
                compound_event = event
                break

        if compound_event is None:
            logger.debug("[CASCADE FAILED] could not regenerate compound block")
            return

        block_entropy = self._entropy_est.estimate_block_entropy(compound_event.block_text)
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(compound_event.parent_node_type or "module")
        result = self._verifier.verify(compound_event.block_text, valid_set, margin)

        # Recording point 6: cascade passed or failed
        # block_text/parent_node_type = regenerated compound_event (per spec)
        self.embed_events.append(EmbedEvent(
            path="cascade",
            block_text=compound_event.block_text,
            parent_node_type=compound_event.parent_node_type or "module",
            node_type=compound_event.node_type,
            passed=result.passed,
        ))

        if result.passed:
            stats.cascade_blocks += 1
            pending_fallbacks.clear()
            logger.debug("[CASCADE OK] regenerated compound block passed")
        else:
            logger.debug("[CASCADE FAILED] regenerated compound block did not pass")
