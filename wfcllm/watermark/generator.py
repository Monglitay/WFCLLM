"""Watermark-embedded code generation using rejection sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.common.block_contract import BlockContract, build_block_contracts
from wfcllm.watermark.cascade import CascadeManager
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext
from wfcllm.watermark.entropy import ENTROPY_SCALE, NodeEntropyEstimator
from wfcllm.watermark.entropy_profile import EntropyProfile
from wfcllm.watermark.gamma_schedule import GammaResolution, PiecewiseQuantileSchedule, quantize_gamma
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
    block_contracts: list[BlockContract] = field(default_factory=list)
    adaptive_mode: str = "fixed"
    profile_id: str | None = None
    alignment_summary: dict[str, int | bool] = field(default_factory=dict)

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
        self._entropy_profile: EntropyProfile | None = None
        self._gamma_schedule: PiecewiseQuantileSchedule | None = None
        self._initialize_adaptive_gamma()

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
            gamma_resolver=self._resolve_gamma_for_block_text,
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
                cascade_mgr.on_compound_block_start(
                    ctx,
                    event,
                    stats_snapshot=self._snapshot_runtime_stats(stats),
                )
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

        final_event = getattr(ctx, "flush_final_event", lambda: None)()
        if final_event is not None and final_event.block_type == "simple":
            stats.total_blocks += 1
            verify_result = self._verify_block(final_event)
            if verify_result.passed:
                stats.embedded_blocks += 1
            else:
                block_cp = ctx.last_block_checkpoint
                if block_cp is None:
                    stats.failed_blocks += 1
                else:
                    retry_result = retry_loop.run(block_cp, final_event)
                    stats.retry_diagnostics.append(retry_result.diagnostics)
                    if retry_result.success:
                        stats.embedded_blocks += 1
                    else:
                        stats.failed_blocks += 1
                        pending_fallbacks.append(final_event.block_text)

        final_code = ctx.generated_text
        gamma_resolver = (
            self._resolve_gamma_for_entropy_units
            if self._is_adaptive_runtime_enabled()
            else None
        )
        block_contracts = build_block_contracts(
            final_code,
            gamma_resolver=gamma_resolver,
        )
        runtime_total_blocks = stats.total_blocks
        final_total_blocks, final_embedded_blocks = self._finalize_stats(final_code)
        stats.total_blocks = final_total_blocks
        stats.embedded_blocks = final_embedded_blocks

        return GenerateResult(
            code=final_code,
            stats=stats,
            block_contracts=block_contracts,
            adaptive_mode=self._adaptive_mode(),
            profile_id=self._profile_id(),
            alignment_summary=self._build_alignment_summary(
                runtime_total_blocks,
                block_contracts,
            ),
        )

    def _verify_block(self, event):
        """Verify a single block against LSH criteria."""
        entropy_units = self._entropy_est.estimate_block_entropy_units(event.block_text)
        block_entropy = entropy_units / ENTROPY_SCALE
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        gamma_resolution = self._resolve_gamma_for_entropy_units(entropy_units)
        valid_set = self._keying.derive(
            event.parent_node_type or "module",
            k=gamma_resolution.k,
        )
        result = self._verifier.verify(event.block_text, valid_set, margin)

        logger.debug(
            "[simple block] node=%s parent=%s entropy=%.4f margin_thresh=%.4f "
            "gamma_target=%.4f k=%d gamma_effective=%.4f\n"
            "  sig=%s in_valid=%s valid_set_size=%d min_margin=%.4f passed=%s\n"
            "  text=%r",
            event.node_type, event.parent_node_type,
            block_entropy, margin,
            gamma_resolution.gamma_target,
            gamma_resolution.k,
            gamma_resolution.gamma_effective,
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

        self._restore_runtime_stats(
            stats,
            getattr(cascade_cp, "stats_snapshot", None),
        )
        stats.cascade_blocks += 1
        pending_fallbacks.clear()
        logger.debug(
            "[CASCADE OK] rolled back to compound block start; resuming main loop"
        )

    def _adaptive_mode(self) -> str:
        """Return canonical adaptive metadata mode for the current config."""
        if self._is_adaptive_runtime_enabled():
            return self._config.adaptive_gamma.strategy
        return "fixed"

    def _profile_id(self) -> str | None:
        if not self._is_adaptive_runtime_enabled():
            return None
        return self._config.adaptive_gamma.profile_id

    def _is_adaptive_runtime_enabled(self) -> bool:
        return self._gamma_schedule is not None

    def _initialize_adaptive_gamma(self) -> None:
        adaptive_config = self._config.adaptive_gamma
        if not adaptive_config.enabled:
            return
        if adaptive_config.profile_path is None:
            return
        if adaptive_config.strategy != "piecewise_quantile":
            raise ValueError(
                f"unsupported adaptive gamma strategy: {adaptive_config.strategy}"
            )

        self._entropy_profile = EntropyProfile.load(adaptive_config.profile_path)
        anchor_quantiles = tuple(adaptive_config.anchors.keys())
        anchor_gammas = tuple(
            adaptive_config.anchors[quantile]
            for quantile in anchor_quantiles
        )
        self._gamma_schedule = PiecewiseQuantileSchedule(
            profile=self._entropy_profile,
            anchor_quantiles=anchor_quantiles,
            anchor_gammas=anchor_gammas,
        )

    def _resolve_gamma_for_block_text(self, block_text: str) -> GammaResolution:
        entropy_units = self._entropy_est.estimate_block_entropy_units(block_text)
        return self._resolve_gamma_for_entropy_units(entropy_units)

    def _resolve_gamma_for_entropy_units(self, entropy_units: int) -> GammaResolution:
        if self._gamma_schedule is not None:
            return self._gamma_schedule.resolve(entropy_units, self._config.lsh_d)
        return quantize_gamma(self._config.lsh_gamma, self._config.lsh_d)

    def _build_alignment_summary(
        self,
        runtime_total_blocks: int,
        block_contracts: list[BlockContract],
    ) -> dict[str, int | bool]:
        """Summarize final AST block metadata vs generation-time counters."""
        final_block_count = len(block_contracts)
        return {
            "final_block_count": final_block_count,
            "generator_total_blocks": runtime_total_blocks,
            "block_count_matches_total_blocks": final_block_count == runtime_total_blocks,
        }

    @staticmethod
    def _snapshot_runtime_stats(stats: EmbedStats) -> dict[str, object]:
        return {
            "total_blocks": stats.total_blocks,
            "embedded_blocks": stats.embedded_blocks,
            "failed_blocks": stats.failed_blocks,
            "fallback_blocks": stats.fallback_blocks,
            "retry_diagnostics": list(stats.retry_diagnostics),
        }

    @staticmethod
    def _restore_runtime_stats(
        stats: EmbedStats,
        snapshot: object | None,
    ) -> None:
        if not isinstance(snapshot, dict):
            return
        stats.total_blocks = int(snapshot.get("total_blocks", stats.total_blocks))
        stats.embedded_blocks = int(snapshot.get("embedded_blocks", stats.embedded_blocks))
        stats.failed_blocks = int(snapshot.get("failed_blocks", stats.failed_blocks))
        stats.fallback_blocks = int(snapshot.get("fallback_blocks", stats.fallback_blocks))
        retry_diagnostics = snapshot.get("retry_diagnostics")
        if isinstance(retry_diagnostics, list):
            stats.retry_diagnostics = list(retry_diagnostics)

    def _finalize_stats(self, final_code: str) -> tuple[int, int]:
        """Recompute final simple-block totals from the emitted code."""
        all_blocks = extract_statement_blocks(final_code)
        simple_blocks = [block for block in all_blocks if block.block_type == "simple"]
        if not simple_blocks:
            return 0, 0

        block_by_id = {block.block_id: block for block in all_blocks}
        embedded_blocks = 0
        for block in simple_blocks:
            parent_node_type = (
                block_by_id[block.parent_id].node_type
                if block.parent_id is not None
                else "module"
            )
            event = type(
                "_FinalBlockEvent",
                (),
                {
                    "block_text": block.source,
                    "block_type": "simple",
                    "node_type": block.node_type,
                    "parent_node_type": parent_node_type,
                    "token_start_idx": 0,
                    "token_count": 0,
                },
            )()
            if self._verify_block(event).passed:
                embedded_blocks += 1

        return len(simple_blocks), embedded_blocks
