"""LSH verification + adaptive-gamma resolution slice of WatermarkGenerator.

Holds a back-reference to the orchestrator and reads its mutable attributes
(_config, _entropy_est, _verifier, _keying, _entropy_profile, _gamma_schedule)
directly. This preserves the existing test pattern of monkey-patching
`generator._verify_block`, `generator._verifier.verify`, etc.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from wfcllm.lang.python.parser import extract_statement_blocks
from wfcllm.common.block_contract import BlockContract
from wfcllm.watermark.adaptive_gamma.entropy import ENTROPY_SCALE
from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
from wfcllm.watermark.adaptive_gamma.schedule import (
    GammaResolution,
    PiecewiseQuantileSchedule,
    quantize_gamma,
)
from wfcllm.watermark.diagnostics import FailureReason

if TYPE_CHECKING:
    from wfcllm.watermark.orchestrator import WatermarkGenerator

logger = logging.getLogger(__name__)


class SemanticChannel:
    def __init__(self, orchestrator: "WatermarkGenerator") -> None:
        self._orch = orchestrator

    def verify_block(self, event):
        """Verify a single block against LSH criteria."""
        orch = self._orch
        entropy_units = orch._entropy_est.estimate_block_entropy_units(event.block_text)
        block_entropy = entropy_units / ENTROPY_SCALE
        margin = orch._entropy_est.compute_margin(block_entropy, orch._config)
        gamma_resolution = self.resolve_gamma_for_entropy_units(entropy_units)
        valid_set = orch._keying.derive(
            event.parent_node_type or "module",
            k=gamma_resolution.k,
        )
        result = orch._verifier.verify(event.block_text, valid_set, margin)

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

    def classify_failure_reason(self, event, verify_result) -> str:
        orch = self._orch
        if verify_result.passed:
            return FailureReason.unknown.value
        entropy_units = orch._entropy_est.estimate_block_entropy_units(event.block_text)
        block_entropy = entropy_units / ENTROPY_SCALE
        margin_threshold = orch._entropy_est.compute_margin(block_entropy, orch._config)
        in_valid_set = verify_result.in_valid_set
        if in_valid_set is None:
            return FailureReason.unknown.value
        margin_passed = verify_result.min_margin > margin_threshold
        if not in_valid_set and margin_passed:
            return FailureReason.signature_miss.value
        if in_valid_set and not margin_passed:
            return FailureReason.margin_miss.value
        if not in_valid_set and not margin_passed:
            return FailureReason.signature_and_margin_miss.value
        return FailureReason.unknown.value

    def resolve_gamma_for_block_text(self, block_text: str) -> GammaResolution:
        orch = self._orch
        entropy_units = orch._entropy_est.estimate_block_entropy_units(block_text)
        return self.resolve_gamma_for_entropy_units(entropy_units)

    def resolve_gamma_for_entropy_units(self, entropy_units: int) -> GammaResolution:
        orch = self._orch
        if orch._gamma_schedule is not None:
            return orch._gamma_schedule.resolve(entropy_units, orch._config.lsh_d)
        return quantize_gamma(orch._config.lsh_gamma, orch._config.lsh_d)

    def is_adaptive_runtime_enabled(self) -> bool:
        return self._orch._gamma_schedule is not None

    def adaptive_mode(self) -> str:
        if self.is_adaptive_runtime_enabled():
            return self._orch._config.adaptive_gamma.strategy
        return "fixed"

    def profile_id(self) -> str | None:
        if not self.is_adaptive_runtime_enabled():
            return None
        return self._orch._config.adaptive_gamma.profile_id

    def initialize_adaptive_gamma(self) -> None:
        orch = self._orch
        adaptive_config = orch._config.adaptive_gamma
        if not adaptive_config.enabled:
            return
        if adaptive_config.profile_path is None:
            return
        if adaptive_config.strategy != "piecewise_quantile":
            raise ValueError(
                f"unsupported adaptive gamma strategy: {adaptive_config.strategy}"
            )

        orch._entropy_profile = EntropyProfile.load(adaptive_config.profile_path)
        anchor_quantiles = tuple(adaptive_config.anchors.keys())
        anchor_gammas = tuple(
            adaptive_config.anchors[quantile]
            for quantile in anchor_quantiles
        )
        orch._gamma_schedule = PiecewiseQuantileSchedule(
            profile=orch._entropy_profile,
            anchor_quantiles=anchor_quantiles,
            anchor_gammas=anchor_gammas,
        )

    def build_alignment_summary(
        self,
        runtime_total_blocks: int,
        block_contracts: list[BlockContract],
    ) -> dict[str, int | bool]:
        final_block_count = len(block_contracts)
        return {
            "final_block_count": final_block_count,
            "generator_total_blocks": runtime_total_blocks,
            "block_count_matches_total_blocks": final_block_count == runtime_total_blocks,
        }

    def finalize_stats(self, final_code: str) -> tuple[int, int]:
        """Recompute final simple-block totals from the emitted code."""
        orch = self._orch
        all_blocks = extract_statement_blocks(final_code)
        simple_blocks = [block for block in all_blocks if block.block_type == "simple"]
        if not simple_blocks:
            return 0, 0
        if not orch._semantic_channel_enabled():
            return len(simple_blocks), 0

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
            if orch._verify_block(event).passed:
                embedded_blocks += 1

        return len(simple_blocks), embedded_blocks
