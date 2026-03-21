"""High-level watermark detection entry point."""

from __future__ import annotations

from typing import Literal

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.alignment import compare_block_contracts, rebuild_block_contracts
from wfcllm.extract.config import BlockScore, DetectionResult, ExtractConfig
from wfcllm.extract.hypothesis import HypothesisTester
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.verifier import ProjectionVerifier


class WatermarkDetector:
    """One-call watermark detection pipeline."""

    def __init__(
        self,
        config: ExtractConfig,
        encoder,
        tokenizer,
        device: str = "cuda",
    ):
        lsh_space = LSHSpace(config.secret_key, config.embed_dim, config.lsh_d)
        keying = WatermarkKeying(config.secret_key, config.lsh_d, config.lsh_gamma)
        verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=device)
        self._scorer = BlockScorer(keying, verifier, default_gamma=config.lsh_gamma)
        self._config = config

    def detect(
        self,
        code: str,
        watermark_metadata: dict | None = None,
    ) -> DetectionResult:
        hypothesis_mode = self._resolve_hypothesis_mode(watermark_metadata)
        blocks = extract_statement_blocks(code)
        if not blocks:
            return self._with_alignment(
                self._empty_result(hypothesis_mode),
                code,
                watermark_metadata,
            )

        # Only simple blocks carry watermark signal
        simple_blocks = [b for b in blocks if b.block_type == "simple"]
        if not simple_blocks:
            return self._with_alignment(
                self._empty_result(hypothesis_mode),
                code,
                watermark_metadata,
            )

        # all_blocks passed for parent_id → node_type lookup
        scores = self._scorer.score_all(
            simple_blocks,
            blocks,
            block_contracts_by_id=self._block_contracts_by_id(watermark_metadata),
        )
        self._apply_gamma_metadata(scores, watermark_metadata)

        # Simple blocks are AST leaves — inherently non-overlapping, skip DP
        tester = HypothesisTester(
            self._config.fpr_threshold,
            gamma=self._config.lsh_gamma,
            mode=hypothesis_mode,
        )
        result = tester.test(scores, total_blocks=len(simple_blocks))
        for s in scores:
            s.selected = True
        result.block_details = scores
        return self._with_alignment(result, code, watermark_metadata)

    def _with_alignment(
        self,
        result: DetectionResult,
        code: str,
        watermark_metadata: dict | None,
    ) -> DetectionResult:
        if watermark_metadata is None:
            return result
        if not self._config.adaptive_detection.require_block_contract_check:
            return result
        if "blocks" not in watermark_metadata:
            return result

        report = compare_block_contracts(
            watermark_metadata.get("blocks", []),
            rebuild_block_contracts(
                code,
                watermark_metadata=watermark_metadata,
                adaptive_gamma_config=self._config.adaptive_gamma,
                default_lsh_d=self._config.lsh_d,
            ),
        )
        result.alignment_report = report
        result.contract_valid = self._resolve_contract_validity(
            result.hypothesis_mode,
            report,
        )
        return result

    @staticmethod
    def _resolve_hypothesis_mode(
        watermark_metadata: dict | None,
    ) -> Literal["fixed", "adaptive"]:
        if watermark_metadata is None:
            return "fixed"

        adaptive_mode = watermark_metadata.get("adaptive_mode")
        if adaptive_mode == "fixed" or adaptive_mode is None:
            return "fixed"
        return "adaptive"

    @staticmethod
    def _apply_gamma_metadata(
        scores: list[BlockScore],
        watermark_metadata: dict | None,
    ) -> None:
        if watermark_metadata is None:
            return

        block_contracts = watermark_metadata.get("blocks")
        if not isinstance(block_contracts, list):
            return

        gamma_by_block_id = {
            contract.get("block_id"): float(contract["gamma_effective"])
            for contract in block_contracts
            if isinstance(contract, dict)
            and contract.get("block_id") is not None
            and "gamma_effective" in contract
        }
        for score in scores:
            if score.block_id in gamma_by_block_id:
                score.gamma_effective = gamma_by_block_id[score.block_id]

    @staticmethod
    def _block_contracts_by_id(
        watermark_metadata: dict | None,
    ) -> dict[str, dict] | None:
        if watermark_metadata is None:
            return None
        block_contracts = watermark_metadata.get("blocks")
        if not isinstance(block_contracts, list):
            return None
        by_id: dict[str, dict] = {}
        for contract in block_contracts:
            if not isinstance(contract, dict):
                continue
            block_id = contract.get("block_id")
            if block_id is None:
                continue
            by_id[str(block_id)] = contract
        return by_id or None

    @staticmethod
    def _empty_result(
        hypothesis_mode: Literal["fixed", "adaptive"],
    ) -> DetectionResult:
        return DetectionResult(
            is_watermarked=False,
            z_score=0.0,
            p_value=1.0,
            total_blocks=0,
            independent_blocks=0,
            hit_blocks=0,
            expected_hits=0.0,
            variance=0.0,
            hypothesis_mode=hypothesis_mode,
            block_details=[],
        )

    @staticmethod
    def _resolve_contract_validity(
        hypothesis_mode: Literal["fixed", "adaptive"],
        report,
    ) -> bool:
        if hypothesis_mode == "adaptive":
            return report.is_aligned
        return report.contract_valid
