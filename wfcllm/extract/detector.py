"""High-level watermark detection entry point."""

from __future__ import annotations

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.config import DetectionResult, ExtractConfig
from wfcllm.extract.dp_selector import DPSelector
from wfcllm.extract.hypothesis import HypothesisTester
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
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
        keying = WatermarkKeying(config.secret_key, config.embed_dim)
        verifier = ProjectionVerifier(encoder, tokenizer, device=device)
        self._scorer = BlockScorer(keying, verifier)
        self._dp = DPSelector()
        self._tester = HypothesisTester(config.z_threshold)

    def detect(self, code: str) -> DetectionResult:
        """Run the full extraction and verification pipeline.

        Args:
            code: Python source code to test for watermarks.

        Returns:
            DetectionResult with verdict, statistics, and per-block details.
        """
        blocks = extract_statement_blocks(code)
        if not blocks:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                total_blocks=0,
                independent_blocks=0,
                hit_blocks=0,
                block_details=[],
            )

        scores = self._scorer.score_all(blocks)
        selected_ids = set(self._dp.select(blocks, scores))

        # Mark selected blocks
        for s in scores:
            s.selected = s.block_id in selected_ids

        selected_scores = [s for s in scores if s.selected]
        result = self._tester.test(selected_scores, total_blocks=len(blocks))

        # Replace block_details with ALL blocks (not just selected)
        result.block_details = scores
        return result
