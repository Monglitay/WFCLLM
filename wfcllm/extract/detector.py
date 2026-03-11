"""High-level watermark detection entry point."""

from __future__ import annotations

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.config import DetectionResult, ExtractConfig
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
        self._scorer = BlockScorer(keying, verifier)
        self._tester = HypothesisTester(config.fpr_threshold, gamma=config.lsh_gamma)

    def detect(self, code: str) -> DetectionResult:
        blocks = extract_statement_blocks(code)
        if not blocks:
            return DetectionResult(
                is_watermarked=False, z_score=0.0, p_value=1.0,
                total_blocks=0, independent_blocks=0, hit_blocks=0,
                block_details=[],
            )

        # Only simple blocks carry watermark signal
        simple_blocks = [b for b in blocks if b.block_type == "simple"]
        if not simple_blocks:
            return DetectionResult(
                is_watermarked=False, z_score=0.0, p_value=1.0,
                total_blocks=0, independent_blocks=0, hit_blocks=0,
                block_details=[],
            )

        # all_blocks passed for parent_id → node_type lookup
        scores = self._scorer.score_all(simple_blocks, blocks)

        # Simple blocks are AST leaves — inherently non-overlapping, skip DP
        result = self._tester.test(scores, total_blocks=len(simple_blocks))
        for s in scores:
            s.selected = True
        result.block_details = scores
        return result
