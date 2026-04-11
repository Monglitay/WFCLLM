"""Z-score hypothesis testing for watermark detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

from wfcllm.extract.config import BlockScore, DetectionResult
from wfcllm.watermark.token_channel.config import TokenChannelConfig


@dataclass(frozen=True)
class SemanticDetectionResult:
    """Semantic-channel summary carried alongside joint outputs."""

    is_watermarked: bool
    z_score: float
    p_value: float
    total_blocks: int
    independent_blocks: int
    hit_blocks: int
    expected_hits: float
    variance: float
    hypothesis_mode: Literal["fixed", "adaptive"]


@dataclass(frozen=True)
class JointDetectionResult:
    """Final explicit fusion result."""

    semantic_z: float
    lexical_z: float
    joint_score: float
    p_joint: float
    prediction: bool
    confidence: float
    rationale: str


@dataclass(frozen=True)
class LexicalDetectionResult:
    """Replay statistics for lexical token-channel evidence."""

    num_positions_scored: int
    num_green_hits: int
    green_fraction: float
    lexical_z_score: float
    lexical_p_value: float

    @property
    def z_score(self) -> float:
        return self.lexical_z_score

    @property
    def p_value(self) -> float:
        return self.lexical_p_value

    @classmethod
    def empty(cls) -> LexicalDetectionResult:
        return cls(
            num_positions_scored=0,
            num_green_hits=0,
            green_fraction=0.0,
            lexical_z_score=0.0,
            lexical_p_value=1.0,
        )

    def to_joint_equivalent(self, threshold: float) -> JointDetectionResult:
        return JointDetectionResult(
            semantic_z=0.0,
            lexical_z=self.lexical_z_score,
            joint_score=self.lexical_z_score,
            p_joint=self.lexical_p_value,
            prediction=self.lexical_z_score >= threshold,
            confidence=1.0 - self.lexical_p_value,
            rationale="lexical-only evidence",
        )


def distribution_parameters(
    scores: list[BlockScore],
    gamma: float,
    mode: Literal["fixed", "adaptive"],
) -> tuple[float, float]:
    """Compute null-distribution mean and variance for block hits."""
    if mode == "adaptive":
        expected_hits = sum(score.gamma_effective for score in scores)
        variance = sum(
            score.gamma_effective * (1 - score.gamma_effective)
            for score in scores
        )
        return expected_hits, variance

    m = len(scores)
    expected_hits = m * gamma
    variance = m * gamma * (1 - gamma)
    return expected_hits, variance


def compute_z_score(
    observed_hits: int,
    expected_hits: float,
    variance: float,
) -> float:
    """Compute a Z score from observed hits and null-distribution statistics."""
    if variance <= 0.0:
        if observed_hits > expected_hits:
            return math.inf
        if observed_hits < expected_hits:
            return -math.inf
        return 0.0

    return (observed_hits - expected_hits) / math.sqrt(variance)


def semantic_detection_from_result(result: DetectionResult) -> SemanticDetectionResult:
    return SemanticDetectionResult(
        is_watermarked=result.is_watermarked,
        z_score=result.z_score,
        p_value=result.p_value,
        total_blocks=result.total_blocks,
        independent_blocks=result.independent_blocks,
        hit_blocks=result.hit_blocks,
        expected_hits=result.expected_hits,
        variance=result.variance,
        hypothesis_mode=result.hypothesis_mode,
    )


def fuse_joint_detection(
    semantic_z_score: float,
    lexical_result: LexicalDetectionResult,
    config: TokenChannelConfig,
) -> JointDetectionResult:
    support_factor = min(
        1.0,
        lexical_result.num_positions_scored / config.lexical_full_weight_min_positions,
    )
    joint_score = (
        config.joint_semantic_weight * semantic_z_score
        + config.joint_lexical_weight * support_factor * lexical_result.lexical_z_score
    )
    p_joint = float(norm.sf(joint_score))
    return JointDetectionResult(
        semantic_z=semantic_z_score,
        lexical_z=lexical_result.lexical_z_score,
        joint_score=joint_score,
        p_joint=p_joint,
        prediction=joint_score >= config.joint_threshold,
        confidence=1.0 - p_joint,
        rationale=_describe_joint_result(semantic_z_score, lexical_result.lexical_z_score),
    )


def _describe_joint_result(semantic_z: float, lexical_z: float) -> str:
    return f"{_semantic_label(semantic_z)}, {_lexical_label(lexical_z)}"


def _semantic_label(z_score: float) -> str:
    if z_score >= 4.0:
        return "semantic strong"
    if z_score >= 2.0:
        return "semantic borderline"
    return "semantic weak"


def _lexical_label(z_score: float) -> str:
    if z_score >= 4.0:
        return "lexical strong"
    if z_score >= 1.0:
        return "lexical supportive"
    return "lexical unsupported"


class HypothesisTester:
    """One-sided Z-test for watermark presence."""

    def __init__(
        self,
        fpr_threshold: float = 3.0,
        gamma: float = 0.5,
        mode: Literal["fixed", "adaptive"] = "fixed",
    ):
        self._fpr_threshold = fpr_threshold
        self._gamma = gamma
        self._mode = mode

    def test(
        self,
        selected_scores: list[BlockScore],
        total_blocks: int,
    ) -> DetectionResult:
        """Run hypothesis test on independent block scores.

        Args:
            selected_scores: Scores of DP-selected independent blocks.
            total_blocks: Total number of statement blocks in the code.

        Returns:
            DetectionResult with Z-score, p-value, and verdict.
        """
        m = len(selected_scores)
        expected_hits, variance = self._distribution_parameters(selected_scores)
        if m == 0:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                total_blocks=total_blocks,
                independent_blocks=0,
                hit_blocks=0,
                expected_hits=expected_hits,
                variance=variance,
                hypothesis_mode=self._mode,
                block_details=list(selected_scores),
            )

        x = sum(1 for s in selected_scores if s.score == 1)
        z_score = self._compute_z_score(x, expected_hits, variance)
        p_value = float(norm.sf(z_score))

        return DetectionResult(
            is_watermarked=z_score >= self._fpr_threshold,
            z_score=z_score,
            p_value=p_value,
            total_blocks=total_blocks,
            independent_blocks=m,
            hit_blocks=x,
            expected_hits=expected_hits,
            variance=variance,
            hypothesis_mode=self._mode,
            block_details=list(selected_scores),
        )

    def _distribution_parameters(
        self,
        selected_scores: list[BlockScore],
    ) -> tuple[float, float]:
        return distribution_parameters(
            selected_scores,
            gamma=self._gamma,
            mode=self._mode,
        )

    @staticmethod
    def _compute_z_score(
        observed_hits: int,
        expected_hits: float,
        variance: float,
    ) -> float:
        return compute_z_score(observed_hits, expected_hits, variance)
