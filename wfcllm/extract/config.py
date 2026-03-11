"""Configuration and data structures for watermark extraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractConfig:
    """Configuration for the watermark extraction pipeline."""

    secret_key: str
    embed_dim: int = 128
    fpr_threshold: float = 3.0  # M_r，由校准脚本生成；默认值 3.0 仅作占位
    lsh_d: int = 3
    lsh_gamma: float = 0.5


@dataclass
class BlockScore:
    """Score result for a single statement block."""

    block_id: str
    score: int        # 1 (hit) or 0 (miss)
    min_margin: float
    selected: bool = False


@dataclass
class DetectionResult:
    """Final watermark detection result."""

    is_watermarked: bool
    z_score: float
    p_value: float
    total_blocks: int
    independent_blocks: int
    hit_blocks: int
    block_details: list[BlockScore] = field(default_factory=list)
