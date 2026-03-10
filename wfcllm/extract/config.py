"""Configuration and data structures for watermark extraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractConfig:
    """Configuration for the watermark extraction pipeline."""

    secret_key: str
    embed_dim: int = 128
    z_threshold: float = 3.0


@dataclass
class BlockScore:
    """Score result for a single statement block."""

    block_id: str
    score: int        # +1 (hit) or -1 (miss)
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
