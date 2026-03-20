"""Configuration and data structures for watermark extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from wfcllm.extract.alignment import AlignmentReport


@dataclass
class AdaptiveDetectionConfig:
    """Adaptive-mode detection policy controls."""

    mode: Literal["fixed", "prefer-adaptive", "require-adaptive"] = "fixed"
    require_block_contract_check: bool = True
    fail_on_structure_mismatch: bool = True
    warn_on_numeric_mismatch: bool = True
    exclude_invalid_samples: bool = True

    @property
    def prefer_adaptive(self) -> bool:
        return self.mode != "fixed"

    @property
    def require_adaptive(self) -> bool:
        return self.mode == "require-adaptive"


@dataclass
class ExtractConfig:
    """Configuration for the watermark extraction pipeline."""

    secret_key: str
    embed_dim: int = 128
    fpr_threshold: float = 3.0  # M_r，由校准脚本生成；默认值 3.0 仅作占位
    lsh_d: int = 3
    lsh_gamma: float = 0.5
    adaptive_detection: AdaptiveDetectionConfig = field(default_factory=AdaptiveDetectionConfig)


@dataclass
class BlockScore:
    """Score result for a single statement block."""

    block_id: str
    score: int        # 1 (hit) or 0 (miss)
    min_margin: float
    gamma_effective: float = 0.5
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
    expected_hits: float = 0.0
    variance: float = 0.0
    hypothesis_mode: Literal["fixed", "adaptive"] = "fixed"
    block_details: list[BlockScore] = field(default_factory=list)
    alignment_report: AlignmentReport | None = None
    contract_valid: bool | None = None

    @property
    def mode(self) -> Literal["fixed", "adaptive"]:
        return self.hypothesis_mode

    @property
    def alignment_ok(self) -> bool | None:
        if self.alignment_report is None:
            return None
        return self.alignment_report.is_aligned
