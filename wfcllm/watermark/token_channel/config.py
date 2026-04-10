"""Configuration for the lexical token-channel watermark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TokenChannelMode = Literal["semantic-only", "lexical-only", "dual-channel"]


@dataclass
class TokenChannelJointConfig:
    """Joint semantic and lexical detection weights."""

    semantic_weight: float = 1.0
    lexical_weight: float = 0.75
    lexical_full_weight_min_positions: int = 32
    threshold: float = 4.0


@dataclass
class TokenChannelConfig:
    """Shared token-channel configuration surface."""

    enabled: bool = False
    mode: TokenChannelMode = "dual-channel"
    model_path: str = "data/models/token-channel"
    context_width: int = 128
    switch_threshold: float = 0.0
    delta: float = 2.0
    ignore_repeated_ngrams: bool = False
    ignore_repeated_prefixes: bool = False
    debug_mode: bool = False

    # Protocol-controlled lexical safeguards.
    lexical_min_block_tokens: int = 8
    lexical_retry_decay_start: int = 2
    lexical_retry_disable_after: int = 4
    lexical_gate_probe_tokens: int = 16
    lexical_gate_min_fraction: float = 0.10

    joint: TokenChannelJointConfig = field(default_factory=TokenChannelJointConfig)

    @property
    def joint_semantic_weight(self) -> float:
        return self.joint.semantic_weight

    @property
    def joint_lexical_weight(self) -> float:
        return self.joint.lexical_weight

    @property
    def lexical_full_weight_min_positions(self) -> int:
        return self.joint.lexical_full_weight_min_positions

    @property
    def joint_threshold(self) -> float:
        return self.joint.threshold
