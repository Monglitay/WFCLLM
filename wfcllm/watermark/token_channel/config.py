"""Configuration for the lexical token-channel watermark."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from typing import Literal


TokenChannelMode = Literal["semantic-only", "lexical-only", "dual-channel"]


@dataclass
class TokenChannelJointConfig:
    """Joint semantic and lexical detection weights."""

    semantic_weight: float = 1.0
    lexical_weight: float = 0.75
    lexical_full_weight_min_positions: int = 32
    threshold: float = 4.0

    @classmethod
    def from_mapping(cls, section: dict[str, Any] | None) -> TokenChannelJointConfig:
        configured = section or {}
        defaults = cls()
        return cls(
            semantic_weight=_coerce_float(
                _coalesce(configured.get("semantic_weight"), defaults.semantic_weight),
                "semantic_weight",
            ),
            lexical_weight=_coerce_float(
                _coalesce(configured.get("lexical_weight"), defaults.lexical_weight),
                "lexical_weight",
            ),
            lexical_full_weight_min_positions=_coerce_int(
                _coalesce(
                    configured.get("lexical_full_weight_min_positions"),
                    defaults.lexical_full_weight_min_positions,
                ),
                "lexical_full_weight_min_positions",
            ),
            threshold=_coerce_float(
                _coalesce(configured.get("threshold"), defaults.threshold),
                "threshold",
            ),
        )

    def __post_init__(self) -> None:
        if self.semantic_weight < 0:
            raise ValueError("semantic_weight must be >= 0")
        if self.lexical_weight < 0:
            raise ValueError("lexical_weight must be >= 0")
        if self.lexical_full_weight_min_positions <= 0:
            raise ValueError("lexical_full_weight_min_positions must be > 0")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")


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

    @classmethod
    def from_mapping(cls, section: dict[str, Any] | None) -> TokenChannelConfig:
        configured = section or {}
        defaults = cls()

        raw_mode = _coalesce(
            configured.get("channel_mode"),
            configured.get("mode"),
            defaults.mode,
        )
        if raw_mode not in {"semantic-only", "lexical-only", "dual-channel"}:
            raise ValueError(
                "channel_mode must be one of ['dual-channel', 'lexical-only', 'semantic-only']"
            )

        raw_joint_section = configured.get("joint")
        if raw_joint_section is None:
            joint_section = {}
        elif isinstance(raw_joint_section, dict):
            joint_section = raw_joint_section
        else:
            raise ValueError("joint must be a JSON object")
        joint = TokenChannelJointConfig.from_mapping(
            {
                **joint_section,
                "semantic_weight": configured.get(
                    "joint_semantic_weight",
                    joint_section.get("semantic_weight"),
                ),
                "lexical_weight": configured.get(
                    "joint_lexical_weight",
                    joint_section.get("lexical_weight"),
                ),
                "lexical_full_weight_min_positions": configured.get(
                    "lexical_full_weight_min_positions",
                    joint_section.get("lexical_full_weight_min_positions"),
                ),
                "threshold": configured.get(
                    "joint_threshold",
                    joint_section.get("threshold"),
                ),
            }
        )

        return cls(
            enabled=_coerce_bool(configured.get("enabled", defaults.enabled), "enabled"),
            mode=raw_mode,
            model_path=_coerce_optional_string(
                configured.get("model_path", defaults.model_path),
                "model_path",
            ),
            context_width=_coerce_int(
                _coalesce(configured.get("context_width"), defaults.context_width),
                "context_width",
            ),
            switch_threshold=_coerce_float(
                _coalesce(configured.get("switch_threshold"), defaults.switch_threshold),
                "switch_threshold",
            ),
            delta=_coerce_float(_coalesce(configured.get("delta"), defaults.delta), "delta"),
            ignore_repeated_ngrams=_coerce_bool(
                configured.get("ignore_repeated_ngrams", defaults.ignore_repeated_ngrams),
                "ignore_repeated_ngrams",
            ),
            ignore_repeated_prefixes=_coerce_bool(
                configured.get(
                    "ignore_repeated_prefixes",
                    defaults.ignore_repeated_prefixes,
                ),
                "ignore_repeated_prefixes",
            ),
            debug_mode=_coerce_bool(configured.get("debug_mode", defaults.debug_mode), "debug_mode"),
            lexical_min_block_tokens=_coerce_int(
                _coalesce(
                    configured.get("lexical_min_block_tokens"),
                    defaults.lexical_min_block_tokens,
                ),
                "lexical_min_block_tokens",
            ),
            lexical_retry_decay_start=_coerce_int(
                _coalesce(
                    configured.get("lexical_retry_decay_start"),
                    defaults.lexical_retry_decay_start,
                ),
                "lexical_retry_decay_start",
            ),
            lexical_retry_disable_after=_coerce_int(
                _coalesce(
                    configured.get("lexical_retry_disable_after"),
                    defaults.lexical_retry_disable_after,
                ),
                "lexical_retry_disable_after",
            ),
            lexical_gate_probe_tokens=_coerce_int(
                _coalesce(
                    configured.get("lexical_gate_probe_tokens"),
                    defaults.lexical_gate_probe_tokens,
                ),
                "lexical_gate_probe_tokens",
            ),
            lexical_gate_min_fraction=_coerce_float(
                _coalesce(
                    configured.get("lexical_gate_min_fraction"),
                    defaults.lexical_gate_min_fraction,
                ),
                "lexical_gate_min_fraction",
            ),
            joint=joint,
        )

    def __post_init__(self) -> None:
        valid_modes = {"semantic-only", "lexical-only", "dual-channel"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}")
        if self.context_width <= 0:
            raise ValueError("context_width must be > 0")
        if self.switch_threshold < 0:
            raise ValueError("switch_threshold must be >= 0")
        if self.delta < 0:
            raise ValueError("delta must be >= 0")
        if self.lexical_min_block_tokens <= 0:
            raise ValueError("lexical_min_block_tokens must be > 0")
        if self.lexical_retry_decay_start < 0:
            raise ValueError("lexical_retry_decay_start must be >= 0")
        if self.lexical_retry_disable_after < self.lexical_retry_decay_start:
            raise ValueError(
                "lexical_retry_disable_after must be >= lexical_retry_decay_start"
            )
        if self.lexical_gate_probe_tokens <= 0:
            raise ValueError("lexical_gate_probe_tokens must be > 0")
        if not 0.0 <= self.lexical_gate_min_fraction <= 1.0:
            raise ValueError("lexical_gate_min_fraction must be between 0 and 1")

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


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} must be an integer")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and any(char in stripped for char in ".eE"):
            raise ValueError(f"{field_name} must be an integer")
    return coerced


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"{field_name} must be finite")
    return coerced


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _coerce_optional_string(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        raise ValueError(f"{field_name} must be a string")
    raise ValueError(f"{field_name} must be a string")


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
