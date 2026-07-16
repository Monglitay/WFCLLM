"""Frozen configuration contracts for gate data, training, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateDataConfig:
    """First-version data collection constants.

    Changing any of these values requires a new data/label contract rather
    than silently mutating the v1 dataset identity.
    """

    training_key_count: int = 32
    holdout_key_count: int = 8
    rewrite_count: int = 6
    rewrite_budgets: tuple[int, ...] = (1, 3, 6)

    def __post_init__(self) -> None:
        expected = {
            "training_key_count": 32,
            "holdout_key_count": 8,
            "rewrite_count": 6,
            "rewrite_budgets": (1, 3, 6),
        }
        for name, required in expected.items():
            actual = getattr(self, name)
            if type(actual) is not type(required) or actual != required:
                raise ValueError(f"{name} must remain fixed at {required!r}")


@dataclass(frozen=True)
class GateTrainConfig:
    """Portable gate-training inputs that never trigger model downloads."""

    max_tokens: int = 512
    base_model_path: Path = Path("data/models/codet5-base")

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens != 512:
            raise ValueError("max_tokens must remain fixed at 512")
        if not isinstance(self.base_model_path, Path):
            raise ValueError("base_model_path must be a pathlib.Path")
        path_text = str(self.base_model_path)
        first_component = path_text.split("/", 1)[0]
        if path_text in {"", "."} or ":" in first_component:
            raise ValueError("base_model_path must identify a local model path")


@dataclass(frozen=True)
class GateValidateConfig:
    """Hard validation thresholds for publishing a formal gate bundle."""

    decision_agreement_min: float = 0.999
    float_quantized_accepted_set_agreement_min: float = 0.999
    formal_accepted_span_consensus_min: float = 1.0
    suitable_false_positive_rate_max: float = 0.05
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self) -> None:
        expected = {
            "decision_agreement_min": 0.999,
            "float_quantized_accepted_set_agreement_min": 0.999,
            "formal_accepted_span_consensus_min": 1.0,
            "suitable_false_positive_rate_max": 0.05,
            "batch_sizes": (1, 2, 4, 8),
        }
        for name, required in expected.items():
            actual = getattr(self, name)
            if type(actual) is not type(required) or actual != required:
                raise ValueError(f"{name} must remain fixed at {required!r}")
