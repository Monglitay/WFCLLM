"""Frozen configuration contracts for gate data and training."""

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
    rewrite_count: int = 3
    rewrite_budgets: tuple[int, ...] = (1, 3)

    def __post_init__(self) -> None:
        fixed = {
            "training_key_count": 32,
            "holdout_key_count": 8,
        }
        for name, required in fixed.items():
            actual = getattr(self, name)
            if type(actual) is not type(required) or actual != required:
                raise ValueError(f"{name} must remain fixed at {required!r}")
        if (
            type(self.rewrite_count) is not int
            or not 0 <= self.rewrite_count <= 48
        ):
            raise ValueError("rewrite_count must be an integer from 0 through 48")
        if (
            not isinstance(self.rewrite_budgets, tuple)
            or not self.rewrite_budgets
            or any(
                type(budget) is not int
                or not 0 <= budget <= self.rewrite_count
                for budget in self.rewrite_budgets
            )
            or len(set(self.rewrite_budgets)) != len(self.rewrite_budgets)
            or tuple(sorted(self.rewrite_budgets)) != self.rewrite_budgets
            or self.rewrite_budgets[-1] != self.rewrite_count
        ):
            raise ValueError(
                "rewrite_budgets must be ordered unique budgets ending at rewrite_count"
            )


@dataclass(frozen=True)
class GateTrainConfig:
    """Portable gate-training inputs that never trigger model downloads."""

    max_tokens: int = 256
    base_model_path: Path = Path("data/models/codet5-base")

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 512:
            raise ValueError("max_tokens must be between 1 and 512")
        if not isinstance(self.base_model_path, Path):
            raise ValueError("base_model_path must be a pathlib.Path")
        path_text = str(self.base_model_path)
        first_component = path_text.split("/", 1)[0]
        if path_text in {"", "."} or ":" in first_component:
            raise ValueError("base_model_path must identify a local model path")
