"""Configuration for the pretrain phase."""
from __future__ import annotations

from dataclasses import dataclass, field

CANONICAL_STAGE_ORDER = ("encoder", "lexical")


@dataclass
class PretrainConfig:
    """Pretrain phase configuration: which stages to run.

    Stages are always executed in CANONICAL_STAGE_ORDER regardless of input order,
    because lexical training depends on a trained encoder checkpoint.
    """

    stages: list[str] = field(default_factory=lambda: list(CANONICAL_STAGE_ORDER))

    def __post_init__(self) -> None:
        unknown = [s for s in self.stages if s not in CANONICAL_STAGE_ORDER]
        if unknown:
            raise ValueError(
                f"unknown stage(s): {unknown!r}; allowed: {list(CANONICAL_STAGE_ORDER)}"
            )
        self.stages = [s for s in CANONICAL_STAGE_ORDER if s in self.stages]
