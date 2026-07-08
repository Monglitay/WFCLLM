from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from wfcllm.datasets.constants import SUPPORTED_DATASETS


_ALLOWED_TORCH_DTYPES = ("auto", "fp32", "fp16", "bf16")
_ALLOWED_PROMPT_MODES = ("completion", "chat")
_ALLOWED_RULE_NAMES = ("hash", "semantic_lsh")
DEFAULT_HUMANEVAL_STOP_SEQUENCES = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


@dataclass(frozen=True)
class SawrGenerationConfig:
    """Local causal-LM generation settings for the SAWR smoke runner."""

    model_path: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    retry_repetition_penalty: float = 1.0
    torch_dtype: str = "auto"
    device: str = "cuda"
    seed: int = 0
    load_in_4bit: bool = False
    eos_token_id: int | None = None
    prompt_mode: str = "completion"
    stop_sequences: tuple[str, ...] = DEFAULT_HUMANEVAL_STOP_SEQUENCES

    def __post_init__(self) -> None:
        if not Path(self.model_path).exists():
            raise ValueError(f"model_path does not exist: {self.model_path}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if (
            isinstance(self.retry_repetition_penalty, bool)
            or not isinstance(self.retry_repetition_penalty, (int, float))
            or not math.isfinite(float(self.retry_repetition_penalty))
        ):
            raise ValueError("retry_repetition_penalty must be a finite number")
        if self.retry_repetition_penalty < 1.0:
            raise ValueError("retry_repetition_penalty must be >= 1.0")
        if self.torch_dtype not in _ALLOWED_TORCH_DTYPES:
            raise ValueError(
                f"torch_dtype must be one of {_ALLOWED_TORCH_DTYPES}, got {self.torch_dtype!r}"
            )
        object.__setattr__(
            self,
            "retry_repetition_penalty",
            float(self.retry_repetition_penalty),
        )
        if self.prompt_mode not in _ALLOWED_PROMPT_MODES:
            raise ValueError(
                f"prompt_mode must be one of {_ALLOWED_PROMPT_MODES}, got {self.prompt_mode!r}"
            )
        if isinstance(self.stop_sequences, str) or not isinstance(
            self.stop_sequences,
            (tuple, list),
        ):
            raise ValueError("stop_sequences must be a sequence of strings")
        if any(
            not isinstance(stop_sequence, str) or not stop_sequence
            for stop_sequence in self.stop_sequences
        ):
            raise ValueError("stop_sequences entries must be non-empty strings")
        object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))


@dataclass(frozen=True)
class SawrRuleConfig:
    """Embedding rule settings for deterministic SAWR smoke decisions."""

    rule_name: str = "hash"
    target_accept_rate: float = 0.5
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rule_name not in _ALLOWED_RULE_NAMES:
            raise ValueError(
                f"rule_name must be one of {_ALLOWED_RULE_NAMES}, got {self.rule_name!r}"
            )
        if not 0 <= self.target_accept_rate <= 1:
            raise ValueError("target_accept_rate must be in [0, 1]")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a dict")
        try:
            parameters_json = json.dumps(self.parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must be JSON-serializable") from exc
        object.__setattr__(self, "parameters", json.loads(parameters_json))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SawrPipelineConfig:
    """Top-level SAWR smoke pipeline settings."""

    dataset: str
    dataset_path: str
    output_dir: str
    generation: SawrGenerationConfig
    rule: SawrRuleConfig = field(default_factory=SawrRuleConfig)
    sample_limit: int | None = None
    sample_offset: int | None = None
    max_group_statements: int = 2
    retry_budget: int = 1
    statement_retry_budget: int | None = None
    window_retry_budget: int | None = None
    compound_retry_budget: int | None = None
    global_rollback_budget: int | None = None
    max_total_sampled_tokens: int | None = None
    evidence_retry_attempts: int = 1
    evidence_retry_seed_stride: int = 1009
    resume: str | None = None
    candidate_sidecar_output: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generation, SawrGenerationConfig):
            raise ValueError("generation must be SawrGenerationConfig")
        if not isinstance(self.rule, SawrRuleConfig):
            raise ValueError("rule must be SawrRuleConfig")
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{self.dataset}'"
            )
        if self.sample_limit is not None and self.sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")
        if self.sample_offset is not None and self.sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")
        if self.max_group_statements <= 0:
            raise ValueError("max_group_statements must be positive")
        if self.retry_budget < 0:
            raise ValueError("retry_budget must be non-negative")
        if self.statement_retry_budget is not None and self.statement_retry_budget < 0:
            raise ValueError("statement_retry_budget must be non-negative")
        if self.window_retry_budget is not None and self.window_retry_budget < 0:
            raise ValueError("window_retry_budget must be non-negative")
        if self.compound_retry_budget is not None and self.compound_retry_budget < 0:
            raise ValueError("compound_retry_budget must be non-negative")
        retry_budget_for_limits = sum(
            (
                0
                if self.statement_retry_budget is None
                else self.statement_retry_budget,
                0 if self.window_retry_budget is None else self.window_retry_budget,
                (
                    self.retry_budget
                    if self.compound_retry_budget is None
                    else self.compound_retry_budget
                ),
            )
        )
        if self.global_rollback_budget is None:
            object.__setattr__(self, "global_rollback_budget", retry_budget_for_limits)
        elif self.global_rollback_budget < 0:
            raise ValueError("global_rollback_budget must be non-negative")
        if self.max_total_sampled_tokens is None:
            derived_budget = self.generation.max_new_tokens * max(
                2,
                int(self.global_rollback_budget) + 2,
            )
            object.__setattr__(self, "max_total_sampled_tokens", derived_budget)
        elif self.max_total_sampled_tokens <= 0:
            raise ValueError("max_total_sampled_tokens must be positive")
        if self.evidence_retry_attempts <= 0:
            raise ValueError("evidence_retry_attempts must be positive")
        if self.evidence_retry_seed_stride <= 0:
            raise ValueError("evidence_retry_seed_stride must be positive")
        if self.resume is not None and self.resume != "latest":
            raise ValueError("resume must be None or 'latest'")
        if self.candidate_sidecar_output is not None and not isinstance(
            self.candidate_sidecar_output,
            str,
        ):
            raise ValueError("candidate_sidecar_output must be a string or None")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
