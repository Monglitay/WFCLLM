from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

from wfcllm.sawr.rules import _quantize_gamma


DETECTOR_MODE = "sawr-structure-aware-proxy-window/v1"
EVIDENCE_MODES = ("hit_plus_margin", "hit_only", "margin_only")
STATISTIC_MODES = (
    "calibrated_context_max",
    "raw_context_max",
    "context_mean_window_evidence",
)

EvidenceMode = Literal["hit_plus_margin", "hit_only", "margin_only"]
StatisticMode = Literal[
    "calibrated_context_max",
    "raw_context_max",
    "context_mean_window_evidence",
]


@dataclass(frozen=True)
class BucketEdges:
    window_count: tuple[int, ...] = (1, 2, 4, 8, 16)
    statement_count: tuple[int, ...] = (1, 2, 4, 8, 16)
    sample_window_count: tuple[int, ...] = (2, 6, 16, 32, 64)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "window_count",
            _validate_edges("window_count", self.window_count),
        )
        object.__setattr__(
            self,
            "statement_count",
            _validate_edges("statement_count", self.statement_count),
        )
        object.__setattr__(
            self,
            "sample_window_count",
            _validate_edges("sample_window_count", self.sample_window_count),
        )

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "window_count": list(self.window_count),
            "statement_count": list(self.statement_count),
            "sample_window_count": list(self.sample_window_count),
        }


@dataclass(frozen=True)
class SawrDetectionConfig:
    secret_key: str
    lsh_d: int = 4
    gamma: float = 0.75
    semantic_margin: float = 0.0
    max_group_statements: int = 2
    min_scoreable_contexts: int = 1
    min_proxy_windows: int = 2
    target_fpr: float = 0.05
    use_ordinal_keying: bool = False
    evidence_mode: EvidenceMode = "hit_plus_margin"
    statistic: StatisticMode = "calibrated_context_max"
    structure_aware: bool = True
    bucket_edges: BucketEdges = field(default_factory=BucketEdges)
    detector_mode: str = DETECTOR_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.secret_key, str) or not self.secret_key:
            raise ValueError("secret_key must be non-empty")
        if not _is_int(self.lsh_d) or self.lsh_d < 1:
            raise ValueError("lsh_d must be >= 1")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not self.semantic_margin >= 0:
            raise ValueError("semantic_margin must be non-negative")
        if not _is_int(self.max_group_statements) or self.max_group_statements <= 0:
            raise ValueError("max_group_statements must be positive")
        if not _is_int(self.min_scoreable_contexts) or self.min_scoreable_contexts <= 0:
            raise ValueError("min_scoreable_contexts must be positive")
        if not _is_int(self.min_proxy_windows) or self.min_proxy_windows <= 0:
            raise ValueError("min_proxy_windows must be positive")
        if not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be in (0, 1)")
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(
                f"evidence_mode must be one of {EVIDENCE_MODES}, got {self.evidence_mode!r}"
            )
        if self.statistic not in STATISTIC_MODES:
            raise ValueError(
                f"statistic must be one of {STATISTIC_MODES}, got {self.statistic!r}"
            )
        if not isinstance(self.bucket_edges, BucketEdges):
            raise ValueError("bucket_edges must be BucketEdges")

    @property
    def k(self) -> int:
        return _quantize_gamma(self.gamma, self.lsh_d).k

    @property
    def gamma_effective(self) -> float:
        return _quantize_gamma(self.gamma, self.lsh_d).gamma_effective

    def to_public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload.pop("secret_key")
        payload["secret_key_sha256"] = hashlib.sha256(
            self.secret_key.encode("utf-8")
        ).hexdigest()
        payload["bucket_edges"] = self.bucket_edges.to_dict()
        payload["k"] = self.k
        payload["gamma_effective"] = self.gamma_effective
        return payload


def bucket_label(value: int, edges: tuple[int, ...]) -> str:
    if value < 0:
        raise ValueError("value must be non-negative")
    edge_tuple = _validate_edges("edges", edges)
    for index, lower_bound in enumerate(edge_tuple):
        if value < lower_bound:
            return str(value)
        if index == len(edge_tuple) - 1:
            return f"{lower_bound}+"
        upper_bound = edge_tuple[index + 1] - 1
        if value <= upper_bound:
            if lower_bound == upper_bound:
                return str(lower_bound)
            return f"{lower_bound}-{upper_bound}"
    return f"{edge_tuple[-1]}+"


def _validate_edges(name: str, edges: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(edges, (str, bytes)):
        raise ValueError(f"{name} edges must be non-empty strictly increasing ints")
    try:
        edge_tuple = tuple(edges)
    except TypeError as exc:
        raise ValueError(
            f"{name} edges must be non-empty strictly increasing ints"
        ) from exc
    if not edge_tuple:
        raise ValueError(f"{name} edges must be non-empty strictly increasing ints")
    if any(not _is_int(edge) for edge in edge_tuple):
        raise ValueError(f"{name} edges must be non-empty strictly increasing ints")
    if any(left >= right for left, right in zip(edge_tuple, edge_tuple[1:])):
        raise ValueError(f"{name} edges must be non-empty strictly increasing ints")
    return edge_tuple


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
