"""WFCLLM ablation framework (spec §5.3)."""
from wfcllm.ablation.metrics import (
    METRIC_REGISTRY,
    MetricExtractor,
    get_metric,
    register_metric,
)
from wfcllm.ablation.sweep import ResolvedConfig, SweepSpec, short_hash

__all__ = [
    "METRIC_REGISTRY",
    "MetricExtractor",
    "ResolvedConfig",
    "SweepSpec",
    "get_metric",
    "register_metric",
    "short_hash",
]
