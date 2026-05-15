"""WFCLLM ablation framework (spec §5.3).

Minimum-viable Cartesian-product sweep + JSONL metrics aggregation.
No distributed scheduling, no HPO, no auto-suggestion.
"""
from wfcllm.ablation.sweep import ResolvedConfig, SweepSpec, short_hash

__all__ = ["ResolvedConfig", "SweepSpec", "short_hash"]
