"""Metric extractors for the ablation framework (spec §5.3)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

MetricExtractor = Callable[[dict[str, Path]], Any]


class _MetricRegistry:
    """Lightweight name → extractor registry. Mirrors common.registry.Registry but
    holds bare callables instead of classes (extractors are stateless functions)."""

    def __init__(self) -> None:
        self._entries: dict[str, MetricExtractor] = {}

    def register(self, name: str, fn: MetricExtractor) -> None:
        if name in self._entries:
            raise ValueError(f"metric '{name}' already registered")
        self._entries[name] = fn

    def get(self, name: str) -> MetricExtractor:
        if name not in self._entries:
            raise KeyError(f"unknown metric: '{name}'. registered: {self.names()}")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)


METRIC_REGISTRY = _MetricRegistry()


def register_metric(name: str):
    def deco(fn: MetricExtractor) -> MetricExtractor:
        METRIC_REGISTRY.register(name, fn)
        return fn
    return deco


def get_metric(name: str) -> MetricExtractor:
    return METRIC_REGISTRY.get(name)


def _load_json_or_none(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- spec §5.3 metrics ---------------------------------------------------------


@register_metric("z_score_mean")
def _z_score_mean(artefacts: dict[str, Path]):
    summary = _load_json_or_none(artefacts.get("extract_summary"))
    if summary is None:
        return None
    return summary.get("summary", {}).get("mean_z_score")


@register_metric("hit_rate")
def _hit_rate(artefacts: dict[str, Path]):
    """Watermark detection rate — alias for the extract pipeline's `watermark_rate`."""
    summary = _load_json_or_none(artefacts.get("extract_summary"))
    if summary is None:
        return None
    return summary.get("summary", {}).get("watermark_rate")


@register_metric("false_positive_rate")
def _false_positive_rate(artefacts: dict[str, Path]):
    """FPR is reported by the calibration phase (`calibrate-threshold`), not
    by the extract pipeline. We look for it in the calibration artefact;
    if not present, return None — the runner records None in the JSONL row."""
    cal = _load_json_or_none(artefacts.get("calibration"))
    if cal is None:
        return None
    return cal.get("estimated_fpr") or cal.get("fpr")


@register_metric("pass_at_1")
def _pass_at_1(artefacts: dict[str, Path]):
    """Read `modes.<first>.generation_metrics.pass_at_1` from evaluation_summary.json.

    Spec §5.3 lists pass@1 as a baseline metric. The `wfcllm/evaluation/dual_channel.py`
    pipeline groups by mode; we pick the first mode deterministically (sorted)."""
    summary = _load_json_or_none(artefacts.get("evaluation_summary"))
    if summary is None:
        return None
    modes = summary.get("modes") or {}
    if not modes:
        return None
    first_mode = sorted(modes)[0]
    return modes[first_mode].get("generation_metrics", {}).get("pass_at_1")
