"""Tests for SweepSpec parsing and expansion (spec §5.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from wfcllm.ablation.sweep import ResolvedConfig, SweepSpec, short_hash


def test_short_hash_is_deterministic_and_8_chars():
    a = short_hash({"watermark.token_channel.mode": "dual-channel", "extract.fpr_threshold": 3.0})
    b = short_hash({"extract.fpr_threshold": 3.0, "watermark.token_channel.mode": "dual-channel"})
    assert a == b
    assert len(a) == 8


def test_short_hash_differs_for_different_axes_values():
    a = short_hash({"x": 1})
    b = short_hash({"x": 2})
    assert a != b


def test_resolved_config_short_hash_matches_module_function():
    rc = ResolvedConfig(
        axes_values={"x": 1, "y": "z"},
        config={"x": 1, "y": "z"},
        run_id="0000_dummy",
    )
    assert rc.short_hash() == short_hash({"x": 1, "y": "z"})
