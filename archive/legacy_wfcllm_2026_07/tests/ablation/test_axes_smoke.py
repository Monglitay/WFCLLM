"""Wire-up test: each spec §5.3 axis group must reach a real key on the base config.

This is *not* a phase-execution test — it just deep-copies base_config.json,
applies each axis group's overrides via SweepSpec.expand(), and asserts no
KeyError / no untouched leaves. Catches schema drift early.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.ablation.sweep import SweepSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base_config.json"


def _make_sweep_yaml(tmp_path: Path, name: str, axes_block: str, phases: str = "[watermark]") -> Path:
    p = tmp_path / "sweep.yaml"
    p.write_text(
        f"""
name: {name}
base_config: {BASE_CONFIG}
phases: {phases}
axes:
{axes_block}
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
        encoding="utf-8",
    )
    return p


def _expand_and_assert(yaml_path: Path, expected_paths: list[str]):
    spec = SweepSpec.from_yaml(yaml_path)
    combos = list(spec.expand())
    assert combos, "expansion produced no combinations"
    for combo in combos:
        for path in expected_paths:
            cursor = combo.config
            for part in path.split("."):
                assert isinstance(cursor, dict), f"non-dict on path '{path}' at '{part}'"
                assert part in cursor, f"path '{path}' missing key '{part}' in resolved config"
                cursor = cursor[part]


def test_axis_group_dual_channel_plus_threshold(tmp_path):
    yaml_path = _make_sweep_yaml(
        tmp_path,
        "g1",
        """  watermark.token_channel.mode: [semantic-only, dual-channel]
  watermark.token_channel.switch_threshold: [0.5, 0.7]
  extract.fpr_threshold: [2.5, 3.0]
  extract.adaptive_detection.mode: [fixed, prefer-adaptive]
""",
    )
    _expand_and_assert(
        yaml_path,
        [
            "watermark.token_channel.mode",
            "watermark.token_channel.switch_threshold",
            "extract.fpr_threshold",
            "extract.adaptive_detection.mode",
        ],
    )


def test_axis_group_margin_plus_lsh(tmp_path):
    yaml_path = _make_sweep_yaml(
        tmp_path,
        "g2",
        """  watermark.margin_base: [0.001, 0.002]
  watermark.margin_alpha: [0.001, 0.003]
  watermark.adaptive_gamma.enabled: [true, false]
  watermark.adaptive_gamma.strategy: [piecewise_quantile]
  watermark.lsh_d: [3, 4]
  watermark.lsh_gamma: [0.4, 0.5]
""",
    )
    _expand_and_assert(
        yaml_path,
        [
            "watermark.margin_base",
            "watermark.margin_alpha",
            "watermark.adaptive_gamma.enabled",
            "watermark.adaptive_gamma.strategy",
            "watermark.lsh_d",
            "watermark.lsh_gamma",
        ],
    )


def test_axis_group_llm_plus_encoder(tmp_path):
    yaml_path = _make_sweep_yaml(
        tmp_path,
        "g3",
        """  watermark.lm_model_path: ["data/models/deepseek-coder-7b-base"]
  encoder.model_name: ["data/models/codet5-base"]
  encoder.use_lora: [true, false]
  encoder.embed_dim: [128, 256]
""",
    )
    _expand_and_assert(
        yaml_path,
        [
            "watermark.lm_model_path",
            "encoder.model_name",
            "encoder.use_lora",
            "encoder.embed_dim",
        ],
    )


def test_example_yaml_loads_cleanly():
    example = REPO_ROOT / "configs" / "ablation" / "example_dual_channel.yaml"
    assert example.exists(), f"missing example sweep: {example}"
    spec = SweepSpec.from_yaml(example)
    assert spec.name == "dual_channel_vs_threshold"
    assert "z_score_mean" in spec.metrics
    combos = list(spec.expand())
    # 3 modes × 4 fpr thresholds = 12
    assert len(combos) == 12
