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


def _write_yaml(tmp_path: Path, body: str) -> Path:
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text(body, encoding="utf-8")
    return yaml_path


def _write_base_config(tmp_path: Path) -> Path:
    base_path = tmp_path / "base.json"
    base_path.write_text(
        '{"watermark": {"token_channel": {"mode": "semantic-only"}}, '
        '"extract": {"fpr_threshold": 3.0}}',
        encoding="utf-8",
    )
    return base_path


def test_from_yaml_loads_all_fields(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
name: test_sweep
base_config: {base.name}
phases: [watermark, extract]
axes:
  watermark.token_channel.mode: [semantic-only, dual-channel]
  extract.fpr_threshold: [2.5, 3.0]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    spec = SweepSpec.from_yaml(yaml_path)
    assert spec.name == "test_sweep"
    assert spec.phases == ("watermark", "extract")
    assert spec.metrics == ("z_score_mean",)
    assert spec.base_config == (tmp_path / "base.json").resolve()
    assert spec.output_dir == (tmp_path / "out").resolve()
    assert dict(spec.axes) == {
        "watermark.token_channel.mode": ("semantic-only", "dual-channel"),
        "extract.fpr_threshold": (2.5, 3.0),
    }


def test_expand_yields_cartesian_product(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: [semantic-only, dual-channel]
  extract.fpr_threshold: [2.5, 3.0]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    spec = SweepSpec.from_yaml(yaml_path)
    combos = list(spec.expand())
    assert len(combos) == 4
    seen = {(c.axes_values["watermark.token_channel.mode"], c.axes_values["extract.fpr_threshold"]) for c in combos}
    assert seen == {
        ("semantic-only", 2.5), ("semantic-only", 3.0),
        ("dual-channel", 2.5),  ("dual-channel", 3.0),
    }


def test_expand_applies_dotted_path_overrides(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: [dual-channel]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    spec = SweepSpec.from_yaml(yaml_path)
    [combo] = list(spec.expand())
    assert combo.config["watermark"]["token_channel"]["mode"] == "dual-channel"
    # Untouched key preserved:
    assert combo.config["extract"]["fpr_threshold"] == 3.0


def test_expand_does_not_mutate_base_config_dict(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: [dual-channel, lexical-only]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    spec = SweepSpec.from_yaml(yaml_path)
    combos = list(spec.expand())
    # The two combos must NOT alias each other's nested dict:
    combos[0].config["watermark"]["token_channel"]["mode"] = "MUTATED"
    assert combos[1].config["watermark"]["token_channel"]["mode"] == "lexical-only"


def test_run_id_is_zero_padded_index_plus_short_hash(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: [dual-channel, lexical-only]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    spec = SweepSpec.from_yaml(yaml_path)
    combos = list(spec.expand())
    assert combos[0].run_id.startswith("0000_")
    assert combos[1].run_id.startswith("0001_")
    assert len(combos[0].run_id) == 5 + 8  # "NNNN_" + 8-char hash


def test_from_yaml_rejects_missing_name(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: [dual-channel]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    with pytest.raises(ValueError, match="name"):
        SweepSpec.from_yaml(yaml_path)


def test_from_yaml_rejects_non_list_axis_values(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: dual-channel
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    with pytest.raises(ValueError, match="must be a non-empty list"):
        SweepSpec.from_yaml(yaml_path)


def test_from_yaml_rejects_empty_axes(tmp_path):
    base = _write_base_config(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes: {{}}
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
    )
    with pytest.raises(ValueError, match="axes"):
        SweepSpec.from_yaml(yaml_path)
