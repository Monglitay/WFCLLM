# Phase 7: `wfcllm/ablation/` Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a self-contained, single-process ablation framework — `wfcllm/ablation/sweep.py` + `runner.py` + `metrics.py` — that takes a YAML sweep spec, expands it into the Cartesian product of axes, dotted-path-overrides them onto a base config, runs the requested phases per combination through the existing `PhaseOrchestrator`, and aggregates a chosen subset of metrics into `data/ablation/<name>/results.jsonl` with per-run resume support. One example YAML (`configs/ablation/example_dual_channel.yaml`) and one CLI entry (`scripts/run_ablation.py`) are added; **no algorithm parameters, no new external deps, no changes to existing phase code.**

**Architecture:** Three thin layers built on top of the Phase 1 orchestration stack. (1) `SweepSpec` (frozen dataclass) loads YAML, validates the `axes` shape, and yields `ResolvedConfig` instances — each is a deep-copied base config with dotted-path overrides applied. (2) `AblationRunner` iterates `spec.expand()`, computes a deterministic `run_id = f"{i:04d}_{short_hash(axes_values)}"`, skips if `run_id` already in `results.jsonl` (resume), writes a per-run JSON config to `output_dir/runs/<run_id>/config.json`, dispatches phases by reusing `PhaseOrchestrator` via a synthetic `argparse.Namespace`, then calls `metrics.collect(...)` to pull a flat dict of requested metric names. (3) `metrics.py` is a registry of metric extractors that read the canonical artefact paths produced by `extract`/`watermark`/`evaluation` (e.g. `<output_dir>/<dataset>_details_summary.json`); only the four metric names listed in spec §5.3 are wired today.

**Tech Stack:** Python 3.10+, `PyYAML==6.0.3` (already a project dep), `dataclasses`, `hashlib`, `json`, `pathlib`, pytest. No new external deps.

**Spec reference:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §5.3 + §10 step 7 + §11 ("不做分布式 sweep 调度；不做超参建议 / Bayesian HPO；不引入新算法参数").

**Out of scope (explicitly NOT in this phase):**

- Distributed scheduling, multi-node fan-out, multiprocessing — single-process sequential only.
- Hyperparameter optimization, Bayesian search, adaptive sampling — pure Cartesian product only.
- Auto-suggesting parameters or "next best run" recommendations.
- Adding new tunable parameters to existing dataclass configs (`WatermarkConfig`, `ExtractConfig`, `TokenChannelConfig`, `EncoderConfig` etc.) — spec §11 explicit "本次重构不引入新参数". The framework only operates on knobs already declared today; the Task 11 wire-up test confirms the three axis groups in spec §5.3 are reachable through dotted paths on the **existing** schema.
- Modifying `wfcllm/cli/`, `wfcllm/orchestration/`, `wfcllm/watermark/`, `wfcllm/extract/`, `wfcllm/evaluation/`, or any first-party caller. The runner reuses `PhaseOrchestrator.dispatch_phase` exactly as the entropy-profile prereq does today (read-only consumer).
- Live-streaming progress display (TUI) — the runner prints one line per finished combination, that's it.
- `experiment/` and `tools/` directories — not touched per project rules.
- Modifying any test under `tests/encoder/`, `tests/watermark/`, `tests/extract/`, `tests/evaluation/`, or `tests/integration/` — Phase 7 only adds new tests under `tests/ablation/`.
- Re-tuning or fixing the 15 pre-existing baseline failures recorded in Phase 6 (`tests/extract/test_joint_detection.py` ×1, `tests/extract/test_pipeline.py::TestExtractPipelineStatistics` ×3, `tests/watermark/token_channel/test_runtime.py` ×1, `tests/watermark/token_channel/test_train_corpus.py` ×2, `tests/watermark/token_channel/test_train_workflow.py` ×8). These remain untouched and excluded from the "no new failures" criterion.

---

## File Structure

**New canonical files:**

| Path | Responsibility |
|---|---|
| `wfcllm/ablation/__init__.py` | Public exports: `SweepSpec`, `ResolvedConfig`, `AblationRunner`, `register_metric`, `get_metric`. |
| `wfcllm/ablation/sweep.py` | `SweepSpec` frozen dataclass + `ResolvedConfig` dataclass + YAML loader + `.expand()` Cartesian-product generator + `short_hash` helper. |
| `wfcllm/ablation/runner.py` | `AblationRunner` class — invokes phases via `PhaseOrchestrator.dispatch_phase`, persists per-run config, appends results JSONL line, handles resume. |
| `wfcllm/ablation/metrics.py` | `Registry[MetricExtractor]` (built on `wfcllm.common.registry.Registry`) + 4 spec-mandated extractors: `z_score_mean`, `hit_rate`, `false_positive_rate`, `pass_at_1`. |
| `configs/ablation/__init__.py` | Empty — present so the directory is git-tracked even if all examples were removed. |
| `configs/ablation/example_dual_channel.yaml` | Spec §5.3 example sweep verbatim. |
| `scripts/run_ablation.py` | CLI entry point — `python scripts/run_ablation.py <yaml> [--resume]`; thin shell over `AblationRunner`. |

**New tests:**

| Path | Coverage |
|---|---|
| `tests/ablation/__init__.py` | Empty marker. |
| `tests/ablation/test_sweep.py` | YAML round-trip; axes Cartesian-product expansion; dotted-path override (`watermark.token_channel.mode`); deep-copy isolation; deterministic `short_hash`; validation errors (missing `name`, non-list axis values). |
| `tests/ablation/test_runner.py` | End-to-end with mocked `PhaseOrchestrator.dispatch_phase`; resume by reading existing `results.jsonl`; per-run config persisted; one row appended per combination. |
| `tests/ablation/test_metrics.py` | Each of the 4 extractors fed a synthetic summary JSON returns the expected scalar; missing-key extractor returns `None` (does not crash); registry duplicate-name error. |
| `tests/ablation/test_cli.py` | `scripts/run_ablation.py --help` works; argparse plumbs `--resume` to runner; exits 0 on a 2-combination toy sweep with mocked phases. |
| `tests/ablation/test_axes_smoke.py` | "Wire-up smoke": for each of the 3 axis groups in spec §5.3 (dual-channel + threshold; margin + LSH; LLM + encoder), assert that `_apply_overrides` mutates the **base** config without raising, i.e. the dotted path is reachable on today's schema. **No phase is executed** — this is purely a config-reachability check.
| `tests/ablation/data/example_summary.json` | Canned `extract` summary fixture used by `test_metrics.py` and `test_runner.py`. |
| `tests/ablation/data/example_evaluation_summary.json` | Canned `evaluation_summary.json` fixture (for `pass_at_1`). |

**Modified existing files (none).**

This phase is purely additive. The runner consumes the public Phase 1 surface (`PhaseRegistry`, `PhaseOrchestrator.dispatch_phase`, `RunStateManager`) without modifying any of those classes.

**Unchanged (do NOT touch):**

- All files under `wfcllm/encoder/`, `wfcllm/watermark/`, `wfcllm/extract/`, `wfcllm/evaluation/`, `wfcllm/pretrain/`, `wfcllm/cli/`, `wfcllm/orchestration/`, `wfcllm/lang/`, `wfcllm/datasets/`, `wfcllm/common/`.
- All existing tests outside `tests/ablation/`.
- `experiment/`, `tools/` directories — project rule.
- `run.py` (the top-level shell).

---

## Conventions

**Test command:** `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest <path> -v`

**Commit message format:** `<type>: <description>` per `CLAUDE.md`. One commit per task end. No `Co-Authored-By` trailer in this project.

**Frozen dataclass everywhere.** `SweepSpec`, `ResolvedConfig`, `MetricExtractor` are `@dataclass(frozen=True)` so equality is by value and the cartesian-product loop is reorder-safe.

**Deterministic run IDs.** `short_hash` uses `hashlib.sha256(json.dumps(axes_values, sort_keys=True).encode()).hexdigest()[:8]` — same axes_values → same hash, regardless of dict insertion order. Ordering of `enumerate(spec.expand())` is deterministic because `itertools.product` follows the YAML's declared axis order; `axes` is loaded into a regular `dict` (Python 3.7+ preserves insertion order).

**Resume contract.** `AblationRunner._already_done(run_id)` reads `output_dir/results.jsonl` line-by-line, parses each `{"run_id": ...}` JSON object, and returns `True` if `run_id` matches. No file locking — single-process. If `results.jsonl` doesn't exist or is empty, returns `False`.

**Phase invocation contract.** The runner builds a synthetic `argparse.Namespace` carrying the resolved config under `_config_cache` (matching `wfcllm.cli.entry:main`'s contract on line 87). It mints `args.force = True` so the orchestrator's "phase already done → skip" check is bypassed for the per-run state, then calls `orchestrator.dispatch_phase(phase, args)` for each phase in `spec.phases`. **Per-run `RunStateManager` instance** writes its own `state.json` under `output_dir/runs/<run_id>/state.json`, isolated from the global `data/run_state.json`.

**Failure isolation.** If a single combination raises, the runner catches it, logs, writes `{"run_id": ..., "error": "<repr>"}` to `results.jsonl`, and continues. Crashes do not abort the whole sweep. Spec §5.3 is silent on this; we choose "log-and-continue" because the alternative (abort) makes overnight sweeps fragile.

**Metric registration.** Metrics are registered at module import time:

```python
@register_metric("z_score_mean")
def _z_score_mean(artefacts: dict[str, Path]) -> float | None:
    summary_path = artefacts.get("extract_summary")
    if summary_path is None or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("summary", {}).get("mean_z_score")
```

The `artefacts` dict maps stable keys (`"extract_summary"`, `"evaluation_summary"`, `"watermark_run_state"`) to paths; the runner builds it after each combination by reading the resolved config's output paths. Adding a new metric is a one-line `@register_metric` decorator — but we don't ship any beyond the 4 listed in spec §5.3.

**YAML schema (verbatim from spec §5.3 §535):**

```yaml
name: <str>                    # required, non-empty
base_config: <path>            # required, path to JSON config, relative paths resolved against YAML file's directory
phases: [<phase>, ...]         # required, non-empty; each must be a key registered in PhaseRegistry
axes:                          # required, non-empty mapping; values must be non-empty lists
  <dotted.path>: [v1, v2, ...]
output_dir: <path>             # required
metrics: [<name>, ...]         # required; each must be a registered metric name
```

Validation errors raise `ValueError` with a message identifying the bad field. No silent coercion.

---

## Task 1: Create `wfcllm/ablation/` package skeleton + first failing test

**Files:**
- Create: `wfcllm/ablation/__init__.py`
- Create: `wfcllm/ablation/sweep.py`
- Create: `tests/ablation/__init__.py`
- Create: `tests/ablation/test_sweep.py`

- [ ] **Step 1: Write `wfcllm/ablation/__init__.py`** (initial — sweep + dataclass exports only; runner/metrics added in later tasks)

```python
"""WFCLLM ablation framework (spec §5.3).

Minimum-viable Cartesian-product sweep + JSONL metrics aggregation.
No distributed scheduling, no HPO, no auto-suggestion.
"""
from wfcllm.ablation.sweep import ResolvedConfig, SweepSpec, short_hash

__all__ = ["ResolvedConfig", "SweepSpec", "short_hash"]
```

- [ ] **Step 2: Write `wfcllm/ablation/sweep.py`** — partial: `SweepSpec` dataclass + `short_hash` only (no `expand` yet so the next failing test bites)

```python
"""Sweep specification: YAML → Cartesian-product expansion."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def short_hash(values: dict[str, Any]) -> str:
    """Return a deterministic 8-char SHA-256 prefix of the JSON-serialised values."""
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class ResolvedConfig:
    """One concrete combination from a sweep: full config dict + the axis values that produced it."""
    axes_values: dict[str, Any]
    config: dict[str, Any]
    run_id: str

    def short_hash(self) -> str:
        return short_hash(self.axes_values)


@dataclass(frozen=True)
class SweepSpec:
    """Parsed YAML sweep specification."""
    name: str
    base_config: Path
    phases: tuple[str, ...]
    axes: tuple[tuple[str, tuple[Any, ...]], ...]   # ordered axis pairs (path, values)
    output_dir: Path
    metrics: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepSpec":
        raise NotImplementedError  # filled in next step

    def expand(self) -> Iterator[ResolvedConfig]:
        raise NotImplementedError  # filled in Task 2
```

- [ ] **Step 3: Write `tests/ablation/__init__.py`** as an empty file.

- [ ] **Step 4: Write `tests/ablation/test_sweep.py`** with the first failing test

```python
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
```

- [ ] **Step 5: Run the test — expect green for these three (only `short_hash` is exercised; `from_yaml`/`expand` not yet hit)**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_sweep.py -v`

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/ablation/__init__.py wfcllm/ablation/sweep.py \
        tests/ablation/__init__.py tests/ablation/test_sweep.py
git commit -m "feat: scaffold wfcllm/ablation with SweepSpec + ResolvedConfig + short_hash"
```

---

## Task 2: Implement `SweepSpec.from_yaml` + `.expand()` Cartesian product

**Files:**
- Modify: `wfcllm/ablation/sweep.py`
- Modify: `tests/ablation/test_sweep.py`

- [ ] **Step 1: Add failing tests for `from_yaml` + `expand` to `tests/ablation/test_sweep.py`**

Append to the existing file:

```python
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
```

- [ ] **Step 2: Run the new tests — expect 8 failures**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_sweep.py -v`

Expected: 3 passed (from Task 1) + 8 failed with `NotImplementedError` or `AttributeError`.

- [ ] **Step 3: Implement `from_yaml` + `expand` in `wfcllm/ablation/sweep.py`**

Replace the file body with:

```python
"""Sweep specification: YAML → Cartesian-product expansion."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


def short_hash(values: dict[str, Any]) -> str:
    """Return a deterministic 8-char SHA-256 prefix of the JSON-serialised values."""
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class ResolvedConfig:
    axes_values: dict[str, Any]
    config: dict[str, Any]
    run_id: str

    def short_hash(self) -> str:
        return short_hash(self.axes_values)


@dataclass(frozen=True)
class SweepSpec:
    name: str
    base_config: Path
    phases: tuple[str, ...]
    axes: tuple[tuple[str, tuple[Any, ...]], ...]
    output_dir: Path
    metrics: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepSpec":
        yaml_path = Path(path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"sweep yaml not found: {yaml_path}")
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"sweep yaml must be a mapping, got {type(raw).__name__}")

        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("sweep yaml: 'name' must be a non-empty string")

        base_config_raw = raw.get("base_config")
        if not base_config_raw:
            raise ValueError("sweep yaml: 'base_config' is required")
        base_config = (yaml_path.parent / base_config_raw).resolve()

        phases_raw = raw.get("phases")
        if not phases_raw or not isinstance(phases_raw, list):
            raise ValueError("sweep yaml: 'phases' must be a non-empty list")
        phases = tuple(str(p) for p in phases_raw)

        axes_raw = raw.get("axes")
        if not isinstance(axes_raw, dict) or not axes_raw:
            raise ValueError("sweep yaml: 'axes' must be a non-empty mapping")
        axes_pairs: list[tuple[str, tuple[Any, ...]]] = []
        for axis_path, values in axes_raw.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"sweep yaml: axes['{axis_path}'] must be a non-empty list, got {values!r}"
                )
            axes_pairs.append((str(axis_path), tuple(values)))

        output_dir_raw = raw.get("output_dir")
        if not output_dir_raw:
            raise ValueError("sweep yaml: 'output_dir' is required")
        output_dir = (yaml_path.parent / output_dir_raw).resolve()

        metrics_raw = raw.get("metrics")
        if not metrics_raw or not isinstance(metrics_raw, list):
            raise ValueError("sweep yaml: 'metrics' must be a non-empty list")
        metrics = tuple(str(m) for m in metrics_raw)

        return cls(
            name=name,
            base_config=base_config,
            phases=phases,
            axes=tuple(axes_pairs),
            output_dir=output_dir,
            metrics=metrics,
        )

    def load_base_config(self) -> dict[str, Any]:
        with open(self.base_config, encoding="utf-8") as f:
            return json.load(f)

    def expand(self) -> Iterator[ResolvedConfig]:
        base = self.load_base_config()
        axis_paths = [p for p, _ in self.axes]
        axis_value_lists = [v for _, v in self.axes]
        for index, combo in enumerate(itertools.product(*axis_value_lists)):
            axes_values = dict(zip(axis_paths, combo))
            cfg = copy.deepcopy(base)
            for path, value in axes_values.items():
                _apply_dotted_override(cfg, path, value)
            run_id = f"{index:04d}_{short_hash(axes_values)}"
            yield ResolvedConfig(
                axes_values=axes_values,
                config=cfg,
                run_id=run_id,
            )


def _apply_dotted_override(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Walk dotted_path, creating intermediate dicts if missing, set leaf."""
    parts = dotted_path.split(".")
    cursor: Any = cfg
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError(
                f"axes path '{dotted_path}': cannot descend into non-dict at '{part}'"
            )
        if part not in cursor or not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise ValueError(f"axes path '{dotted_path}': leaf parent is not a dict")
    cursor[parts[-1]] = value
```

- [ ] **Step 4: Re-run all sweep tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_sweep.py -v`

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/ablation/sweep.py tests/ablation/test_sweep.py
git commit -m "feat: implement SweepSpec.from_yaml + cartesian-product expand"
```

---

## Task 3: Implement `metrics.py` registry + 4 spec-mandated extractors

**Files:**
- Create: `wfcllm/ablation/metrics.py`
- Create: `tests/ablation/data/example_summary.json`
- Create: `tests/ablation/data/example_evaluation_summary.json`
- Create: `tests/ablation/test_metrics.py`
- Modify: `wfcllm/ablation/__init__.py`

- [ ] **Step 1: Write fixture `tests/ablation/data/example_summary.json`** (mirrors the schema written by `wfcllm/extract/pipeline.py:88-115` exactly)

```json
{
  "meta": {
    "input_file": "data/watermarked/humaneval_watermarked.jsonl",
    "total_samples": 100,
    "scored_samples": 90,
    "invalid_samples": 10
  },
  "summary": {
    "watermark_rate": 0.78,
    "watermark_rate_ci_95": [0.69, 0.85],
    "mean_z_score": 4.21,
    "std_z_score": 1.13,
    "mean_p_value": 0.018,
    "mean_blocks": 7.3,
    "embed_rate_distribution": {"mean": 0.62, "p50": 0.65},
    "mean_lexical_z_score": 2.0,
    "mean_green_fraction": 0.55,
    "mean_joint_score": 3.1,
    "joint_prediction_rate": 0.74,
    "mode_counts": {"fixed": 30, "adaptive": 60},
    "invalid_reason_counts": {"alignment_failed": 4, "adaptive_contract_invalid": 6}
  }
}
```

- [ ] **Step 2: Write fixture `tests/ablation/data/example_evaluation_summary.json`**

```json
{
  "modes": {
    "dual-channel": {
      "generation_metrics": {
        "pass_at_1": 0.42,
        "pass_at_10": 0.71
      },
      "detection_metrics": {
        "watermark_rate": 0.78,
        "z_score_mean": 4.21
      }
    }
  }
}
```

- [ ] **Step 3: Write `tests/ablation/test_metrics.py`** with the failing test set

```python
"""Tests for ablation metric extractors (spec §5.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from wfcllm.ablation.metrics import (
    METRIC_REGISTRY,
    get_metric,
    register_metric,
)

DATA_DIR = Path(__file__).parent / "data"


def _artefacts() -> dict[str, Path]:
    return {
        "extract_summary": DATA_DIR / "example_summary.json",
        "evaluation_summary": DATA_DIR / "example_evaluation_summary.json",
    }


def test_z_score_mean_extractor():
    metric = get_metric("z_score_mean")
    assert metric(_artefacts()) == pytest.approx(4.21)


def test_hit_rate_extractor():
    metric = get_metric("hit_rate")
    assert metric(_artefacts()) == pytest.approx(0.78)


def test_false_positive_rate_extractor_returns_none_when_field_absent():
    """Extract summary alone has no FPR — it lives in calibration outputs.

    The extractor must return None gracefully so JSONL row stays well-formed.
    """
    metric = get_metric("false_positive_rate")
    assert metric(_artefacts()) is None


def test_pass_at_1_extractor():
    metric = get_metric("pass_at_1")
    # Pulls the first mode's pass_at_1 from evaluation_summary
    assert metric(_artefacts()) == pytest.approx(0.42)


def test_extractor_returns_none_when_artefact_missing(tmp_path):
    metric = get_metric("z_score_mean")
    artefacts = {"extract_summary": tmp_path / "does_not_exist.json"}
    assert metric(artefacts) is None


def test_register_metric_rejects_duplicate():
    with pytest.raises(ValueError, match="already registered"):
        @register_metric("z_score_mean")
        def _conflicting(artefacts):  # noqa: ARG001
            return 1.0


def test_get_metric_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown metric"):
        get_metric("not_a_real_metric")


def test_registry_lists_all_four_spec_metrics():
    assert set(METRIC_REGISTRY.names()) >= {
        "z_score_mean", "hit_rate", "false_positive_rate", "pass_at_1",
    }
```

- [ ] **Step 4: Run — expect 8 failures (module not yet present)**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_metrics.py -v`

Expected: 8 errors (`ImportError`).

- [ ] **Step 5: Implement `wfcllm/ablation/metrics.py`**

```python
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
```

- [ ] **Step 6: Update `wfcllm/ablation/__init__.py`**

```python
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
```

- [ ] **Step 7: Re-run**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_metrics.py -v`

Expected: 8 passed.

- [ ] **Step 8: Commit**

```bash
git add wfcllm/ablation/metrics.py wfcllm/ablation/__init__.py \
        tests/ablation/data/example_summary.json \
        tests/ablation/data/example_evaluation_summary.json \
        tests/ablation/test_metrics.py
git commit -m "feat: add ablation metric registry with 4 spec-mandated extractors"
```

---

## Task 4: Implement `AblationRunner` core (no resume yet)

**Files:**
- Create: `wfcllm/ablation/runner.py`
- Create: `tests/ablation/test_runner.py`
- Modify: `wfcllm/ablation/__init__.py`

- [ ] **Step 1: Write `tests/ablation/test_runner.py`** with the first failing test

```python
"""Tests for AblationRunner (spec §5.3)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wfcllm.ablation.runner import AblationRunner
from wfcllm.ablation.sweep import SweepSpec


def _write_base(tmp_path: Path) -> Path:
    p = tmp_path / "base.json"
    p.write_text(
        '{"watermark": {"token_channel": {"mode": "semantic-only"}}, '
        '"extract": {"fpr_threshold": 3.0}}',
        encoding="utf-8",
    )
    return p


def _build_spec(tmp_path: Path, axes_yaml_block: str) -> SweepSpec:
    base = _write_base(tmp_path)
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text(
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
{axes_yaml_block}
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
        encoding="utf-8",
    )
    return SweepSpec.from_yaml(yaml_path)


class _FakeOrchestrator:
    """Stand-in for PhaseOrchestrator. Records calls; never invokes real phases."""

    def __init__(self):
        self.calls = []

    def dispatch_phase(self, phase, args):
        self.calls.append((phase, dict(getattr(args, "_config_cache", {}))))
        return 0


def test_runner_iterates_combos_and_writes_jsonl(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [semantic-only, dual-channel]",
    )
    fake = _FakeOrchestrator()
    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: fake)

    # Stub metric extractors so we don't depend on real artefacts:
    monkeypatch.setattr(
        "wfcllm.ablation.runner.get_metric",
        lambda name: (lambda artefacts: 1.23),
    )

    runner.run(spec)

    results_path = spec.output_dir / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2
    assert rows[0]["axes_values"]["watermark.token_channel.mode"] == "semantic-only"
    assert rows[1]["axes_values"]["watermark.token_channel.mode"] == "dual-channel"
    assert rows[0]["metrics"] == {"z_score_mean": 1.23}
    assert rows[0]["run_id"].startswith("0000_")
    assert rows[1]["run_id"].startswith("0001_")


def test_runner_persists_per_run_config(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [dual-channel]",
    )
    fake = _FakeOrchestrator()
    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: fake)
    monkeypatch.setattr("wfcllm.ablation.runner.get_metric", lambda name: lambda a: 0.0)

    runner.run(spec)

    [combo] = list(spec.expand())
    cfg_path = spec.output_dir / "runs" / combo.run_id / "config.json"
    assert cfg_path.exists()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["watermark"]["token_channel"]["mode"] == "dual-channel"
    assert payload["extract"]["fpr_threshold"] == 3.0


def test_runner_dispatches_each_phase_in_spec(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [dual-channel]",
    )
    fake = _FakeOrchestrator()
    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: fake)
    monkeypatch.setattr("wfcllm.ablation.runner.get_metric", lambda name: lambda a: 0.0)

    runner.run(spec)

    assert [c[0] for c in fake.calls] == ["watermark"]


def test_runner_continues_on_phase_failure_and_records_error(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [semantic-only, dual-channel]",
    )

    class _FailingOrchestrator:
        def __init__(self):
            self.n = 0

        def dispatch_phase(self, phase, args):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("boom")
            return 0

    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: _FailingOrchestrator())
    monkeypatch.setattr("wfcllm.ablation.runner.get_metric", lambda name: lambda a: 0.0)

    runner.run(spec)

    rows = [json.loads(l) for l in (spec.output_dir / "results.jsonl").read_text().splitlines() if l]
    assert len(rows) == 2
    assert "error" in rows[0]
    assert "boom" in rows[0]["error"]
    assert "error" not in rows[1]
```

- [ ] **Step 2: Run — expect failures (module missing)**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_runner.py -v`

Expected: errors on import.

- [ ] **Step 3: Implement `wfcllm/ablation/runner.py`**

```python
"""AblationRunner: iterate Cartesian product of a SweepSpec and aggregate metrics."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from wfcllm.ablation.metrics import get_metric
from wfcllm.ablation.sweep import ResolvedConfig, SweepSpec
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.pipeline import PhaseOrchestrator
from wfcllm.orchestration.prereq import PrereqRegistry
from wfcllm.orchestration.state import RunStateManager

logger = logging.getLogger(__name__)


class _OrchestratorLike(Protocol):
    def dispatch_phase(self, phase: str, args: argparse.Namespace) -> int: ...


def _default_orchestrator_factory(state: RunStateManager) -> _OrchestratorLike:
    """Build a real PhaseOrchestrator backed by the standard CLI runner registrations."""
    from wfcllm.cli.entry import _populate_phase_registry

    registry = PhaseRegistry()
    _populate_phase_registry(registry)
    return PhaseOrchestrator(
        state=state,
        phase_registry=registry,
        prereq_registry=PrereqRegistry(),
    )


class AblationRunner:
    """Sequentially execute a SweepSpec, one combination at a time.

    Crash-isolated: a single combination's exception is caught, logged, and
    written to results.jsonl as `{"run_id": ..., "error": ...}`. The next
    combination is then attempted.
    """

    def __init__(
        self,
        orchestrator_factory: Callable[[RunStateManager], _OrchestratorLike] | None = None,
    ) -> None:
        self._factory = orchestrator_factory or _default_orchestrator_factory

    def run(self, spec: SweepSpec) -> None:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        results_path = spec.output_dir / "results.jsonl"
        already = self._load_done_run_ids(results_path)

        for combo in spec.expand():
            if combo.run_id in already:
                logger.info("ablation: skip %s (resume)", combo.run_id)
                continue
            row = self._run_one(spec, combo)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[ablation] {combo.run_id} {row.get('metrics') or row.get('error')}")

    def _run_one(self, spec: SweepSpec, combo: ResolvedConfig) -> dict[str, Any]:
        run_dir = spec.output_dir / "runs" / combo.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(combo.config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            self._dispatch_phases(spec, combo, run_dir)
            metrics = self._collect_metrics(spec, combo, run_dir)
            return {
                "run_id": combo.run_id,
                "axes_values": combo.axes_values,
                "metrics": metrics,
            }
        except Exception as exc:  # noqa: BLE001 — log-and-continue is the contract
            logger.exception("ablation: combo %s failed", combo.run_id)
            return {
                "run_id": combo.run_id,
                "axes_values": combo.axes_values,
                "error": repr(exc),
            }

    def _dispatch_phases(self, spec: SweepSpec, combo: ResolvedConfig, run_dir: Path) -> None:
        state = RunStateManager(path=run_dir / "state.json")
        orchestrator = self._factory(state)
        for phase in spec.phases:
            args = self._build_args(combo, run_dir)
            rc = orchestrator.dispatch_phase(phase, args)
            if rc != 0:
                raise RuntimeError(f"phase '{phase}' returned non-zero rc={rc}")

    def _collect_metrics(
        self, spec: SweepSpec, combo: ResolvedConfig, run_dir: Path
    ) -> dict[str, Any]:
        artefacts = self._build_artefact_map(combo, run_dir)
        out: dict[str, Any] = {}
        for name in spec.metrics:
            extractor = get_metric(name)
            out[name] = extractor(artefacts)
        return out

    @staticmethod
    def _build_args(combo: ResolvedConfig, run_dir: Path) -> argparse.Namespace:
        """Synthesise the argparse.Namespace expected by phase runners.

        Mirrors what wfcllm.cli.entry:main attaches: `_config_cache`, `force=True`,
        `eval_only=False`, plus the small set of fields runners read directly from
        args (we copy the resolved config's relevant scalars so runners that
        prefer args-overrides over config still see the right values)."""
        ns = argparse.Namespace()
        ns._config_cache = copy.deepcopy(combo.config)
        ns.force = True
        ns.eval_only = False
        ns.resume = None
        ns.checkpoint = None
        ns.input_file = None
        ns.config = run_dir / "config.json"
        return ns

    @staticmethod
    def _build_artefact_map(combo: ResolvedConfig, run_dir: Path) -> dict[str, Path]:
        """Return the canonical artefact paths produced by the phases for this run.

        Keys here must match those used inside `wfcllm/ablation/metrics.py`.
        """
        cfg = combo.config
        wm_dir = Path(cfg.get("watermark", {}).get("output_dir") or run_dir / "watermark")
        ext_dir = Path(cfg.get("extract", {}).get("output_dir") or run_dir / "extract")
        # Best-effort artefact discovery — extractors return None when missing.
        extract_summary = next(iter(ext_dir.glob("*_summary.json")), None) if ext_dir.exists() else None
        evaluation_summary = ext_dir / "evaluation_summary.json"
        return {
            "extract_summary": extract_summary or ext_dir / "_missing.json",
            "evaluation_summary": evaluation_summary,
            "calibration": ext_dir / "calibration.json",
            "watermark_dir": wm_dir,
            "run_dir": run_dir,
        }

    @staticmethod
    def _load_done_run_ids(results_path: Path) -> set[str]:
        if not results_path.exists():
            return set()
        ids: set[str] = set()
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["run_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids
```

- [ ] **Step 4: Update `wfcllm/ablation/__init__.py`**

```python
"""WFCLLM ablation framework (spec §5.3)."""
from wfcllm.ablation.metrics import (
    METRIC_REGISTRY,
    MetricExtractor,
    get_metric,
    register_metric,
)
from wfcllm.ablation.runner import AblationRunner
from wfcllm.ablation.sweep import ResolvedConfig, SweepSpec, short_hash

__all__ = [
    "AblationRunner",
    "METRIC_REGISTRY",
    "MetricExtractor",
    "ResolvedConfig",
    "SweepSpec",
    "get_metric",
    "register_metric",
    "short_hash",
]
```

- [ ] **Step 5: Run runner tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_runner.py -v`

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/ablation/runner.py wfcllm/ablation/__init__.py tests/ablation/test_runner.py
git commit -m "feat: implement AblationRunner core with crash-isolation and JSONL output"
```

---

## Task 5: Add resume support test (separate task — distinct logic from Task 4)

**Files:**
- Modify: `tests/ablation/test_runner.py`

- [ ] **Step 1: Append failing resume test**

```python
def test_runner_skips_runs_already_in_results_jsonl(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [semantic-only, dual-channel]",
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    [combo0, combo1] = list(spec.expand())
    # Pre-seed results.jsonl with combo0 already done:
    (spec.output_dir / "results.jsonl").write_text(
        json.dumps({"run_id": combo0.run_id, "axes_values": combo0.axes_values, "metrics": {}}) + "\n",
        encoding="utf-8",
    )

    fake = _FakeOrchestrator()
    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: fake)
    monkeypatch.setattr("wfcllm.ablation.runner.get_metric", lambda name: lambda a: 0.5)
    runner.run(spec)

    rows = [json.loads(l) for l in (spec.output_dir / "results.jsonl").read_text().splitlines() if l]
    assert len(rows) == 2
    assert rows[0]["run_id"] == combo0.run_id  # original row preserved
    assert rows[0]["metrics"] == {}            # original metrics preserved (not 0.5)
    assert rows[1]["run_id"] == combo1.run_id
    # And only one combination was actually dispatched:
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "watermark"
```

- [ ] **Step 2: Run — should already pass** (resume logic was implemented in Task 4 Step 3 inside `run()`)

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_runner.py::test_runner_skips_runs_already_in_results_jsonl -v`

Expected: passed. If it fails, the resume code in `_load_done_run_ids` is incorrect — fix before continuing.

- [ ] **Step 3: Re-run all runner tests for full suite green**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_runner.py -v`

Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/ablation/test_runner.py
git commit -m "test: add ablation resume coverage"
```

---

## Task 6: Add `scripts/run_ablation.py` CLI + tests

**Files:**
- Create: `scripts/run_ablation.py`
- Create: `tests/ablation/test_cli.py`

- [ ] **Step 1: Write failing CLI test `tests/ablation/test_cli.py`**

```python
"""Tests for scripts/run_ablation.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_ablation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_run_ablation_under_test", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script_module():
    return _load_script_module()


def test_help_does_not_crash(capsys, script_module):
    with pytest.raises(SystemExit) as exc:
        script_module.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "yaml" in captured.out.lower() or "yaml" in captured.err.lower()


def test_main_invokes_runner_with_resume_flag(tmp_path, monkeypatch, script_module):
    base = tmp_path / "base.json"
    base.write_text('{"watermark": {}, "extract": {}}', encoding="utf-8")
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text(
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: [dual-channel]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
        encoding="utf-8",
    )

    captured_specs = []

    class _FakeRunner:
        def __init__(self, *a, **kw):  # noqa: ARG002
            pass
        def run(self, spec):
            captured_specs.append(spec)

    monkeypatch.setattr(script_module, "AblationRunner", _FakeRunner)

    rc = script_module.main([str(yaml_path)])
    assert rc == 0
    assert len(captured_specs) == 1
    assert captured_specs[0].name == "t"


def test_main_returns_nonzero_on_missing_yaml(capsys, script_module):
    rc = script_module.main(["does/not/exist.yaml"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower()
```

- [ ] **Step 2: Run — expect failure (script missing)**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_cli.py -v`

Expected: errors / failures (script does not exist yet).

- [ ] **Step 3: Implement `scripts/run_ablation.py`**

```python
#!/usr/bin/env python
"""Run an ablation sweep described by a YAML file (spec §5.3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.ablation import AblationRunner, SweepSpec  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an ablation sweep described by a YAML spec.",
    )
    parser.add_argument("yaml", help="Path to sweep YAML file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="No-op flag (kept for spec parity); resume is automatic from results.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"[error] sweep yaml not found: {yaml_path}", file=sys.stderr)
        return 2
    try:
        spec = SweepSpec.from_yaml(yaml_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    runner = AblationRunner()
    runner.run(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Re-run**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_cli.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_ablation.py tests/ablation/test_cli.py
git commit -m "feat: add scripts/run_ablation.py CLI for sweep execution"
```

---

## Task 7: Add example sweep YAML + axes-reachability smoke test

**Files:**
- Create: `configs/ablation/__init__.py`
- Create: `configs/ablation/example_dual_channel.yaml`
- Create: `tests/ablation/test_axes_smoke.py`

This task pins down spec §5.3's "三组消融轴" (dual-channel + threshold; margin + LSH; LLM + encoder). It does **not** run any real phase — it only confirms each axis path is reachable on today's `configs/base_config.json` schema, fulfilling spec §5.3's closing line: "AblationRunner 不需要懂参数语义，只做沿点号路径覆盖".

- [ ] **Step 1: Write `configs/ablation/__init__.py`**

```python
"""Example ablation sweep specifications (spec §5.3)."""
```

- [ ] **Step 2: Write `configs/ablation/example_dual_channel.yaml`** (verbatim from spec §5.3 §513)

```yaml
name: dual_channel_vs_threshold
base_config: ../base_config.json
phases: [watermark, extract]
axes:
  watermark.token_channel.mode: [semantic-only, lexical-only, dual-channel]
  extract.fpr_threshold:        [2.0, 2.5, 3.0, 3.5]
output_dir: ../../data/ablation/dual_channel_vs_threshold
metrics:
  - z_score_mean
  - hit_rate
  - false_positive_rate
  - pass_at_1
```

- [ ] **Step 3: Write `tests/ablation/test_axes_smoke.py`**

```python
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
```

- [ ] **Step 4: Run smoke tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/test_axes_smoke.py -v`

Expected: 4 passed. If a test fails, it pinpoints the missing schema key — DO NOT modify `base_config.json` or any phase dataclass. Stop and report; the spec promise is that all knobs already exist on the schema. (If a smoke test fails, it surfaces a real schema drift — out of scope to fix here. Per the project rule "测试失败立刻停下来报告", stop and surface for human review.)

- [ ] **Step 5: Commit**

```bash
git add configs/ablation/__init__.py configs/ablation/example_dual_channel.yaml \
        tests/ablation/test_axes_smoke.py
git commit -m "feat: add example dual-channel ablation spec + axes-reachability smoke tests"
```

---

## Task 8: Targeted regression run

**Files:** none modified — verification only.

- [ ] **Step 1: Run scoped pytest covering the new package + adjacent suites**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/ablation/ \
  tests/common/ \
  tests/integration/ \
  tests/lang/ \
  tests/datasets/ \
  -v 2>&1 | tail -60
```

Expected: all `tests/ablation/` green (24 tests), all other selected suites unchanged from the Phase 6 baseline. No new failures.

- [ ] **Step 2: Run the broader regression set to confirm zero impact on phase code**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/encoder/ \
  tests/extract/ \
  tests/watermark/ \
  tests/evaluation/ \
  -v 2>&1 | tail -60
```

Expected: exactly the 15 pre-existing baseline failures recorded at the end of Phase 6. **Zero new failures.** If a new failure appears, STOP and report per project rule "测试失败立刻停下来报告".

- [ ] **Step 3: Commit (none — verification only)**

No commit needed unless verification surfaced a regression to fix.

---

## Task 9: Final smoke — `run.py --status` + import smoke

**Files:** none modified — verification only.

- [ ] **Step 1: Confirm `run.py --status` still works**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status`

Expected: clean status table; no `ModuleNotFoundError`, no import error from importing `wfcllm.cli.entry` (which is reachable from the runner factory).

- [ ] **Step 2: Import smoke — ablation public surface**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "
from wfcllm.ablation import (
    AblationRunner, SweepSpec, ResolvedConfig,
    METRIC_REGISTRY, get_metric, register_metric, short_hash,
)
print('metric names:', METRIC_REGISTRY.names())
"
```

Expected: `metric names: ['false_positive_rate', 'hit_rate', 'pass_at_1', 'z_score_mean']`.

- [ ] **Step 3: CLI smoke — example YAML parses**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "
from wfcllm.ablation import SweepSpec
spec = SweepSpec.from_yaml('configs/ablation/example_dual_channel.yaml')
print('name:', spec.name)
print('combos:', sum(1 for _ in spec.expand()))
print('metrics:', spec.metrics)
"
```

Expected: `name: dual_channel_vs_threshold`, `combos: 12`, `metrics: ('z_score_mean', 'hit_rate', 'false_positive_rate', 'pass_at_1')`.

- [ ] **Step 4: Commit (none — verification only)**

Phase 7 chain is complete at this point.

---

## Phase 7 done-criteria checklist

- [ ] All Phase 7 commits land on `main` with `<type>: <description>` messages, in the order: scaffold → sweep → metrics → runner-core → runner-resume → CLI → example+smoke (≈7 commits).
- [ ] `pytest tests/ablation/` is fully green (≥ 24 tests).
- [ ] `pytest tests/encoder/ tests/extract/ tests/watermark/ tests/evaluation/ tests/common/ tests/integration/ tests/lang/ tests/datasets/` shows exactly the Phase 6 baseline failure list — **zero** new failures.
- [ ] `run.py --status` runs cleanly.
- [ ] `wfcllm/encoder/`, `wfcllm/watermark/`, `wfcllm/extract/`, `wfcllm/evaluation/`, `wfcllm/cli/`, `wfcllm/orchestration/`, `wfcllm/lang/`, `wfcllm/datasets/`, `wfcllm/common/` are **byte-identical** to the Phase 6 endpoint.
- [ ] No file under `experiment/` or `tools/` is touched.
- [ ] `requirements.txt` is **unchanged** — no new external deps.
- [ ] No new tunable parameter introduced anywhere in `configs/base_config.json` or any dataclass config.
- [ ] State-dict / weight files unaffected — Phase 7 never imports `wfcllm.encoder.train` or `wfcllm.watermark.token_channel.core.model` at runtime.
