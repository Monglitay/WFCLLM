# Phase 8: `watermark/generator.py` → `orchestrator.py` + `semantic_channel.py` Split Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1262-line `wfcllm/watermark/generator.py` into two responsibility-scoped files — `wfcllm/watermark/orchestrator.py` (main loop, token-channel runtime state, block ledger, cascade machinery, dataclasses) and `wfcllm/watermark/semantic_channel.py` (LSH verification, entropy estimation, adaptive-gamma resolution, alignment/finalization helpers) — without changing any algorithm, KV-cache snapshot timing, cascade fallback ordering, or visible attribute names. The old import path `wfcllm.watermark.generator` keeps working via a deprecation shim. Tests that monkey-patch `gen._verify_block`, `gen._verifier.verify`, `gen._keying.derive`, `gen._entropy_profile` continue to work unchanged.

**Architecture:** Pure code-move + responsibility split. `WatermarkGenerator` (in `orchestrator.py`) composes a single `SemanticChannel` instance via `self._semantic = SemanticChannel(self)`. `SemanticChannel` is a thin namespace class that holds a reference back to the generator and reads its mutable attributes (`_config`, `_entropy_profile`, `_gamma_schedule`, `_verifier`, `_keying`, `_entropy_est`, `_lsh_space`) directly — these attributes remain on the generator so existing tests' monkey-patches keep working. `WatermarkGenerator` retains thin proxy methods (`_verify_block`, `_classify_failure_reason`, `_resolve_gamma_for_*`, `_is_adaptive_runtime_enabled`, `_adaptive_mode`, `_profile_id`, `_build_alignment_summary`, `_finalize_stats`, `_initialize_adaptive_gamma`) that delegate one-line to `self._semantic.<same_name_without_leading_underscore>(...)`. Verbatim body moves only — zero algorithm changes.

**Tech Stack:** Python 3.10+, pytest, dataclasses, torch (existing dep), tree-sitter (existing dep). No new external deps.

**Spec reference:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §3 (target tree lines 82-83), §10 step 8 ("最有风险，放最后"), §11 ("不重写 generator.py 内部算法（仅按职责切两片）"), §12 ("`generator.py` 拆两片可能破坏 KV-cache snapshot 和 cascade 时序 / 放实施顺序最后；保留完整集成测试（特别是 cascade fallback 路径）").

**Out of scope (explicitly NOT in this phase):**

- Any algorithm change inside `generate()`, `_process_simple_block`, `_try_cascade`, retry-loop, cascade manager, KV-cache snapshot/rollback timing, token-channel bias logic. Spec §11 is explicit: "仅按职责切两片". This phase is verbatim moves.
- Removing the `EmbedStats`/`GenerateResult`/`TokenChannelRuntimeState` dataclasses from their current shape or adding new fields.
- Refactoring the token-channel runtime out of `orchestrator.py` into a third file. The spec target tree shows exactly two files; do not invent a third.
- Renaming any public/private attribute that tests currently monkey-patch (`_verify_block`, `_verifier`, `_keying`, `_entropy_est`, `_entropy_profile`, `_gamma_schedule`, `_lsh_space`).
- Touching `wfcllm/watermark/pipeline.py`, `wfcllm/watermark/cascade.py`, `wfcllm/watermark/retry_loop.py`, `wfcllm/watermark/context.py`, `wfcllm/watermark/kv_cache.py`, `wfcllm/watermark/interceptor.py`, `wfcllm/watermark/verifier.py`, `wfcllm/watermark/keying.py`, `wfcllm/watermark/lsh_space.py`, `wfcllm/watermark/diagnostics.py`, `wfcllm/watermark/config.py`, `wfcllm/watermark/adaptive_gamma/`, `wfcllm/watermark/token_channel/` — these are "不动" per spec §3 lines 84-113.
- Rewriting the first-party callers under `wfcllm/cli/runners.py:212` and `wfcllm/watermark/pipeline.py:21` — they keep importing from `wfcllm.watermark.generator` to exercise the shim (same Phase 2–7 convention).
- Rewriting test files. All 3000+ lines under `tests/watermark/test_generator.py` keep importing `from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult, EmbedStats` to exercise the shim.
- `experiment/`, `tools/`, `scripts/` directories — not touched per project rules.

---

## File Structure

**New canonical files:**

| New path | Responsibility | Source body |
|---|---|---|
| `wfcllm/watermark/orchestrator.py` | `WatermarkGenerator` (the orchestrator), `EmbedStats`, `GenerateResult`, `TokenChannelRuntimeState`, plus all token-channel runtime / block-ledger / cascade-replacement / stats-snapshot machinery. Calls `SemanticChannel` for verification + adaptive-gamma. | verbatim from current `generator.py` minus the methods listed under `semantic_channel.py` below |
| `wfcllm/watermark/semantic_channel.py` | `SemanticChannel` class: holds a back-ref to the generator and implements `verify_block`, `classify_failure_reason`, `resolve_gamma_for_block_text`, `resolve_gamma_for_entropy_units`, `is_adaptive_runtime_enabled`, `adaptive_mode`, `profile_id`, `initialize_adaptive_gamma`, `build_alignment_summary`, `finalize_stats`. | verbatim move from current `generator.py` — bodies unchanged, only the `self.` references rewritten to read from `self._orch.<attr>` |

**Modified existing files (shim / one-line import change only — no algorithm change):**

| Path | Change |
|---|---|
| `wfcllm/watermark/generator.py` | Replaced with deprecation shim re-exporting `WatermarkGenerator`, `GenerateResult`, `EmbedStats`, `TokenChannelRuntimeState` from `wfcllm.watermark.orchestrator`. Pattern matches Phase 2/3/4/5/6 shims. |
| `wfcllm/watermark/__init__.py` | Change line 5 (`from wfcllm.watermark.generator import ...`) to `from wfcllm.watermark.orchestrator import GenerateResult, WatermarkGenerator`. Public API (`__all__`) unchanged. |

**New tests:**

- `tests/watermark/test_generator_shim.py` — shim test: every public symbol resolvable from old `wfcllm.watermark.generator` path, emits `DeprecationWarning` exactly once, is identity-equal to the canonical symbol in `wfcllm.watermark.orchestrator`.
- `tests/watermark/test_semantic_channel.py` — unit tests for the `SemanticChannel` class in isolation: holds a back-ref, delegates `verify_block` to the underlying `_verifier.verify`, `resolve_gamma_for_entropy_units` honours `_gamma_schedule` when set / falls back to fixed gamma otherwise, `adaptive_mode` reports `"piecewise_quantile"` when adaptive is enabled and `"fixed"` otherwise, `profile_id` returns `None` for non-adaptive runs, `build_alignment_summary` is pure and reports `block_count_matches_total_blocks`.

**First-party callers that MUST continue importing from old paths** (exercise the shim — same Phase 6 convention):

```
wfcllm/cli/runners.py:212                    # WatermarkGenerator
wfcllm/watermark/pipeline.py:21              # WatermarkGenerator
tests/watermark/test_generator_integration.py:9
tests/watermark/test_context.py:10
tests/watermark/test_generator.py:14 + 1550
tests/watermark/test_embed_metadata_roundtrip.py:6
tests/watermark/test_pipeline_checkpoint.py:8
tests/watermark/test_pipeline.py:97 + 1060
```

This list is the source of truth for Task 5's grep verification.

**Unchanged (do NOT touch):**

- All `wfcllm/watermark/*.py` files NOT named `generator.py` or `__init__.py` (per spec §3 lines 84-113 "不动").
- All `wfcllm/watermark/adaptive_gamma/` and `wfcllm/watermark/token_channel/` files.
- All call-site files outside `wfcllm/watermark/` — they keep importing from old paths.
- All test files under `tests/watermark/` except the two new ones above.
- `experiment/`, `tools/`, `scripts/` — out of scope.

---

## Conventions

**Deprecation shim template** (matches Phase 2/3/4/5/6/7 pattern):

```python
"""Deprecated import path. Use wfcllm.watermark.orchestrator instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.generator is deprecated; use wfcllm.watermark.orchestrator",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.orchestrator import (
    EmbedStats,
    GenerateResult,
    TokenChannelRuntimeState,
    WatermarkGenerator,
)

__all__ = ["EmbedStats", "GenerateResult", "TokenChannelRuntimeState", "WatermarkGenerator"]
```

**SemanticChannel construction pattern** — the orchestrator holds its own state, the semantic channel reads it off the orchestrator reference:

```python
# wfcllm/watermark/semantic_channel.py
from __future__ import annotations

from typing import TYPE_CHECKING

from wfcllm.common.block_contract import BlockContract
from wfcllm.watermark.adaptive_gamma.entropy import ENTROPY_SCALE
from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
from wfcllm.watermark.adaptive_gamma.schedule import (
    GammaResolution,
    PiecewiseQuantileSchedule,
    quantize_gamma,
)
from wfcllm.watermark.diagnostics import FailureReason
from wfcllm.common.ast_parser import extract_statement_blocks

if TYPE_CHECKING:
    from wfcllm.watermark.orchestrator import WatermarkGenerator


class SemanticChannel:
    """LSH verification + adaptive-gamma resolution side of the generator.

    Reads the orchestrator's mutable attributes directly (verifier, keying,
    entropy estimator, LSH space, entropy profile, gamma schedule, config)
    so existing test monkey-patches on `generator._verifier` etc. keep
    flowing through unchanged.
    """

    def __init__(self, orchestrator: "WatermarkGenerator") -> None:
        self._orch = orchestrator

    # ... methods below ...
```

**Orchestrator delegation pattern** — `WatermarkGenerator` keeps thin proxy methods named exactly as before so test monkey-patches like `gen._verify_block = MagicMock(...)` still shadow the method:

```python
def _verify_block(self, event):
    return self._semantic.verify_block(event)

def _classify_failure_reason(self, event, verify_result):
    return self._semantic.classify_failure_reason(event, verify_result)

def _resolve_gamma_for_block_text(self, block_text):
    return self._semantic.resolve_gamma_for_block_text(block_text)

def _resolve_gamma_for_entropy_units(self, entropy_units):
    return self._semantic.resolve_gamma_for_entropy_units(entropy_units)

def _is_adaptive_runtime_enabled(self):
    return self._semantic.is_adaptive_runtime_enabled()

def _adaptive_mode(self):
    return self._semantic.adaptive_mode()

def _profile_id(self):
    return self._semantic.profile_id()

def _initialize_adaptive_gamma(self):
    self._semantic.initialize_adaptive_gamma()

def _build_alignment_summary(self, runtime_total_blocks, block_contracts):
    return self._semantic.build_alignment_summary(runtime_total_blocks, block_contracts)

def _finalize_stats(self, final_code):
    return self._semantic.finalize_stats(final_code)
```

**Test invocation** (memorise — applies to every Task that runs tests):

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest <paths> -v
```

`HF_HUB_OFFLINE=1` is mandatory per `CLAUDE.md` — testing without it triggers HF Hub access and fails.

**Commit message format** (per `CLAUDE.md`):

- `<type>: <description>`, type ∈ {feat, fix, refactor, test, docs, chore}
- **No** `Co-Authored-By:` trailer
- Imperative mood

---

## Methods Going to Each File

**Stays in `wfcllm/watermark/orchestrator.py` (with `WatermarkGenerator`):**

- `EmbedStats`, `GenerateResult`, `TokenChannelRuntimeState` dataclasses
- `__init__` (full body, but final line creates `self._semantic = SemanticChannel(self)` after the adaptive-gamma init call)
- `config` property
- `generate(prompt)` — main loop, unchanged
- Token-channel runtime: `_initialize_token_channel_runtime`, `_generation_mode`, `_semantic_channel_enabled`, `_lexical_channel_enabled`, `_create_token_channel_state`, `_reset_token_channel_state`, `_update_token_channel_state_after_simple_block`, `_resolve_token_channel_delta`, `_should_disable_token_channel_for_low_gate_fraction`, `_build_runtime_token_features`, `_resolve_runtime_token_span`, `_resolve_runtime_offset_mapping`, `_normalize_runtime_offset`, `_runtime_feature_source_variants`, `_fallback_runtime_token_features`, `_ensure_next_logits`, `_store_block_start_checkpoint_logits_if_supported`, `_apply_token_channel_bias`, `_build_retry_token_channel_hook`, `_build_retry_token_channel_hook_factory`, `_is_short_token_channel_block`, `_regenerate_short_lexical_only_block`
- Retry merging: `_merge_retry_diagnostics`, `_merge_retry_results`
- Cascade orchestration: `_try_cascade`, `_process_simple_block`, `_acquire_block_ledger`, `_capture_block_identity`, `_sync_retry_terminal_identity`, `_append_retry_attempts`, `_mark_block_success`, `_mark_block_failure`, `_activate_cascade_replacement_scope`, `_resolve_cascade_replacement_ordinal`, `_update_active_cascade_scope_for_compound`, `_serialize_block_ledgers`, `_fallback_cascade_metadata`, `_fallback_replacement_scope`, `_snapshot_runtime_stats`, `_restore_runtime_stats`
- `_sample_id_for_prompt`
- **Thin proxies** to `self._semantic.*`: `_verify_block`, `_classify_failure_reason`, `_resolve_gamma_for_block_text`, `_resolve_gamma_for_entropy_units`, `_is_adaptive_runtime_enabled`, `_adaptive_mode`, `_profile_id`, `_initialize_adaptive_gamma`, `_build_alignment_summary`, `_finalize_stats`

**Moves to `wfcllm/watermark/semantic_channel.py` (as `SemanticChannel` methods):**

| Old method on `WatermarkGenerator` (line range in current `generator.py`) | New method on `SemanticChannel` |
|---|---|
| `_verify_block(self, event)` — lines 662-689 | `verify_block(self, event)` |
| `_classify_failure_reason(self, event, verify_result)` — lines 984-1000 | `classify_failure_reason(self, event, verify_result)` |
| `_resolve_gamma_for_block_text(self, block_text)` — lines 777-779 | `resolve_gamma_for_block_text(self, block_text)` |
| `_resolve_gamma_for_entropy_units(self, entropy_units)` — lines 781-784 | `resolve_gamma_for_entropy_units(self, entropy_units)` |
| `_is_adaptive_runtime_enabled(self)` — lines 751-752 | `is_adaptive_runtime_enabled(self)` |
| `_adaptive_mode(self)` — lines 740-744 | `adaptive_mode(self)` |
| `_profile_id(self)` — lines 746-749 | `profile_id(self)` |
| `_initialize_adaptive_gamma(self)` — lines 754-775 | `initialize_adaptive_gamma(self)` |
| `_build_alignment_summary(self, runtime_total_blocks, block_contracts)` — lines 786-797 | `build_alignment_summary(self, runtime_total_blocks, block_contracts)` |
| `_finalize_stats(self, final_code)` — lines 1230-1262 | `finalize_stats(self, final_code)` |

**Internal body rewrites inside `SemanticChannel` methods** — every `self.<attr>` becomes `self._orch.<attr>` for these references (and **only** these):

- `self._config` → `self._orch._config`
- `self._entropy_est` → `self._orch._entropy_est`
- `self._verifier` → `self._orch._verifier`
- `self._keying` → `self._orch._keying`
- `self._lsh_space` → `self._orch._lsh_space` (not accessed in moved methods — only constructed in `__init__`)
- `self._entropy_profile` → `self._orch._entropy_profile` (read inside `initialize_adaptive_gamma` after `EntropyProfile.load` and inside `_gamma_schedule` construction)
- `self._gamma_schedule` → `self._orch._gamma_schedule`
- `self._semantic_channel_enabled()` (called from `finalize_stats` line 1236) → `self._orch._semantic_channel_enabled()`
- `self._verify_block(event)` (called from `finalize_stats` line 1259) → `self._orch._verify_block(event)` **(deliberately goes back through the orchestrator's proxy so test monkey-patches on `gen._verify_block` still take effect during `finalize_stats`)**

Inside `initialize_adaptive_gamma`, the two assignments `self._entropy_profile = ...` and `self._gamma_schedule = ...` become `self._orch._entropy_profile = ...` and `self._orch._gamma_schedule = ...` — keeping these as direct attributes on the orchestrator (not on `SemanticChannel`) preserves the existing test pattern `generator._entropy_profile = SimpleNamespace(...)` at `tests/watermark/test_pipeline.py:225`.

---

## Tasks

### Task 1: Scaffold `semantic_channel.py` skeleton with TODO body

**Files:**
- Create: `wfcllm/watermark/semantic_channel.py`
- Test: `tests/watermark/test_semantic_channel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/watermark/test_semantic_channel.py
"""Unit tests for SemanticChannel — the LSH+gamma slice of WatermarkGenerator."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_semantic_channel_module_exists():
    from wfcllm.watermark import semantic_channel
    assert hasattr(semantic_channel, "SemanticChannel")


def test_semantic_channel_holds_orchestrator_reference():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace()
    sc = SemanticChannel(orch)
    assert sc._orch is orch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_semantic_channel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.watermark.semantic_channel'`

- [ ] **Step 3: Write minimal implementation**

```python
# wfcllm/watermark/semantic_channel.py
"""LSH verification + adaptive-gamma resolution slice of WatermarkGenerator.

Holds a back-reference to the orchestrator and reads its mutable attributes
(_config, _entropy_est, _verifier, _keying, _entropy_profile, _gamma_schedule)
directly. This preserves the existing test pattern of monkey-patching
`generator._verify_block`, `generator._verifier.verify`, etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wfcllm.watermark.orchestrator import WatermarkGenerator


class SemanticChannel:
    def __init__(self, orchestrator: "WatermarkGenerator") -> None:
        self._orch = orchestrator
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_semantic_channel.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/semantic_channel.py tests/watermark/test_semantic_channel.py
git commit -m "feat: scaffold wfcllm/watermark/semantic_channel.py"
```

---

### Task 2: Scaffold `orchestrator.py` as side-by-side alias of generator.py

We do NOT yet split anything. We make `orchestrator.py` be a verbatim copy of `generator.py` (so all existing tests pass when imported through it), and then in subsequent tasks we move methods out of it.

**Files:**
- Create: `wfcllm/watermark/orchestrator.py`
- Test: `tests/watermark/test_generator_shim.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/watermark/test_generator_shim.py
"""Shim test: old wfcllm.watermark.generator path keeps working after Phase 8."""
from __future__ import annotations

import warnings


def test_orchestrator_module_exists():
    from wfcllm.watermark import orchestrator
    assert hasattr(orchestrator, "WatermarkGenerator")
    assert hasattr(orchestrator, "GenerateResult")
    assert hasattr(orchestrator, "EmbedStats")
    assert hasattr(orchestrator, "TokenChannelRuntimeState")


def test_orchestrator_exports_same_class_as_old_path():
    from wfcllm.watermark.orchestrator import WatermarkGenerator as W_new
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from wfcllm.watermark.generator import WatermarkGenerator as W_old
    assert W_new is W_old
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator_shim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.watermark.orchestrator'`

- [ ] **Step 3: Copy `generator.py` verbatim to `orchestrator.py`**

```bash
cp wfcllm/watermark/generator.py wfcllm/watermark/orchestrator.py
```

**Verify** the copy is byte-identical:

```bash
diff -q wfcllm/watermark/generator.py wfcllm/watermark/orchestrator.py
```

Expected: no output (identical).

- [ ] **Step 4: Verify `orchestrator` is importable but generator is still authoritative**

The first test (`test_orchestrator_module_exists`) should now PASS. The second (`test_orchestrator_exports_same_class_as_old_path`) should still FAIL because `wfcllm.watermark.generator.WatermarkGenerator` and `wfcllm.watermark.orchestrator.WatermarkGenerator` are currently *different* class objects (two independent module loads). That is expected — Task 4 fixes it by making `generator.py` re-export from `orchestrator.py`.

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator_shim.py::test_orchestrator_module_exists -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/orchestrator.py tests/watermark/test_generator_shim.py
git commit -m "feat: scaffold wfcllm/watermark/orchestrator.py as verbatim copy of generator.py"
```

---

### Task 3: Move semantic-channel methods into `SemanticChannel`

This is the bulk of the work. We move 10 methods (the ones tabled under "Methods Going to Each File" above) out of `WatermarkGenerator` in `orchestrator.py` and into `SemanticChannel`. We then replace each removed method on `WatermarkGenerator` with a one-line proxy. The orchestrator's `__init__` gets one new line creating the SemanticChannel.

**Files:**
- Modify: `wfcllm/watermark/semantic_channel.py`
- Modify: `wfcllm/watermark/orchestrator.py`
- Test: `tests/watermark/test_semantic_channel.py` (extend), `tests/watermark/test_generator.py` (existing — must stay green)

- [ ] **Step 1: Add failing tests for the new `SemanticChannel` surface**

Append to `tests/watermark/test_semantic_channel.py`:

```python
def test_resolve_gamma_for_entropy_units_falls_back_to_fixed_when_no_schedule():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    from wfcllm.watermark.adaptive_gamma.schedule import quantize_gamma

    orch = SimpleNamespace(
        _config=SimpleNamespace(lsh_d=4, lsh_gamma=0.5),
        _gamma_schedule=None,
        _entropy_est=SimpleNamespace(estimate_block_entropy_units=lambda _t: 7),
    )
    sc = SemanticChannel(orch)
    result = sc.resolve_gamma_for_entropy_units(7)
    assert result == quantize_gamma(0.5, 4)


def test_is_adaptive_runtime_enabled_tracks_orch_gamma_schedule():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(_gamma_schedule=None)
    sc = SemanticChannel(orch)
    assert sc.is_adaptive_runtime_enabled() is False
    orch._gamma_schedule = object()
    assert sc.is_adaptive_runtime_enabled() is True


def test_adaptive_mode_reports_fixed_when_disabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=None,
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(strategy="piecewise_quantile")),
    )
    sc = SemanticChannel(orch)
    assert sc.adaptive_mode() == "fixed"


def test_adaptive_mode_reports_strategy_when_enabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=object(),
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(strategy="piecewise_quantile")),
    )
    sc = SemanticChannel(orch)
    assert sc.adaptive_mode() == "piecewise_quantile"


def test_profile_id_returns_none_when_disabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=None,
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(profile_id="x")),
    )
    sc = SemanticChannel(orch)
    assert sc.profile_id() is None


def test_profile_id_returns_config_value_when_enabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=object(),
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(profile_id="entropy-profile-v1")),
    )
    sc = SemanticChannel(orch)
    assert sc.profile_id() == "entropy-profile-v1"


def test_build_alignment_summary_is_pure():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    sc = SemanticChannel(SimpleNamespace())
    summary = sc.build_alignment_summary(5, [object(), object(), object(), object(), object()])
    assert summary == {
        "final_block_count": 5,
        "generator_total_blocks": 5,
        "block_count_matches_total_blocks": True,
    }
    summary2 = sc.build_alignment_summary(7, [object(), object(), object()])
    assert summary2 == {
        "final_block_count": 3,
        "generator_total_blocks": 7,
        "block_count_matches_total_blocks": False,
    }
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_semantic_channel.py -v`
Expected: 7 failures with `AttributeError: 'SemanticChannel' object has no attribute '...'`

- [ ] **Step 3: Move method bodies into `SemanticChannel`**

Rewrite `wfcllm/watermark/semantic_channel.py` to be the full file:

```python
"""LSH verification + adaptive-gamma resolution slice of WatermarkGenerator.

Holds a back-reference to the orchestrator and reads its mutable attributes
(_config, _entropy_est, _verifier, _keying, _entropy_profile, _gamma_schedule)
directly. This preserves the existing test pattern of monkey-patching
`generator._verify_block`, `generator._verifier.verify`, etc.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.common.block_contract import BlockContract
from wfcllm.watermark.adaptive_gamma.entropy import ENTROPY_SCALE
from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
from wfcllm.watermark.adaptive_gamma.schedule import (
    GammaResolution,
    PiecewiseQuantileSchedule,
    quantize_gamma,
)
from wfcllm.watermark.diagnostics import FailureReason

if TYPE_CHECKING:
    from wfcllm.watermark.orchestrator import WatermarkGenerator

logger = logging.getLogger(__name__)


class SemanticChannel:
    def __init__(self, orchestrator: "WatermarkGenerator") -> None:
        self._orch = orchestrator

    def verify_block(self, event):
        """Verify a single block against LSH criteria."""
        orch = self._orch
        entropy_units = orch._entropy_est.estimate_block_entropy_units(event.block_text)
        block_entropy = entropy_units / ENTROPY_SCALE
        margin = orch._entropy_est.compute_margin(block_entropy, orch._config)
        gamma_resolution = self.resolve_gamma_for_entropy_units(entropy_units)
        valid_set = orch._keying.derive(
            event.parent_node_type or "module",
            k=gamma_resolution.k,
        )
        result = orch._verifier.verify(event.block_text, valid_set, margin)

        logger.debug(
            "[simple block] node=%s parent=%s entropy=%.4f margin_thresh=%.4f "
            "gamma_target=%.4f k=%d gamma_effective=%.4f\n"
            "  sig=%s in_valid=%s valid_set_size=%d min_margin=%.4f passed=%s\n"
            "  text=%r",
            event.node_type, event.parent_node_type,
            block_entropy, margin,
            gamma_resolution.gamma_target,
            gamma_resolution.k,
            gamma_resolution.gamma_effective,
            result.lsh_signature,
            result.lsh_signature in valid_set,
            len(valid_set), result.min_margin, result.passed,
            event.block_text[:80],
        )
        return result

    def classify_failure_reason(self, event, verify_result) -> str:
        orch = self._orch
        if verify_result.passed:
            return FailureReason.unknown.value
        entropy_units = orch._entropy_est.estimate_block_entropy_units(event.block_text)
        block_entropy = entropy_units / ENTROPY_SCALE
        margin_threshold = orch._entropy_est.compute_margin(block_entropy, orch._config)
        in_valid_set = verify_result.in_valid_set
        if in_valid_set is None:
            return FailureReason.unknown.value
        margin_passed = verify_result.min_margin > margin_threshold
        if not in_valid_set and margin_passed:
            return FailureReason.signature_miss.value
        if in_valid_set and not margin_passed:
            return FailureReason.margin_miss.value
        if not in_valid_set and not margin_passed:
            return FailureReason.signature_and_margin_miss.value
        return FailureReason.unknown.value

    def resolve_gamma_for_block_text(self, block_text: str) -> GammaResolution:
        orch = self._orch
        entropy_units = orch._entropy_est.estimate_block_entropy_units(block_text)
        return self.resolve_gamma_for_entropy_units(entropy_units)

    def resolve_gamma_for_entropy_units(self, entropy_units: int) -> GammaResolution:
        orch = self._orch
        if orch._gamma_schedule is not None:
            return orch._gamma_schedule.resolve(entropy_units, orch._config.lsh_d)
        return quantize_gamma(orch._config.lsh_gamma, orch._config.lsh_d)

    def is_adaptive_runtime_enabled(self) -> bool:
        return self._orch._gamma_schedule is not None

    def adaptive_mode(self) -> str:
        if self.is_adaptive_runtime_enabled():
            return self._orch._config.adaptive_gamma.strategy
        return "fixed"

    def profile_id(self) -> str | None:
        if not self.is_adaptive_runtime_enabled():
            return None
        return self._orch._config.adaptive_gamma.profile_id

    def initialize_adaptive_gamma(self) -> None:
        orch = self._orch
        adaptive_config = orch._config.adaptive_gamma
        if not adaptive_config.enabled:
            return
        if adaptive_config.profile_path is None:
            return
        if adaptive_config.strategy != "piecewise_quantile":
            raise ValueError(
                f"unsupported adaptive gamma strategy: {adaptive_config.strategy}"
            )

        orch._entropy_profile = EntropyProfile.load(adaptive_config.profile_path)
        anchor_quantiles = tuple(adaptive_config.anchors.keys())
        anchor_gammas = tuple(
            adaptive_config.anchors[quantile]
            for quantile in anchor_quantiles
        )
        orch._gamma_schedule = PiecewiseQuantileSchedule(
            profile=orch._entropy_profile,
            anchor_quantiles=anchor_quantiles,
            anchor_gammas=anchor_gammas,
        )

    def build_alignment_summary(
        self,
        runtime_total_blocks: int,
        block_contracts: list[BlockContract],
    ) -> dict[str, int | bool]:
        final_block_count = len(block_contracts)
        return {
            "final_block_count": final_block_count,
            "generator_total_blocks": runtime_total_blocks,
            "block_count_matches_total_blocks": final_block_count == runtime_total_blocks,
        }

    def finalize_stats(self, final_code: str) -> tuple[int, int]:
        """Recompute final simple-block totals from the emitted code."""
        orch = self._orch
        all_blocks = extract_statement_blocks(final_code)
        simple_blocks = [block for block in all_blocks if block.block_type == "simple"]
        if not simple_blocks:
            return 0, 0
        if not orch._semantic_channel_enabled():
            return len(simple_blocks), 0

        block_by_id = {block.block_id: block for block in all_blocks}
        embedded_blocks = 0
        for block in simple_blocks:
            parent_node_type = (
                block_by_id[block.parent_id].node_type
                if block.parent_id is not None
                else "module"
            )
            event = type(
                "_FinalBlockEvent",
                (),
                {
                    "block_text": block.source,
                    "block_type": "simple",
                    "node_type": block.node_type,
                    "parent_node_type": parent_node_type,
                    "token_start_idx": 0,
                    "token_count": 0,
                },
            )()
            if orch._verify_block(event).passed:
                embedded_blocks += 1

        return len(simple_blocks), embedded_blocks
```

- [ ] **Step 4: Rewrite `WatermarkGenerator` in `orchestrator.py` — proxies + composition**

Two edits in `wfcllm/watermark/orchestrator.py`:

**Edit A**: At the top of the file, add the import (after the existing `from wfcllm.watermark.verifier import ProjectionVerifier` line):

```python
from wfcllm.watermark.semantic_channel import SemanticChannel
```

**Edit B**: In `WatermarkGenerator.__init__`, immediately after the line `self._initialize_adaptive_gamma()` (currently line 123), but **change** `self._initialize_adaptive_gamma()` to construct the semantic channel first, then call init through it. Replace these three lines:

```python
        self._entropy_profile: EntropyProfile | None = None
        self._gamma_schedule: PiecewiseQuantileSchedule | None = None
        self._initialize_adaptive_gamma()
```

with:

```python
        self._entropy_profile: EntropyProfile | None = None
        self._gamma_schedule: PiecewiseQuantileSchedule | None = None
        self._semantic = SemanticChannel(self)
        self._initialize_adaptive_gamma()
```

(The `_initialize_adaptive_gamma` proxy added below now writes back to `self._entropy_profile` / `self._gamma_schedule` via the SemanticChannel.)

**Edit C**: Replace each of the following 10 method bodies in `WatermarkGenerator` with a one-line proxy. Do this by deleting the old method body and inserting the proxy in its place:

```python
    def _verify_block(self, event):
        return self._semantic.verify_block(event)

    def _classify_failure_reason(self, event, verify_result) -> str:
        return self._semantic.classify_failure_reason(event, verify_result)

    def _resolve_gamma_for_block_text(self, block_text: str) -> GammaResolution:
        return self._semantic.resolve_gamma_for_block_text(block_text)

    def _resolve_gamma_for_entropy_units(self, entropy_units: int) -> GammaResolution:
        return self._semantic.resolve_gamma_for_entropy_units(entropy_units)

    def _is_adaptive_runtime_enabled(self) -> bool:
        return self._semantic.is_adaptive_runtime_enabled()

    def _adaptive_mode(self) -> str:
        return self._semantic.adaptive_mode()

    def _profile_id(self) -> str | None:
        return self._semantic.profile_id()

    def _initialize_adaptive_gamma(self) -> None:
        self._semantic.initialize_adaptive_gamma()

    def _build_alignment_summary(
        self,
        runtime_total_blocks: int,
        block_contracts: list[BlockContract],
    ) -> dict[str, int | bool]:
        return self._semantic.build_alignment_summary(runtime_total_blocks, block_contracts)

    def _finalize_stats(self, final_code: str) -> tuple[int, int]:
        return self._semantic.finalize_stats(final_code)
```

The 10 methods to replace, by current `generator.py` line range:
- `_verify_block` — lines 662-689
- `_try_cascade` is **NOT** moved (cascade orchestration stays on orchestrator)
- `_adaptive_mode` — lines 740-744
- `_profile_id` — lines 746-749
- `_is_adaptive_runtime_enabled` — lines 751-752
- `_initialize_adaptive_gamma` — lines 754-775
- `_resolve_gamma_for_block_text` — lines 777-779
- `_resolve_gamma_for_entropy_units` — lines 781-784
- `_build_alignment_summary` — lines 786-797
- `_classify_failure_reason` — lines 984-1000
- `_finalize_stats` — lines 1230-1262

After this edit, `orchestrator.py` should be roughly 870 lines (1262 - ~400 moved out + 10 proxies of ~3 lines each).

**Imports cleanup in `orchestrator.py`** — after moving the bodies, these imports are no longer used by `orchestrator.py` and should be removed:
- `from wfcllm.watermark.adaptive_gamma.entropy import ENTROPY_SCALE` (only used inside `verify_block` and `classify_failure_reason`)
- `from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile` (only used inside `initialize_adaptive_gamma`)
- `from wfcllm.watermark.adaptive_gamma.schedule import GammaResolution, PiecewiseQuantileSchedule, quantize_gamma` (only used inside the moved methods)
- `from wfcllm.watermark.diagnostics import FailureReason` (only used inside `classify_failure_reason`) — BUT keep `BlockLifecycleRecord`, `hash_block_text`, `summarize_sample_diagnostics` since those are used elsewhere in the orchestrator. Edit the import line to drop only `FailureReason`.

Keep:
- `from wfcllm.common.ast_parser import extract_statement_blocks` — still used? Check: after `_finalize_stats` moves, this is **not** used in orchestrator anymore. Remove it.
- `from wfcllm.common.block_contract import BlockContract, build_block_contracts` — `build_block_contracts` is still used in `generate()`; `BlockContract` is used in the `_build_alignment_summary` proxy type hint. Keep both.

Type annotations on the proxies use `GammaResolution` and `BlockContract` — `GammaResolution` is no longer imported. Two choices: re-import it locally just for the type hint, OR drop the annotation. Drop the annotation on the two proxies (`_resolve_gamma_for_block_text`, `_resolve_gamma_for_entropy_units`) — type hints on private methods don't need to survive the move, and reimporting `GammaResolution` just for hints feels noisy:

```python
    def _resolve_gamma_for_block_text(self, block_text: str):
        return self._semantic.resolve_gamma_for_block_text(block_text)

    def _resolve_gamma_for_entropy_units(self, entropy_units: int):
        return self._semantic.resolve_gamma_for_entropy_units(entropy_units)
```

- [ ] **Step 5: Run the new `test_semantic_channel.py` tests + the entire existing watermark test suite**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_semantic_channel.py tests/watermark/ -v --no-header 2>&1 | tail -60
```

Expected:
- All 9 (2 from Task 1 + 7 from this task) `test_semantic_channel.py` tests PASS.
- The pre-existing watermark suite (which imports from `wfcllm.watermark.generator` — still pointing at our `orchestrator.py`-copy at this point) maintains its Phase 7 baseline: **651 passed + 15 failed** with the *same* 15 baseline failures (see Phase 6/7 docs for the exact list). The orchestrator-internal change must NOT introduce a single new failure.

If a *new* failure appears, STOP and report. Do NOT attempt a second fix without checking what changed — per loop boundary: "出现任何测试失败立刻停下来报告, 同一错误尝试 2 次还没解决就停".

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/semantic_channel.py wfcllm/watermark/orchestrator.py tests/watermark/test_semantic_channel.py
git commit -m "refactor: split semantic-channel methods into wfcllm.watermark.semantic_channel"
```

---

### Task 4: Replace `wfcllm/watermark/generator.py` with deprecation shim

Now that `orchestrator.py` is the canonical home of `WatermarkGenerator`, replace `generator.py` with a deprecation shim. After this, `wfcllm.watermark.generator.WatermarkGenerator is wfcllm.watermark.orchestrator.WatermarkGenerator`.

**Files:**
- Modify: `wfcllm/watermark/generator.py` (was canonical, becomes shim)
- Test: `tests/watermark/test_generator_shim.py` (extend)

- [ ] **Step 1: Extend the failing test**

Append to `tests/watermark/test_generator_shim.py`:

```python
def test_deprecation_warning_emitted_on_old_path_import():
    import importlib
    import sys
    # Force a clean re-import to observe the warning
    for mod in ["wfcllm.watermark.generator"]:
        sys.modules.pop(mod, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.import_module("wfcllm.watermark.generator")
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("wfcllm.watermark.generator" in str(w.message) for w in deprecations)


def test_old_symbols_resolvable_through_shim():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from wfcllm.watermark.generator import (
            EmbedStats,
            GenerateResult,
            TokenChannelRuntimeState,
            WatermarkGenerator,
        )
    from wfcllm.watermark.orchestrator import (
        EmbedStats as E2,
        GenerateResult as G2,
        TokenChannelRuntimeState as T2,
        WatermarkGenerator as W2,
    )
    assert EmbedStats is E2
    assert GenerateResult is G2
    assert TokenChannelRuntimeState is T2
    assert WatermarkGenerator is W2
```

- [ ] **Step 2: Run extended test to verify both new + the Task 2 leftover fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator_shim.py -v`
Expected: `test_orchestrator_exports_same_class_as_old_path` FAILs (because the two modules currently hold *independent* class objects after Task 2's `cp`), and the two new tests FAIL too.

- [ ] **Step 3: Replace `generator.py` with shim**

Overwrite `wfcllm/watermark/generator.py` with the deprecation shim shown in the Conventions section:

```python
"""Deprecated import path. Use wfcllm.watermark.orchestrator instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.generator is deprecated; use wfcllm.watermark.orchestrator",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.orchestrator import (
    EmbedStats,
    GenerateResult,
    TokenChannelRuntimeState,
    WatermarkGenerator,
)

__all__ = ["EmbedStats", "GenerateResult", "TokenChannelRuntimeState", "WatermarkGenerator"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator_shim.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/generator.py tests/watermark/test_generator_shim.py
git commit -m "refactor: replace watermark/generator.py with deprecation shim to orchestrator"
```

---

### Task 5: Update `wfcllm/watermark/__init__.py` to point at canonical path

The package `__init__.py` should source `WatermarkGenerator` and `GenerateResult` from the canonical `orchestrator.py`, not through the deprecation shim. This keeps internal package code on canonical paths (Phase 6 convention) while leaving first-party call sites (`pipeline.py`, `cli/runners.py`) on the old path to exercise the shim.

**Files:**
- Modify: `wfcllm/watermark/__init__.py`

- [ ] **Step 1: Read current `__init__.py`**

Confirm line 5 currently reads:

```python
from wfcllm.watermark.generator import GenerateResult, WatermarkGenerator
```

- [ ] **Step 2: Replace it**

Change line 5 to:

```python
from wfcllm.watermark.orchestrator import GenerateResult, WatermarkGenerator
```

Do NOT touch any other line — `__all__` and the other imports remain.

- [ ] **Step 3: Verify `import wfcllm.watermark` no longer emits the deprecation warning**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -W error::DeprecationWarning -c "import wfcllm.watermark; print(wfcllm.watermark.WatermarkGenerator)"
```

Expected output: a class repr like `<class 'wfcllm.watermark.orchestrator.WatermarkGenerator'>`. NO `DeprecationWarning` should be raised (the `-W error::DeprecationWarning` flag turns any leak into an error).

Note: this check is intentionally a smoke test, not a unit test — adding it to the test suite would force `-W error::DeprecationWarning` globally, which would break the *other* shim tests in Tasks 4 and 6 that deliberately observe the warning.

- [ ] **Step 4: Run the full watermark suite again**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -v --no-header 2>&1 | tail -30
```

Expected: Phase 7 baseline maintained (651 passed + 15 failed, same 15 baseline failures), **plus** the new `test_semantic_channel.py` (9 tests) and `test_generator_shim.py` (4 tests) all pass.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/__init__.py
git commit -m "refactor: route wfcllm.watermark public API through orchestrator canonical path"
```

---

### Task 6: First-party caller audit (verify they still go through the shim)

Per Phase 6/7 convention, first-party callers in `wfcllm/cli/runners.py` and `wfcllm/watermark/pipeline.py` continue importing from `wfcllm.watermark.generator` to exercise the shim. This task is a verification step, not a code change.

**Files:**
- No edits. Verification only.

- [ ] **Step 1: Confirm the two callers still import from the old path**

```bash
grep -n "from wfcllm.watermark.generator" wfcllm/cli/runners.py wfcllm/watermark/pipeline.py
```

Expected output:

```
wfcllm/cli/runners.py:212:    from wfcllm.watermark.generator import WatermarkGenerator
wfcllm/watermark/pipeline.py:21:from wfcllm.watermark.generator import WatermarkGenerator
```

If the lines have shifted, that's fine as long as the imports remain. If they've been *changed* to a different module, restore them (per Phase 6 shim-exercise convention).

- [ ] **Step 2: Confirm no other `wfcllm/` files reference the old path beyond these two + the shim itself**

```bash
grep -rn "from wfcllm.watermark.generator\|wfcllm.watermark.generator import" wfcllm/
```

Expected output: exactly three matches:
- `wfcllm/cli/runners.py:212`
- `wfcllm/watermark/pipeline.py:21`
- (no third — `wfcllm/watermark/__init__.py` should have been retargeted in Task 5)

If a fourth match exists, STOP and report — something was missed.

- [ ] **Step 3: Confirm tests still reach for the old path (intentional — they exercise the shim)**

```bash
grep -rn "from wfcllm.watermark.generator" tests/
```

Expected output (matches Phase 7 audit):

```
tests/watermark/test_generator_integration.py:9
tests/watermark/test_context.py:10
tests/watermark/test_generator.py:14
tests/watermark/test_generator.py:1550
tests/watermark/test_embed_metadata_roundtrip.py:6
tests/watermark/test_pipeline_checkpoint.py:8
tests/watermark/test_pipeline.py:97
tests/watermark/test_pipeline.py:1060
```

If anything is added or removed unexpectedly, investigate.

- [ ] **Step 4: No commit** (verification only).

---

### Task 7: Cascade-fallback integration smoke test

Per spec §12: "`generator.py` 拆两片可能破坏 KV-cache snapshot 和 cascade 时序 / 放实施顺序最后；保留完整集成测试（特别是 cascade fallback 路径）". We don't write new cascade tests — the existing `tests/watermark/test_generator.py` already has comprehensive cascade coverage (the file is 3000+ lines, with extensive `_FailingOrchestrator`-style mock tests and cascade-rollback assertions). We re-run it explicitly to confirm cascade behaviour is unchanged.

**Files:**
- No edits. Verification only.

- [ ] **Step 1: Run the cascade-heavy subset and confirm Phase 7 baseline**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py tests/watermark/test_pipeline.py tests/watermark/test_pipeline_checkpoint.py tests/watermark/test_generator_integration.py tests/watermark/test_embed_metadata_roundtrip.py -v --no-header 2>&1 | tail -50
```

Expected: same 15 baseline failures from Phase 6/7 (1 token_channel runtime + 2 token_channel train_corpus + 8 watermark train_workflow + 1 extract joint_detection + 3 extract pipeline stats — note: only the ones in these test files; counts below differ from Phase 7's full-tree count). Specifically for **these files**:

- `test_generator.py` should be all-green (or match the Phase 7 baseline exactly).
- `test_pipeline.py` should be all-green.
- `test_pipeline_checkpoint.py` should be all-green.
- `test_generator_integration.py` should be all-green.
- `test_embed_metadata_roundtrip.py` should be all-green.

Cascade-specific tests with names matching `cascade` should all pass:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -k "cascade" -v --no-header 2>&1 | tail -20
```

Expected: all cascade tests PASS.

- [ ] **Step 2: Run KV-cache snapshot tests explicitly**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -k "kv_cache or snapshot or rollback or checkpoint" -v --no-header 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 3: No commit** (verification only).

If any cascade or KV-cache test newly fails after the Phase 8 split, STOP — that's the exact risk spec §12 flagged. Do not attempt to "fix" by tweaking the split; instead report the regression to the user and let them decide.

---

### Task 8: Full Phase 6/7 baseline verification (post-split, end-to-end)

This is the same end-of-phase verification gate as Phases 6 and 7. Confirms zero new regressions outside the documented 15 baseline failures.

**Files:**
- No edits. Verification only.

- [ ] **Step 1: Run the canonical phase-6/7 baseline subset**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/ tests/extract/ tests/watermark/ --no-header 2>&1 | tail -20
```

Expected (matching Phase 7 baseline exactly): **651 passed + 15 failed** with the same 15 baseline failures:

- `tests/extract/test_joint_detection.py` × 1
- `tests/extract/test_pipeline.py::TestExtractPipelineStatistics` × 3
- `tests/watermark/token_channel/test_runtime.py` × 1
- `tests/watermark/token_channel/test_train_corpus.py` × 2
- `tests/watermark/test_train_workflow.py` × 8

After Phase 8, we ALSO expect the new tests to pass:

- `tests/watermark/test_semantic_channel.py` — 9 tests
- `tests/watermark/test_generator_shim.py` — 4 tests

Final expected total: **664 passed + 15 failed** in the encoder + extract + watermark subset.

- [ ] **Step 2: Sanity-check `run.py --status`**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status
```

Expected: clean output (whatever Phase 7 produced — no new errors).

- [ ] **Step 3: Confirm `wfcllm.watermark.orchestrator` is importable in isolation**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "from wfcllm.watermark.orchestrator import WatermarkGenerator, EmbedStats, GenerateResult, TokenChannelRuntimeState; from wfcllm.watermark.semantic_channel import SemanticChannel; print('OK', WatermarkGenerator, SemanticChannel)"
```

Expected: prints `OK <class 'wfcllm.watermark.orchestrator.WatermarkGenerator'> <class 'wfcllm.watermark.semantic_channel.SemanticChannel'>` with no errors.

- [ ] **Step 4: No commit** (verification only; commits live on Tasks 1–5).

---

## Self-Review

This section is a pre-flight checklist run by the plan author (now) against the spec, before handoff.

**Spec coverage:**

- §3 target tree lines 82-83 (`orchestrator.py` + `semantic_channel.py`) → Tasks 1-3 + 5.
- §10 step 8 ("generator.py 拆 orchestrator.py + semantic_channel.py: 最有风险，放最后") → entire plan; Task 7 + 8 are the risk-mitigation gates.
- §11 ("不重写 generator.py 内部算法（仅按职责切两片）") → Task 3 explicitly forbids algorithm changes; method bodies are verbatim moves with only `self.<attr>` → `self._orch.<attr>` rewrites inside `SemanticChannel`.
- §12 risk ("可能破坏 KV-cache snapshot 和 cascade 时序") → Task 7 isolates cascade + KV-cache test runs; baseline assertion in Task 8 catches any regression.
- Phase 6/7 shim convention → Task 4 (`wfcllm/watermark/generator.py` becomes deprecation shim) + Task 6 verifies callers still import from old path.

**Placeholder scan:** zero TBDs, zero "implement later"s, zero "handle edge cases" without code; every step has either a full file body, a precise edit instruction, or an exact verification command.

**Type consistency:** `SemanticChannel` method names omit the leading underscore (public on `SemanticChannel`, encapsulated by the orchestrator's proxies). The 10 proxies on `WatermarkGenerator` keep their old underscored names verbatim so test monkey-patches (`gen._verify_block = MagicMock(...)`, `gen._verifier.verify = MagicMock(...)`, `generator._entropy_profile = SimpleNamespace(...)`) still hit. Attribute ownership table:

| Attribute | Owner | Read/Written via |
|---|---|---|
| `_config` | orchestrator | both |
| `_entropy_est` | orchestrator | both |
| `_lsh_space` | orchestrator | orchestrator only (read by `_keying`/`_verifier` construction) |
| `_keying` | orchestrator | both |
| `_verifier` | orchestrator | both |
| `_entropy_profile` | orchestrator | written by `SemanticChannel.initialize_adaptive_gamma`, read by tests via `generator._entropy_profile` |
| `_gamma_schedule` | orchestrator | written by `SemanticChannel.initialize_adaptive_gamma`, read by both |
| `_semantic` | orchestrator | new; tests don't touch it |
| `_orch` | SemanticChannel | new; not visible from outside |

**Verbatim-move discipline:** the method bodies inside `SemanticChannel` are byte-equivalent to the originals minus exactly two transformations:
1. `self.<attr>` → `self._orch.<attr>` for the 7 attributes above plus `self._verify_block` and `self._semantic_channel_enabled()`.
2. method `self._resolve_gamma_for_entropy_units(...)` calls (currently from `_verify_block` and `_resolve_gamma_for_block_text`) become `self.resolve_gamma_for_entropy_units(...)` since they're now sibling methods on the same `SemanticChannel`.

No other edits — no formatting changes, no logic tweaks, no docstring rewrites, no error-handling additions. This is the strictest verbatim discipline of the refactor.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-14-repo-refactor-phase8-generator-split.md`.

Two execution options:

1. **Subagent-Driven** (recommended given the high-risk flag in spec §12) — fresh subagent per task with two-stage review; the spec-compliance reviewer catches any sneaky algorithm change, the code-quality reviewer catches any non-verbatim drift inside `SemanticChannel` method bodies.
2. **Inline Execution** — batch the tasks in this session.

The autonomous loop should default to (1) since this is the riskiest of the 10 refactor phases.
