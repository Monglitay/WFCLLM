# Phase 5: token_channel 三分层 + pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `wfcllm/watermark/token_channel/` into `core/` + `training/` + `runtime/` three-layer structure (verbatim moves + deprecation shims), then add a thin `wfcllm/pretrain/` orchestrator that exposes the new `pretrain` phase with `--stages encoder|lexical` selectability backed by the existing `run_encoder` / `run_token_channel_train` runners.

**Architecture:** Pure code-move plus thin orchestration, mirroring Phases 2-4. Each `token_channel` source file moves to its responsibility-layer location; the original file becomes a deprecation shim re-exporting public symbols. The new `pretrain` phase is a dispatcher that calls back into the existing single-phase runners — no algorithm changes, no state-schema rewrite, no model-weight format changes. State_dict compatibility is preserved because `core/model.py` is a byte-for-byte copy of `token_channel/model.py` (class names + `nn.Module` attribute paths unchanged).

**Tech Stack:** Python 3.10+, pytest, argparse, dataclasses, no new external deps.

**Spec reference:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §4.1 + §10 step 5.

**Out of scope (explicitly NOT in this phase):**

- `run_state.json` schema migration to nested `pretrain.encoder`/`pretrain.lexical` keys (spec §8.3) — `pretrain` phase still records under flat `encoder` / `token-channel-train` keys to keep `run_watermark` / `run_extract` `state.is_done("encoder")` checks working without modification. A later phase can introduce nested-key access + auto-migration.
- `config_resolver` legacy schema bridge merging top-level `encoder` / `token_channel_train` into `pretrain.encoder` / `pretrain.lexical` (spec §7 / §8.3) — defer.
- Cleaning up the existing `encoder` / `token-channel-train` phase registrations (spec §10 step 10 shim cleanup) — they remain registered.
- Modifying `wfcllm/encoder/` package layout — out of scope; only `wfcllm/watermark/token_channel/` is restructured.
- `experiment/` and `tools/` directories — not touched per project rules.

---

## File Structure

**New canonical files (moves):**

| Old path | New canonical path | Notes |
|---|---|---|
| `wfcllm/watermark/token_channel/config.py` | `wfcllm/watermark/token_channel/core/config.py` | verbatim |
| `wfcllm/watermark/token_channel/protocol.py` | `wfcllm/watermark/token_channel/core/protocol.py` | verbatim |
| `wfcllm/watermark/token_channel/features.py` | `wfcllm/watermark/token_channel/core/features.py` | verbatim |
| `wfcllm/watermark/token_channel/model.py` | `wfcllm/watermark/token_channel/core/model.py` | verbatim; rewrite intra-package `from wfcllm.watermark.token_channel.features import …` → `from wfcllm.watermark.token_channel.core.features import …` |
| `wfcllm/watermark/token_channel/runtime.py` | `wfcllm/watermark/token_channel/runtime/injector.py` | verbatim; rewrite intra-package imports to `core/*` |
| `wfcllm/watermark/token_channel/train.py` | `wfcllm/watermark/token_channel/training/trainer.py` | verbatim; rewrite intra-package imports |
| `wfcllm/watermark/token_channel/train_corpus.py` | `wfcllm/watermark/token_channel/training/corpus.py` | verbatim; rewrite intra-package imports |
| `wfcllm/watermark/token_channel/train_corpus_streaming.py` | `wfcllm/watermark/token_channel/training/corpus_streaming.py` | verbatim; rewrite intra-package imports |
| `wfcllm/watermark/token_channel/teacher.py` | `wfcllm/watermark/token_channel/training/teacher.py` | verbatim; rewrite intra-package imports if any |
| `wfcllm/watermark/token_channel/train_workflow.py` | `wfcllm/watermark/token_channel/training/workflow.py` | verbatim; rewrite intra-package imports |

**New scaffolding files:**

- `wfcllm/watermark/token_channel/core/__init__.py` — empty (created Task 1).
- `wfcllm/watermark/token_channel/training/__init__.py` — empty (created Task 1).
- `wfcllm/watermark/token_channel/runtime/__init__.py` — package-level re-export of `injector.py` symbols (created Task 6, atomically with old `runtime.py` deletion to avoid Python module/package shadowing).

**New `pretrain/` package:**

- `wfcllm/pretrain/__init__.py` — re-exports `PretrainConfig`, `PretrainPipeline`, `run_pretrain`.
- `wfcllm/pretrain/config.py` — `PretrainConfig` dataclass (`stages: list[str]`).
- `wfcllm/pretrain/pipeline.py` — `PretrainPipeline` class with `run(args, state) -> int`.
- `wfcllm/pretrain/runner.py` — `run_pretrain(args, state) -> int` CLI runner.

**New tests:**

- `tests/integration/test_token_channel_layout_shims.py` — old import paths still work + emit `DeprecationWarning`; new canonical paths importable; symbol identity (old.X is new.X) for representative symbols.
- `tests/integration/test_pretrain_phase.py` — `--stages encoder` runs only encoder; `--stages lexical` fail-fast when encoder not done; `--stages encoder lexical` runs both; default (no `--stages`) runs both.

**Modified existing files:**

- `wfcllm/watermark/token_channel/__init__.py` — switch each `from wfcllm.watermark.token_channel.<old> import …` to the canonical `core/*`, `training/*`, or `runtime/*` path.
- The 10 files listed above — each replaced with a deprecation shim.
- `wfcllm/cli/runners.py` — switch the 4 internal token_channel imports (already enumerated below in Task 12) to canonical paths; add `run_pretrain` to the `runners` dispatch dict.
- `wfcllm/cli/config_resolver.py:103` — switch import to `core/config`.
- `wfcllm/cli/entry.py` — register `pretrain` phase + import `run_pretrain`.
- `wfcllm/cli/arguments.py` — add `--stages` argument under the `pretrain` argparse group.
- `wfcllm/extract/config.py:9`, `wfcllm/extract/token_channel.py:12-17`, `wfcllm/extract/detector.py:18-19`, `wfcllm/extract/hypothesis.py:12`, `wfcllm/watermark/generator.py:29-32`, `wfcllm/watermark/config.py:7` — switch each to canonical paths.
- `wfcllm/orchestration/state.py:9-10` — append `"pretrain"` to `OPTIONAL_PHASES`.

**Unchanged (do NOT touch):**

- All test files under `tests/watermark/token_channel/`, `tests/extract/`, `tests/test_run.py`, `tests/test_run_config.py` — they keep importing from the old paths to exercise the shims (same convention as Phase 4).
- `wfcllm/encoder/` package — out of scope.
- `experiment/`, `tools/` — out of scope.

---

## Conventions

**Deprecation shim template** (matches Phase 2/3/4 pattern):

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.<NEW> instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.<OLD> is deprecated; "
    "use wfcllm.watermark.token_channel.<NEW>",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.<NEW> import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.<NEW> import (  # noqa: E402, F401
    <SYMBOLS_USED_BY_TESTS_OR_MONKEYPATCHES>,
)
```

The `from … import *` covers ordinary consumers; the explicit named imports preserve attribute access for `monkeypatch.setattr("wfcllm.watermark.token_channel.<OLD>.<SYMBOL>", …)` and `import … as train_module` style tests (e.g., `tests/watermark/token_channel/test_model.py:322`).

**State-dict compatibility:** `core/model.py` is byte-for-byte identical to `token_channel/model.py` with the single allowed change being intra-package import statements at the top. `nn.Module` subclass names, attribute names, and instantiation order are untouched, so existing `.pt` checkpoints load unchanged.

**Test command:** `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest <path> -v`

**Commit message format:** `<type>: <description>` per `CLAUDE.md`. Each task ends with one commit.

**Pre-existing baseline failures (NOT introduced by Phase 5 — exclude from "no new failures" criterion):**

- `tests/watermark/test_pipeline.py` — 6 `MagicMock` mock-pollution failures around `_config.min_blocks`.
- `tests/extract/test_joint_detection.py` — 1 detection failure.
- `tests/test_run_config.py` — 4 `SimpleNamespace` fixture missing `min_blocks` failures at `runners.py:403`.

These existed on `main` before Phase 5 began. If your verification step shows exactly these failures and nothing else, treat the task as passing.

---

## Task 1: Create empty `core/` + `training/` subpackages + first failing shim test

**Files:**
- Create: `wfcllm/watermark/token_channel/core/__init__.py`
- Create: `wfcllm/watermark/token_channel/training/__init__.py`
- Create: `tests/integration/test_token_channel_layout_shims.py`

**Note: do NOT create `wfcllm/watermark/token_channel/runtime/__init__.py` yet.** Python prefers the directory-package over the same-named `runtime.py` module file when both exist; if we created `runtime/__init__.py` here, every consumer that does `from wfcllm.watermark.token_channel.runtime import X` would silently start resolving to the empty package and fail. The `runtime/` package is created in Task 6, atomically with the deletion of `runtime.py`.

- [ ] **Step 1: Create the two empty package `__init__.py` files**

Write `wfcllm/watermark/token_channel/core/__init__.py`:

```python
"""Token-channel core: shared between training and runtime."""
```

Write `wfcllm/watermark/token_channel/training/__init__.py`:

```python
"""Token-channel training: only used during pretrain phase."""
```

- [ ] **Step 2: Write the first failing shim test (canonical core path importable)**

Write `tests/integration/test_token_channel_layout_shims.py`:

```python
"""Tests for wfcllm/watermark/token_channel/* three-layer migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import warnings


# --- core: new path works ---

def test_core_config_new_path_importable():
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
    assert callable(TokenChannelConfig)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py::test_core_config_new_path_importable -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.watermark.token_channel.core.config'`.

- [ ] **Step 4: Commit the package skeleton + failing test**

```bash
git add wfcllm/watermark/token_channel/core/__init__.py \
        wfcllm/watermark/token_channel/training/__init__.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "test: add failing shim test for token_channel three-layer split"
```

---

## Task 2: Move `config.py` → `core/config.py` with shim

**Files:**
- Create: `wfcllm/watermark/token_channel/core/config.py`
- Modify: `wfcllm/watermark/token_channel/config.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `config.py` body to `core/config.py` verbatim**

Run:

```bash
cp wfcllm/watermark/token_channel/config.py wfcllm/watermark/token_channel/core/config.py
```

`config.py` has no intra-package imports (only stdlib + dataclasses), so no rewrite needed. Verify with:

```bash
grep -n "from wfcllm.watermark.token_channel" wfcllm/watermark/token_channel/core/config.py
```

Expected: no output.

- [ ] **Step 2: Replace old `config.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/config.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.core.config instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.config is deprecated; "
    "use wfcllm.watermark.token_channel.core.config",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.config import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.config import (  # noqa: E402, F401
    TokenChannelConfig,
    TokenChannelJointConfig,
)
```

- [ ] **Step 3: Add the canonical-path test for config + old-path-emits-warning test**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_core_config_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.config", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.config")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "core.config" in str(w.message)
            for w in caught
        )


def test_core_config_symbol_identity():
    from wfcllm.watermark.token_channel.config import TokenChannelConfig as old_cls
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig as new_cls
    assert old_cls is new_cls
```

- [ ] **Step 4: Run the three config tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "config"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/token_channel/core/config.py \
        wfcllm/watermark/token_channel/config.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/config.py to core/ + shim"
```

---

## Task 3: Move `protocol.py` → `core/protocol.py` with shim

**Files:**
- Create: `wfcllm/watermark/token_channel/core/protocol.py`
- Modify: `wfcllm/watermark/token_channel/protocol.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `protocol.py` body to `core/protocol.py` verbatim**

Run:

```bash
cp wfcllm/watermark/token_channel/protocol.py wfcllm/watermark/token_channel/core/protocol.py
```

`protocol.py` has no intra-package imports. Verify:

```bash
grep -n "from wfcllm.watermark.token_channel" wfcllm/watermark/token_channel/core/protocol.py
```

Expected: no output.

- [ ] **Step 2: Replace old `protocol.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/protocol.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.core.protocol instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.protocol is deprecated; "
    "use wfcllm.watermark.token_channel.core.protocol",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.protocol import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.protocol import (  # noqa: E402, F401
    PartitionResult,
    build_partition,
    make_prefix_key,
    make_scored_token_key,
)
```

- [ ] **Step 3: Append protocol tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_core_protocol_new_path_importable():
    from wfcllm.watermark.token_channel.core.protocol import build_partition
    assert callable(build_partition)


def test_core_protocol_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.protocol", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.protocol")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "core.protocol" in str(w.message)
            for w in caught
        )


def test_core_protocol_symbol_identity():
    from wfcllm.watermark.token_channel.protocol import build_partition as old_fn
    from wfcllm.watermark.token_channel.core.protocol import build_partition as new_fn
    assert old_fn is new_fn
```

- [ ] **Step 4: Run protocol tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "protocol"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/token_channel/core/protocol.py \
        wfcllm/watermark/token_channel/protocol.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/protocol.py to core/ + shim"
```

---

## Task 4: Move `features.py` → `core/features.py` with shim

**Files:**
- Create: `wfcllm/watermark/token_channel/core/features.py`
- Modify: `wfcllm/watermark/token_channel/features.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `features.py` body to `core/features.py` verbatim**

Run:

```bash
cp wfcllm/watermark/token_channel/features.py wfcllm/watermark/token_channel/core/features.py
```

`features.py` has no intra-package imports. Verify:

```bash
grep -n "from wfcllm.watermark.token_channel" wfcllm/watermark/token_channel/core/features.py
```

Expected: no output.

- [ ] **Step 2: Replace old `features.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/features.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.core.features instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.features is deprecated; "
    "use wfcllm.watermark.token_channel.core.features",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.features import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.features import (  # noqa: E402, F401
    FEATURE_VERSION,
    ExcludedSpan,
    TokenChannelFeatureContext,
    TokenChannelFeatures,
    build_structure_masks,
    build_token_channel_features,
    build_token_channel_features_from_context,
    collect_excluded_token_spans,
    is_structure_safe_span,
    prepare_token_channel_feature_context,
)
```

The explicit named-import block covers `tests/watermark/token_channel/test_features.py` and `tests/watermark/token_channel/test_train_corpus.py` consumers.

- [ ] **Step 3: Append features tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_core_features_new_path_importable():
    from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
    assert callable(TokenChannelFeatures)


def test_core_features_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.features", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.features")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "core.features" in str(w.message)
            for w in caught
        )


def test_core_features_symbol_identity():
    from wfcllm.watermark.token_channel.features import TokenChannelFeatures as old_cls
    from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures as new_cls
    assert old_cls is new_cls
```

- [ ] **Step 4: Run features tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "features"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/token_channel/core/features.py \
        wfcllm/watermark/token_channel/features.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/features.py to core/ + shim"
```

---

## Task 5: Move `model.py` → `core/model.py` with import rewrite + shim

**Files:**
- Create: `wfcllm/watermark/token_channel/core/model.py`
- Modify: `wfcllm/watermark/token_channel/model.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `model.py` body to `core/model.py`**

Run:

```bash
cp wfcllm/watermark/token_channel/model.py wfcllm/watermark/token_channel/core/model.py
```

- [ ] **Step 2: Rewrite intra-package imports in `core/model.py`**

In `wfcllm/watermark/token_channel/core/model.py`, find lines 16-17:

```python
from wfcllm.watermark.token_channel.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
```

Verify there are no remaining old-path imports in `core/model.py`:

```bash
grep -n "from wfcllm.watermark.token_channel\." wfcllm/watermark/token_channel/core/model.py | grep -v "core\."
```

Expected: no output.

- [ ] **Step 3: Replace old `model.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/model.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.core.model instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.model is deprecated; "
    "use wfcllm.watermark.token_channel.core.model",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.core.model import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.core.model import (  # noqa: E402, F401
    TokenChannelArtifact,
    TokenChannelArtifactMetadata,
    TokenChannelCheckpointExport,
    TokenChannelCompatibility,
    TokenChannelLossWeights,
    TokenChannelModel,
    TokenChannelModelOutput,
    check_token_channel_compatibility,
    export_token_channel_checkpoint,
    load_token_channel_artifact,
    load_token_channel_artifact_metadata,
    load_training_state,
    require_token_channel_compatibility,
    save_token_channel_artifact_metadata,
)
```

Explicit named imports cover the consumers in `tests/watermark/token_channel/test_model.py` and `tests/watermark/token_channel/test_runtime.py`.

- [ ] **Step 4: Append model tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_core_model_new_path_importable():
    from wfcllm.watermark.token_channel.core.model import TokenChannelModel
    assert callable(TokenChannelModel)


def test_core_model_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.model", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.model")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "core.model" in str(w.message)
            for w in caught
        )


def test_core_model_symbol_identity():
    from wfcllm.watermark.token_channel.model import TokenChannelModel as old_cls
    from wfcllm.watermark.token_channel.core.model import TokenChannelModel as new_cls
    assert old_cls is new_cls
```

- [ ] **Step 5: Run model tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "model"`
Expected: 3 passed (the three new model_* tests).

- [ ] **Step 6: Run existing token_channel model tests to verify nothing broke**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_model.py -v`
Expected: all pass (these import from the OLD path; the shim must still serve them).

- [ ] **Step 7: Commit**

```bash
git add wfcllm/watermark/token_channel/core/model.py \
        wfcllm/watermark/token_channel/model.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/model.py to core/ + shim"
```

---

## Task 6: Move `runtime.py` → `runtime/injector.py` with import rewrite + shim

**Files:**
- Create: `wfcllm/watermark/token_channel/runtime/__init__.py` (NEW — created here, not in Task 1)
- Create: `wfcllm/watermark/token_channel/runtime/injector.py`
- Delete: `wfcllm/watermark/token_channel/runtime.py`
- Modify: `tests/integration/test_token_channel_layout_shims.py`

**Naming conflict:** Python picks the directory-package over a same-named `.py` module file when both exist. `runtime.py` and `runtime/` cannot coexist usefully — the package shadows the module without warning. So we MUST treat directory creation + module deletion as one atomic operation in this task: copy `runtime.py` body into `runtime/injector.py`, write the new `runtime/__init__.py` re-export, then delete the old `runtime.py`. This is why we deferred creating `runtime/__init__.py` from Task 1 to here.

The `runtime/__init__.py` doubles as the shim (re-exporting `TokenChannelRuntime` etc. at the package level for old callers `from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime`).

- [ ] **Step 1: Create the `runtime/` directory and copy body to `runtime/injector.py`**

Run:

```bash
mkdir -p wfcllm/watermark/token_channel/runtime
cp wfcllm/watermark/token_channel/runtime.py wfcllm/watermark/token_channel/runtime/injector.py
```

After this `cp` finishes, `runtime.py` and `runtime/injector.py` both exist but `runtime/__init__.py` does NOT yet — so Python still resolves `wfcllm.watermark.token_channel.runtime` to the old `runtime.py` module. Code that imports from `runtime.injector` will fail with ModuleNotFoundError until Step 3 creates `runtime/__init__.py`. Don't run any tests between this step and Step 4.

- [ ] **Step 2: Rewrite intra-package imports in `runtime/injector.py`**

In `wfcllm/watermark/token_channel/runtime/injector.py`, find lines 10-17:

```python
from wfcllm.watermark.token_channel.config import TokenChannelConfig
from wfcllm.watermark.token_channel.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.model import TokenChannelArtifactMetadata
from wfcllm.watermark.token_channel.model import TokenChannelModel
from wfcllm.watermark.token_channel.model import require_token_channel_compatibility
from wfcllm.watermark.token_channel.protocol import PartitionResult
from wfcllm.watermark.token_channel.protocol import build_partition
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
from wfcllm.watermark.token_channel.core.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.core.model import TokenChannelArtifactMetadata
from wfcllm.watermark.token_channel.core.model import TokenChannelModel
from wfcllm.watermark.token_channel.core.model import require_token_channel_compatibility
from wfcllm.watermark.token_channel.core.protocol import PartitionResult
from wfcllm.watermark.token_channel.core.protocol import build_partition
```

Verify:

```bash
grep -n "from wfcllm.watermark.token_channel\." wfcllm/watermark/token_channel/runtime/injector.py | grep -v "core\.\|runtime\."
```

Expected: no output.

- [ ] **Step 3: Create `runtime/__init__.py` as the package-level re-export shim**

Write `wfcllm/watermark/token_channel/runtime/__init__.py`:

```python
"""Token-channel runtime layer.

Public canonical path: wfcllm.watermark.token_channel.runtime.injector
Importing from this package directly (e.g. `from … import TokenChannelRuntime`) is
preserved for backward compatibility; new code should import from the explicit submodule.
"""
from __future__ import annotations

from wfcllm.watermark.token_channel.runtime.injector import *  # noqa: F401, F403
from wfcllm.watermark.token_channel.runtime.injector import (  # noqa: F401
    TokenChannelDecision,
    TokenChannelRuntime,
)
```

This dual-purpose `__init__.py`:
- exposes the public symbols at `wfcllm.watermark.token_channel.runtime.X` (what existing code uses)
- still allows the new explicit import `wfcllm.watermark.token_channel.runtime.injector.X`

We do NOT emit a `DeprecationWarning` from `runtime/__init__.py` because Python imports `__init__.py` whenever ANY consumer touches the `runtime` subpackage (including the canonical `runtime.injector` import path). Emitting a warning here would fire on every import — including correct, canonical ones.

- [ ] **Step 4: Delete the old `runtime.py` module file**

Run:

```bash
rm wfcllm/watermark/token_channel/runtime.py
```

After this step, Python resolves `wfcllm.watermark.token_channel.runtime` to the new package (the directory). The old module file is gone.

- [ ] **Step 5: Append runtime tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_runtime_injector_new_path_importable():
    from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime
    assert callable(TokenChannelRuntime)


def test_runtime_package_level_reexport():
    """Old code does `from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime`.

    Note: no DeprecationWarning is emitted because runtime/ is now a real subpackage,
    and emitting a warning from runtime/__init__.py would fire on every canonical import too.
    """
    from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime as pkg_cls
    from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime as mod_cls
    assert pkg_cls is mod_cls
```

- [ ] **Step 6: Run runtime tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "runtime"`
Expected: 2 passed.

- [ ] **Step 7: Run existing token_channel runtime tests to verify nothing broke**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_runtime.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add wfcllm/watermark/token_channel/runtime/injector.py \
        wfcllm/watermark/token_channel/runtime/__init__.py \
        tests/integration/test_token_channel_layout_shims.py
git rm wfcllm/watermark/token_channel/runtime.py
git commit -m "refactor: move token_channel/runtime.py to runtime/injector.py + package re-export"
```

---

## Task 7: Move `teacher.py` → `training/teacher.py` with shim

**Files:**
- Create: `wfcllm/watermark/token_channel/training/teacher.py`
- Modify: `wfcllm/watermark/token_channel/teacher.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `teacher.py` body to `training/teacher.py`**

Run:

```bash
cp wfcllm/watermark/token_channel/teacher.py wfcllm/watermark/token_channel/training/teacher.py
```

`teacher.py` has no intra-`token_channel` imports (only `import torch` + stdlib). Verify:

```bash
grep -n "from wfcllm.watermark.token_channel\." wfcllm/watermark/token_channel/training/teacher.py
```

Expected: no output.

- [ ] **Step 2: Replace old `teacher.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/teacher.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.training.teacher instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.teacher is deprecated; "
    "use wfcllm.watermark.token_channel.training.teacher",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.teacher import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.teacher import (  # noqa: E402, F401
    batch_extract_teacher_rows,
    extract_teacher_rows,
    load_teacher_cache,
    save_teacher_cache,
    _batch_forward_all_positions,
    _compute_dynamic_batch_size,
    _extract_single_text_all_positions,
    _get_available_memory_gb,
    _get_vocab_size,
    _group_texts_by_length,
)
```

The trailing `_*` private symbols are explicitly re-exported because `tests/watermark/token_channel/test_teacher_batch.py` directly imports them (lines 59, 71, 83, 108, 120, 148, 171, 211, 227, 245, 256, 267, 316, 360, 388, 417, 430, 463, 487, 497, 509, 535, 550, 567, 577, 597, 616, 657, 681, 698, 733).

- [ ] **Step 3: Append teacher tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_training_teacher_new_path_importable():
    from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows
    assert callable(extract_teacher_rows)


def test_training_teacher_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.teacher", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.teacher")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "training.teacher" in str(w.message)
            for w in caught
        )


def test_training_teacher_symbol_identity():
    from wfcllm.watermark.token_channel.teacher import extract_teacher_rows as old_fn
    from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows as new_fn
    assert old_fn is new_fn
```

- [ ] **Step 4: Run teacher tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "teacher"`
Expected: 3 passed.

- [ ] **Step 5: Run existing teacher tests to verify the shim still serves them**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/token_channel/training/teacher.py \
        wfcllm/watermark/token_channel/teacher.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/teacher.py to training/ + shim"
```

---

## Task 8: Move `train.py` → `training/trainer.py` with import rewrite + shim

**Files:**
- Create: `wfcllm/watermark/token_channel/training/trainer.py`
- Modify: `wfcllm/watermark/token_channel/train.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `train.py` body to `training/trainer.py`**

Run:

```bash
cp wfcllm/watermark/token_channel/train.py wfcllm/watermark/token_channel/training/trainer.py
```

- [ ] **Step 2: Rewrite intra-package imports in `training/trainer.py`**

In `wfcllm/watermark/token_channel/training/trainer.py`, find lines 17-21:

```python
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.model import TokenChannelLossWeights
from wfcllm.watermark.token_channel.model import TokenChannelModel
from wfcllm.watermark.token_channel.model import export_token_channel_checkpoint
from wfcllm.watermark.token_channel.teacher import load_teacher_cache
from wfcllm.watermark.token_channel.train_corpus import load_training_cache
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.core.model import TokenChannelLossWeights
from wfcllm.watermark.token_channel.core.model import TokenChannelModel
from wfcllm.watermark.token_channel.core.model import export_token_channel_checkpoint
from wfcllm.watermark.token_channel.training.teacher import load_teacher_cache
from wfcllm.watermark.token_channel.training.corpus import load_training_cache
```

Note: `corpus.py` doesn't exist yet — it will be created in Task 9. The import will fail until then. We accept this transient breakage because the shim re-exports the symbol.

Verify the new file has no other old-path imports:

```bash
grep -n "from wfcllm.watermark.token_channel\." wfcllm/watermark/token_channel/training/trainer.py | grep -v "core\.\|training\."
```

Expected: no output.

- [ ] **Step 3: Replace old `train.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/train.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.training.trainer instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train is deprecated; "
    "use wfcllm.watermark.token_channel.training.trainer",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.trainer import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.trainer import (  # noqa: E402, F401
    TokenChannelEpochMetrics,
    TokenChannelTrainingEvidence,
    build_token_channel_batch,
    build_training_evidence,
    evaluate_batch_loss,
    main,
    run_training_step,
    save_token_channel_training_artifacts,
    save_training_evidence,
    train_one_epoch,
)
```

- [ ] **Step 4: Append trainer tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_training_trainer_new_path_importable():
    from wfcllm.watermark.token_channel.training.trainer import train_one_epoch
    assert callable(train_one_epoch)


def test_training_trainer_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.train", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.train")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "training.trainer" in str(w.message)
            for w in caught
        )


def test_training_trainer_symbol_identity():
    from wfcllm.watermark.token_channel.train import train_one_epoch as old_fn
    from wfcllm.watermark.token_channel.training.trainer import train_one_epoch as new_fn
    assert old_fn is new_fn
```

- [ ] **Step 5: DO NOT run trainer tests yet — they depend on `training.corpus` (Task 9)**

Skip verification at this task. The shim is wired but `training.corpus` does not exist. Expected behavior at this point: importing the shim FAILS with `ModuleNotFoundError: No module named 'wfcllm.watermark.token_channel.training.corpus'`. The next task fixes this.

- [ ] **Step 6: Commit anyway (red intermediate state)**

```bash
git add wfcllm/watermark/token_channel/training/trainer.py \
        wfcllm/watermark/token_channel/train.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/train.py to training/trainer.py + shim (corpus dep pending)"
```

---

## Task 9: Move `train_corpus.py` → `training/corpus.py` with import rewrite + shim

**Files:**
- Create: `wfcllm/watermark/token_channel/training/corpus.py`
- Modify: `wfcllm/watermark/token_channel/train_corpus.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `train_corpus.py` body to `training/corpus.py`**

Run:

```bash
cp wfcllm/watermark/token_channel/train_corpus.py wfcllm/watermark/token_channel/training/corpus.py
```

- [ ] **Step 2: Rewrite intra-package imports in `training/corpus.py`**

In `wfcllm/watermark/token_channel/training/corpus.py`, find lines 17-20:

```python
from wfcllm.watermark.token_channel.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.teacher import batch_extract_teacher_rows
from wfcllm.watermark.token_channel.teacher import extract_teacher_rows
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.core.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.training.teacher import batch_extract_teacher_rows
from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows
```

Verify:

```bash
grep -n "from wfcllm.watermark.token_channel\." wfcllm/watermark/token_channel/training/corpus.py | grep -v "core\.\|training\."
```

Expected: no output.

- [ ] **Step 3: Replace old `train_corpus.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/train_corpus.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.training.corpus instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train_corpus is deprecated; "
    "use wfcllm.watermark.token_channel.training.corpus",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.corpus import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.corpus import (  # noqa: E402, F401
    TRAINING_CACHE_SCHEMA_VERSION,
    build_augmented_variants,
    build_training_rows,
    load_training_cache,
    save_training_cache,
    save_training_cache_streaming,
)
```

- [ ] **Step 4: Append corpus tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_training_corpus_new_path_importable():
    from wfcllm.watermark.token_channel.training.corpus import build_training_rows
    assert callable(build_training_rows)


def test_training_corpus_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.train_corpus", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.train_corpus")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "training.corpus" in str(w.message)
            for w in caught
        )


def test_training_corpus_symbol_identity():
    from wfcllm.watermark.token_channel.train_corpus import build_training_rows as old_fn
    from wfcllm.watermark.token_channel.training.corpus import build_training_rows as new_fn
    assert old_fn is new_fn
```

- [ ] **Step 5: Run corpus tests + previously-deferred trainer tests to verify both pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "corpus or trainer"`
Expected: 6 passed (3 corpus + 3 trainer; the trainer tests were red after Task 8 and now go green).

- [ ] **Step 6: Run existing train_corpus tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_corpus.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add wfcllm/watermark/token_channel/training/corpus.py \
        wfcllm/watermark/token_channel/train_corpus.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/train_corpus.py to training/corpus.py + shim"
```

---

## Task 10: Move `train_corpus_streaming.py` → `training/corpus_streaming.py` with import rewrite + shim

**Files:**
- Create: `wfcllm/watermark/token_channel/training/corpus_streaming.py`
- Modify: `wfcllm/watermark/token_channel/train_corpus_streaming.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `train_corpus_streaming.py` body to `training/corpus_streaming.py`**

Run:

```bash
cp wfcllm/watermark/token_channel/train_corpus_streaming.py wfcllm/watermark/token_channel/training/corpus_streaming.py
```

- [ ] **Step 2: Rewrite intra-package imports in `training/corpus_streaming.py`**

In `wfcllm/watermark/token_channel/training/corpus_streaming.py`, find line 9:

```python
from wfcllm.watermark.token_channel.train_corpus import TRAINING_CACHE_SCHEMA_VERSION
```

Replace with:

```python
from wfcllm.watermark.token_channel.training.corpus import TRAINING_CACHE_SCHEMA_VERSION
```

Verify:

```bash
grep -n "from wfcllm.watermark.token_channel\." wfcllm/watermark/token_channel/training/corpus_streaming.py | grep -v "training\."
```

Expected: no output.

- [ ] **Step 3: Replace old `train_corpus_streaming.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/train_corpus_streaming.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.training.corpus_streaming instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train_corpus_streaming is deprecated; "
    "use wfcllm.watermark.token_channel.training.corpus_streaming",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.corpus_streaming import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.corpus_streaming import (  # noqa: E402, F401
    count_training_cache_rows,
    load_rows_by_indices,
    split_training_cache_streaming,
    stream_training_cache,
)
```

- [ ] **Step 4: Append corpus_streaming tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_training_corpus_streaming_new_path_importable():
    from wfcllm.watermark.token_channel.training.corpus_streaming import stream_training_cache
    assert callable(stream_training_cache)


def test_training_corpus_streaming_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.train_corpus_streaming", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.train_corpus_streaming")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "training.corpus_streaming" in str(w.message)
            for w in caught
        )


def test_training_corpus_streaming_symbol_identity():
    from wfcllm.watermark.token_channel.train_corpus_streaming import stream_training_cache as old_fn
    from wfcllm.watermark.token_channel.training.corpus_streaming import stream_training_cache as new_fn
    assert old_fn is new_fn
```

- [ ] **Step 5: Run corpus_streaming tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "corpus_streaming"`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/token_channel/training/corpus_streaming.py \
        wfcllm/watermark/token_channel/train_corpus_streaming.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/train_corpus_streaming.py to training/corpus_streaming.py + shim"
```

---

## Task 11: Move `train_workflow.py` → `training/workflow.py` with import rewrite + shim

**Files:**
- Create: `wfcllm/watermark/token_channel/training/workflow.py`
- Modify: `wfcllm/watermark/token_channel/train_workflow.py` (replace with shim)
- Modify: `tests/integration/test_token_channel_layout_shims.py`

- [ ] **Step 1: Copy `train_workflow.py` body to `training/workflow.py`**

Run:

```bash
cp wfcllm/watermark/token_channel/train_workflow.py wfcllm/watermark/token_channel/training/workflow.py
```

- [ ] **Step 2: Rewrite intra-package imports in `training/workflow.py`**

In `wfcllm/watermark/token_channel/training/workflow.py`, find lines 17-32:

```python
from wfcllm.watermark.token_channel.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.model import TokenChannelModel
from wfcllm.watermark.token_channel.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.model import load_training_state
from wfcllm.watermark.token_channel.model import require_token_channel_compatibility
from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
from wfcllm.watermark.token_channel.train import build_token_channel_batch
from wfcllm.watermark.token_channel.train import build_training_evidence
from wfcllm.watermark.token_channel.train import save_token_channel_training_artifacts
from wfcllm.watermark.token_channel.train import train_one_epoch
from wfcllm.watermark.token_channel.train_corpus import build_training_rows
from wfcllm.watermark.token_channel.train_corpus import load_training_cache
from wfcllm.watermark.token_channel.train_corpus import save_training_cache_streaming
from wfcllm.watermark.token_channel.train_corpus_streaming import count_training_cache_rows
from wfcllm.watermark.token_channel.train_corpus_streaming import load_rows_by_indices
from wfcllm.watermark.token_channel.train_corpus_streaming import split_training_cache_streaming
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.core.model import TokenChannelModel
from wfcllm.watermark.token_channel.core.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.core.model import load_training_state
from wfcllm.watermark.token_channel.core.model import require_token_channel_compatibility
from wfcllm.watermark.token_channel.training.trainer import TokenChannelEpochMetrics
from wfcllm.watermark.token_channel.training.trainer import build_token_channel_batch
from wfcllm.watermark.token_channel.training.trainer import build_training_evidence
from wfcllm.watermark.token_channel.training.trainer import save_token_channel_training_artifacts
from wfcllm.watermark.token_channel.training.trainer import train_one_epoch
from wfcllm.watermark.token_channel.training.corpus import build_training_rows
from wfcllm.watermark.token_channel.training.corpus import load_training_cache
from wfcllm.watermark.token_channel.training.corpus import save_training_cache_streaming
from wfcllm.watermark.token_channel.training.corpus_streaming import count_training_cache_rows
from wfcllm.watermark.token_channel.training.corpus_streaming import load_rows_by_indices
from wfcllm.watermark.token_channel.training.corpus_streaming import split_training_cache_streaming
```

Also rewrite the deferred imports inside function bodies. Find these lines:

```python
# Inside _build_training_artifact_metadata-area (around original line 404):
from wfcllm.watermark.token_channel.model import export_token_channel_checkpoint

# Inside _build_batches_streaming (around original line 565):
from wfcllm.watermark.token_channel.train import build_token_channel_batch

# Inside _load_sample_rows (around original line 597):
from wfcllm.watermark.token_channel.train_corpus_streaming import stream_training_cache
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.model import export_token_channel_checkpoint
from wfcllm.watermark.token_channel.training.trainer import build_token_channel_batch
from wfcllm.watermark.token_channel.training.corpus_streaming import stream_training_cache
```

Verify:

```bash
grep -n "from wfcllm.watermark.token_channel\." wfcllm/watermark/token_channel/training/workflow.py | grep -v "core\.\|training\."
```

Expected: no output.

- [ ] **Step 3: Replace old `train_workflow.py` with deprecation shim**

Write `wfcllm/watermark/token_channel/train_workflow.py`:

```python
"""Deprecated import path. Use wfcllm.watermark.token_channel.training.workflow instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.watermark.token_channel.train_workflow is deprecated; "
    "use wfcllm.watermark.token_channel.training.workflow",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.watermark.token_channel.training.workflow import *  # noqa: E402, F401, F403
from wfcllm.watermark.token_channel.training.workflow import (  # noqa: E402, F401
    TokenChannelTrainWorkflowConfig,
    TokenChannelTrainWorkflowSummary,
    format_token_channel_train_workflow_summary,
    normalize_reference_solution_rows,
    run_token_channel_train_workflow,
    split_training_rows,
)
```

- [ ] **Step 4: Append workflow tests to the shim test file**

Append to `tests/integration/test_token_channel_layout_shims.py`:

```python
def test_training_workflow_new_path_importable():
    from wfcllm.watermark.token_channel.training.workflow import run_token_channel_train_workflow
    assert callable(run_token_channel_train_workflow)


def test_training_workflow_old_path_emits_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.watermark.token_channel.train_workflow", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.watermark.token_channel.train_workflow")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "training.workflow" in str(w.message)
            for w in caught
        )


def test_training_workflow_symbol_identity():
    from wfcllm.watermark.token_channel.train_workflow import run_token_channel_train_workflow as old_fn
    from wfcllm.watermark.token_channel.training.workflow import run_token_channel_train_workflow as new_fn
    assert old_fn is new_fn
```

- [ ] **Step 5: Run workflow tests + existing train_workflow tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_layout_shims.py -v -k "workflow"`
Expected: 3 passed.

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py -v`
Expected: all pass (these import from the OLD path; the shim must serve them).

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/token_channel/training/workflow.py \
        wfcllm/watermark/token_channel/train_workflow.py \
        tests/integration/test_token_channel_layout_shims.py
git commit -m "refactor: move token_channel/train_workflow.py to training/workflow.py + shim"
```

---

## Task 12: Switch first-party callers to canonical paths + update package `__init__.py`

**Files:**
- Modify: `wfcllm/watermark/token_channel/__init__.py`
- Modify: `wfcllm/cli/config_resolver.py:103`
- Modify: `wfcllm/cli/runners.py` (lines 744, 747, 750)
- Modify: `wfcllm/extract/config.py:9`
- Modify: `wfcllm/extract/token_channel.py` (lines 12-17)
- Modify: `wfcllm/extract/detector.py` (lines 18-19)
- Modify: `wfcllm/extract/hypothesis.py:12`
- Modify: `wfcllm/watermark/generator.py` (lines 29-32)
- Modify: `wfcllm/watermark/config.py:7`

This task removes all first-party callers' imports of the old paths so the only consumers of the deprecation shims are external scripts and tests. Each rewrite is a 1-line substitution.

- [ ] **Step 1: Update `wfcllm/watermark/token_channel/__init__.py`**

Replace its body with:

```python
"""Token-channel model, training, and runtime helpers."""

from __future__ import annotations

from wfcllm.watermark.token_channel.core.model import TokenChannelArtifact
from wfcllm.watermark.token_channel.core.model import TokenChannelArtifactMetadata
from wfcllm.watermark.token_channel.core.model import TokenChannelCheckpointExport
from wfcllm.watermark.token_channel.core.model import TokenChannelLossWeights
from wfcllm.watermark.token_channel.core.model import TokenChannelModel
from wfcllm.watermark.token_channel.core.model import TokenChannelModelOutput
from wfcllm.watermark.token_channel.core.model import export_token_channel_checkpoint
from wfcllm.watermark.token_channel.training.trainer import TokenChannelEpochMetrics
from wfcllm.watermark.token_channel.training.trainer import TokenChannelTrainingEvidence
from wfcllm.watermark.token_channel.training.trainer import build_token_channel_batch
from wfcllm.watermark.token_channel.training.trainer import run_training_step
from wfcllm.watermark.token_channel.training.trainer import save_training_evidence
from wfcllm.watermark.token_channel.training.trainer import train_one_epoch

__all__ = [
    "TokenChannelArtifact",
    "TokenChannelArtifactMetadata",
    "TokenChannelCheckpointExport",
    "TokenChannelEpochMetrics",
    "TokenChannelLossWeights",
    "TokenChannelModel",
    "TokenChannelModelOutput",
    "TokenChannelTrainingEvidence",
    "build_token_channel_batch",
    "export_token_channel_checkpoint",
    "run_training_step",
    "save_training_evidence",
    "train_one_epoch",
]
```

- [ ] **Step 2: Update `wfcllm/cli/config_resolver.py:103`**

Find:

```python
    from wfcllm.watermark.token_channel.config import TokenChannelConfig
```

Replace with:

```python
    from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
```

- [ ] **Step 3: Update `wfcllm/cli/runners.py` (3 sites at lines 744, 747, 750)**

Find:

```python
    from wfcllm.watermark.token_channel.train_workflow import (
        TokenChannelTrainWorkflowConfig,
    )
    from wfcllm.watermark.token_channel.train_workflow import (
        format_token_channel_train_workflow_summary,
    )
    from wfcllm.watermark.token_channel.train_workflow import (
        run_token_channel_train_workflow,
    )
```

Replace with:

```python
    from wfcllm.watermark.token_channel.training.workflow import (
        TokenChannelTrainWorkflowConfig,
    )
    from wfcllm.watermark.token_channel.training.workflow import (
        format_token_channel_train_workflow_summary,
    )
    from wfcllm.watermark.token_channel.training.workflow import (
        run_token_channel_train_workflow,
    )
```

- [ ] **Step 4: Update `wfcllm/extract/config.py:9`**

Find:

```python
from wfcllm.watermark.token_channel.config import TokenChannelConfig
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
```

- [ ] **Step 5: Update `wfcllm/extract/token_channel.py` (lines 12-17)**

Find:

```python
from wfcllm.watermark.token_channel.config import TokenChannelConfig
from wfcllm.watermark.token_channel.features import TokenChannelFeatureContext
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.protocol import make_prefix_key
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatureContext
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.core.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.core.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.core.protocol import make_prefix_key
```

- [ ] **Step 6: Update `wfcllm/extract/detector.py` (lines 18-19)**

Find:

```python
from wfcllm.watermark.token_channel.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime
```

- [ ] **Step 7: Update `wfcllm/extract/hypothesis.py:12`**

Find:

```python
from wfcllm.watermark.token_channel.config import TokenChannelConfig
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
```

- [ ] **Step 8: Update `wfcllm/watermark/generator.py` (lines 29-32)**

Find:

```python
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.features import build_token_channel_features
from wfcllm.watermark.token_channel.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.runtime import TokenChannelRuntime
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.core.features import build_token_channel_features
from wfcllm.watermark.token_channel.core.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime
```

- [ ] **Step 9: Update `wfcllm/watermark/config.py:7`**

Find:

```python
from wfcllm.watermark.token_channel.config import TokenChannelConfig
```

Replace with:

```python
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
```

- [ ] **Step 10: Run all token_channel-related tests + extract tests + watermark tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/ tests/extract/ tests/watermark/test_config.py tests/watermark/test_pipeline.py tests/watermark/test_generator.py tests/integration/test_token_channel_layout_shims.py -v`

Expected: all green except the documented pre-existing baseline (6 mock-pollution in `tests/watermark/test_pipeline.py`, 1 in `tests/extract/test_joint_detection.py`).

Note: this scoped run replaces the full-suite run for verification — see baseline failures section in the header.

- [ ] **Step 11: Verify no first-party caller still imports a deprecated path**

Run:

```bash
grep -rn "from wfcllm.watermark.token_channel\." wfcllm/ scripts/ 2>/dev/null \
  | grep -v __pycache__ \
  | grep -v "core\.\|training\.\|runtime\." \
  | grep -v "wfcllm/watermark/token_channel/__init__.py:[0-9]*:from wfcllm.watermark.token_channel\.\(config\|protocol\|features\|model\|runtime\|train\(_corpus\|_corpus_streaming\|_workflow\)\?\|teacher\)"
```

Expected: no output (other than the deprecation shim files themselves, which legitimately import from the old paths' new homes — those are filtered by the `core.|training.|runtime.` guard).

If output appears, audit those sites and rewrite to canonical paths.

- [ ] **Step 12: Commit**

```bash
git add wfcllm/watermark/token_channel/__init__.py \
        wfcllm/cli/config_resolver.py \
        wfcllm/cli/runners.py \
        wfcllm/extract/config.py \
        wfcllm/extract/token_channel.py \
        wfcllm/extract/detector.py \
        wfcllm/extract/hypothesis.py \
        wfcllm/watermark/generator.py \
        wfcllm/watermark/config.py
git commit -m "refactor: switch first-party callers to wfcllm.watermark.token_channel.{core,training,runtime} canonical paths"
```

---

## Task 13: Create `wfcllm/pretrain/` package — `PretrainConfig`, `PretrainPipeline`, `run_pretrain`

**Files:**
- Create: `wfcllm/pretrain/__init__.py`
- Create: `wfcllm/pretrain/config.py`
- Create: `wfcllm/pretrain/pipeline.py`
- Create: `wfcllm/pretrain/runner.py`
- Create: `tests/integration/test_pretrain_phase.py`

- [ ] **Step 1: Write failing test for `PretrainConfig` default stages**

Write `tests/integration/test_pretrain_phase.py`:

```python
"""Tests for the pretrain phase: stage selection + fail-fast on missing encoder."""
from __future__ import annotations

import argparse

import pytest


def test_pretrain_config_default_stages_both():
    from wfcllm.pretrain.config import PretrainConfig
    cfg = PretrainConfig()
    assert cfg.stages == ["encoder", "lexical"]


def test_pretrain_config_normalizes_stage_order():
    """Even if user passes ['lexical', 'encoder'], pipeline runs in canonical order."""
    from wfcllm.pretrain.config import PretrainConfig
    cfg = PretrainConfig(stages=["lexical", "encoder"])
    assert cfg.stages == ["encoder", "lexical"]


def test_pretrain_config_rejects_unknown_stage():
    from wfcllm.pretrain.config import PretrainConfig
    with pytest.raises(ValueError, match="unknown stage"):
        PretrainConfig(stages=["encoder", "bogus"])
```

- [ ] **Step 2: Run failing test**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_pretrain_phase.py::test_pretrain_config_default_stages_both -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.pretrain'`.

- [ ] **Step 3: Create `wfcllm/pretrain/config.py`**

Write `wfcllm/pretrain/config.py`:

```python
"""Configuration for the pretrain phase."""
from __future__ import annotations

from dataclasses import dataclass, field

CANONICAL_STAGE_ORDER = ("encoder", "lexical")


@dataclass
class PretrainConfig:
    """Pretrain phase configuration: which stages to run.

    Stages are always executed in CANONICAL_STAGE_ORDER regardless of input order,
    because lexical training depends on a trained encoder checkpoint.
    """

    stages: list[str] = field(default_factory=lambda: list(CANONICAL_STAGE_ORDER))

    def __post_init__(self) -> None:
        unknown = [s for s in self.stages if s not in CANONICAL_STAGE_ORDER]
        if unknown:
            raise ValueError(
                f"unknown stage(s): {unknown!r}; allowed: {list(CANONICAL_STAGE_ORDER)}"
            )
        self.stages = [s for s in CANONICAL_STAGE_ORDER if s in self.stages]
```

- [ ] **Step 4: Run config tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_pretrain_phase.py -v -k "config"`
Expected: 3 passed.

- [ ] **Step 5: Write failing test for pipeline stage dispatch**

Append to `tests/integration/test_pretrain_phase.py`:

```python
def test_pipeline_runs_only_encoder_when_stages_encoder(monkeypatch):
    """--stages encoder dispatches only the encoder runner."""
    from wfcllm.pretrain.pipeline import PretrainPipeline
    from wfcllm.pretrain.config import PretrainConfig

    calls: list[str] = []

    def fake_encoder(args, state):
        calls.append("encoder")
        return 0

    def fake_lexical(args, state):
        calls.append("lexical")
        return 0

    monkeypatch.setattr("wfcllm.pretrain.pipeline.run_encoder", fake_encoder)
    monkeypatch.setattr("wfcllm.pretrain.pipeline.run_token_channel_train", fake_lexical)

    pipeline = PretrainPipeline(PretrainConfig(stages=["encoder"]))
    rc = pipeline.run(args=argparse.Namespace(), state=_DummyState(encoder_done=False))
    assert rc == 0
    assert calls == ["encoder"]


def test_pipeline_lexical_failfast_when_encoder_not_done(monkeypatch, capsys):
    """--stages lexical with no encoder ckpt prints error and returns 1."""
    from wfcllm.pretrain.pipeline import PretrainPipeline
    from wfcllm.pretrain.config import PretrainConfig

    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_encoder",
        lambda *a, **k: pytest.fail("encoder runner must not be called when stages=['lexical']"),
    )
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda *a, **k: pytest.fail("lexical runner must not be reached when encoder not done"),
    )

    pipeline = PretrainPipeline(PretrainConfig(stages=["lexical"]))
    rc = pipeline.run(args=argparse.Namespace(), state=_DummyState(encoder_done=False))
    assert rc == 1
    captured = capsys.readouterr()
    assert "encoder" in captured.err.lower() and "未完成" in captured.err


def test_pipeline_runs_both_in_order(monkeypatch):
    """--stages encoder lexical (or default) dispatches encoder then lexical."""
    from wfcllm.pretrain.pipeline import PretrainPipeline
    from wfcllm.pretrain.config import PretrainConfig

    calls: list[str] = []
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_encoder",
        lambda args, state: (calls.append("encoder") or 0),
    )
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda args, state: (calls.append("lexical") or 0),
    )
    pipeline = PretrainPipeline(PretrainConfig(stages=["encoder", "lexical"]))
    rc = pipeline.run(args=argparse.Namespace(), state=_DummyState(encoder_done=True))
    assert rc == 0
    assert calls == ["encoder", "lexical"]


class _DummyState:
    def __init__(self, *, encoder_done: bool):
        self._encoder_done = encoder_done

    def is_done(self, phase: str) -> bool:
        return self._encoder_done if phase == "encoder" else False
```

- [ ] **Step 6: Run pipeline tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_pretrain_phase.py -v -k "pipeline"`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.pretrain.pipeline'`.

- [ ] **Step 7: Create `wfcllm/pretrain/pipeline.py`**

Write `wfcllm/pretrain/pipeline.py`:

```python
"""Pretrain pipeline: dispatches encoder + lexical stages in canonical order."""
from __future__ import annotations

import argparse
import sys

from wfcllm.cli.runners import run_encoder, run_token_channel_train
from wfcllm.orchestration.state import RunStateManager
from wfcllm.pretrain.config import PretrainConfig


class PretrainPipeline:
    """Sequential dispatcher for pretrain stages.

    For each stage in cfg.stages (already canonically ordered):
      - "encoder": call run_encoder
      - "lexical": fail-fast if encoder not done, else call run_token_channel_train
    Returns the first non-zero exit code, or 0 if all stages succeed.
    """

    def __init__(self, cfg: PretrainConfig):
        self.cfg = cfg

    def run(self, args: argparse.Namespace, state: RunStateManager) -> int:
        for stage in self.cfg.stages:
            if stage == "encoder":
                rc = run_encoder(args, state)
                if rc != 0:
                    return rc
            elif stage == "lexical":
                if not state.is_done("encoder"):
                    print(
                        "[错误] pretrain --stages lexical 需要先完成 encoder 阶段（encoder 未完成）",
                        file=sys.stderr,
                    )
                    return 1
                rc = run_token_channel_train(args, state)
                if rc != 0:
                    return rc
            else:  # defensive — PretrainConfig validates
                print(f"[错误] unknown stage: {stage}", file=sys.stderr)
                return 1
        return 0
```

- [ ] **Step 8: Run pipeline tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_pretrain_phase.py -v -k "pipeline"`
Expected: 3 passed.

- [ ] **Step 9: Create `wfcllm/pretrain/runner.py`**

Write `wfcllm/pretrain/runner.py`:

```python
"""CLI runner for the pretrain phase."""
from __future__ import annotations

import argparse

from wfcllm.orchestration.state import RunStateManager
from wfcllm.pretrain.config import PretrainConfig
from wfcllm.pretrain.pipeline import PretrainPipeline


def run_pretrain(args: argparse.Namespace, state: RunStateManager) -> int:
    """Phase entry point. Reads --stages from args (None → default both)."""
    stages = getattr(args, "stages", None)
    cfg = PretrainConfig(stages=list(stages)) if stages else PretrainConfig()
    print(f"=== Phase: pretrain (stages={cfg.stages}) ===")
    return PretrainPipeline(cfg).run(args, state)
```

- [ ] **Step 10: Create `wfcllm/pretrain/__init__.py`**

Write `wfcllm/pretrain/__init__.py`:

```python
"""Pretrain phase: encoder + lexical channel joint training orchestration."""
from __future__ import annotations

from wfcllm.pretrain.config import PretrainConfig
from wfcllm.pretrain.pipeline import PretrainPipeline
from wfcllm.pretrain.runner import run_pretrain

__all__ = ["PretrainConfig", "PretrainPipeline", "run_pretrain"]
```

- [ ] **Step 11: Verify the package imports cleanly**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "from wfcllm.pretrain import PretrainConfig, PretrainPipeline, run_pretrain; print('ok')"`
Expected: `ok`.

- [ ] **Step 12: Commit**

```bash
git add wfcllm/pretrain/__init__.py \
        wfcllm/pretrain/config.py \
        wfcllm/pretrain/pipeline.py \
        wfcllm/pretrain/runner.py \
        tests/integration/test_pretrain_phase.py
git commit -m "feat: add wfcllm/pretrain/ package (PretrainConfig + PretrainPipeline + run_pretrain)"
```

---

## Task 14: Register `pretrain` phase in CLI + arguments + orchestration state

**Files:**
- Modify: `wfcllm/orchestration/state.py:9-10`
- Modify: `wfcllm/cli/arguments.py` (add `--stages` argument)
- Modify: `wfcllm/cli/entry.py` (register pretrain phase + import run_pretrain)
- Modify: `wfcllm/cli/runners.py` (add pretrain to `runners` dict)
- Modify: `tests/integration/test_pretrain_phase.py`

- [ ] **Step 1: Write failing CLI integration test**

Append to `tests/integration/test_pretrain_phase.py`:

```python
def test_cli_phase_pretrain_dispatches_pipeline(monkeypatch, tmp_path):
    """`run.py --phase pretrain --stages encoder` calls run_encoder via PretrainPipeline."""
    from wfcllm.cli.entry import main

    state_path = tmp_path / "run_state.json"
    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", state_path)

    calls: list[str] = []
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_encoder",
        lambda args, state: (calls.append("encoder") or 0),
    )
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda args, state: (calls.append("lexical") or 0),
    )

    rc = main(["--phase", "pretrain", "--stages", "encoder"])
    assert rc == 0
    assert calls == ["encoder"]


def test_cli_phase_pretrain_default_runs_both_in_order(monkeypatch, tmp_path):
    """`run.py --phase pretrain` (no --stages) runs encoder then lexical."""
    from wfcllm.cli.entry import main

    state_path = tmp_path / "run_state.json"
    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", state_path)

    calls: list[str] = []

    def fake_encoder(args, state):
        # Simulate encoder completing successfully so the lexical stage proceeds.
        state.mark_done("encoder", checkpoint="/tmp/fake")
        calls.append("encoder")
        return 0

    monkeypatch.setattr("wfcllm.pretrain.pipeline.run_encoder", fake_encoder)
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda args, state: (calls.append("lexical") or 0),
    )

    rc = main(["--phase", "pretrain"])
    assert rc == 0
    assert calls == ["encoder", "lexical"]
```

- [ ] **Step 2: Run failing CLI tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_pretrain_phase.py -v -k "cli_phase"`
Expected: FAIL — argparse rejects `--phase pretrain` because it is not in `ALL_PHASES`.

- [ ] **Step 3: Add `pretrain` to `OPTIONAL_PHASES` in `wfcllm/orchestration/state.py`**

Find:

```python
PHASES = ["encoder", "watermark", "extract"]
OPTIONAL_PHASES = ["generate-negative", "token-channel-train", "build-entropy-profile"]
```

Replace with:

```python
PHASES = ["encoder", "watermark", "extract"]
OPTIONAL_PHASES = ["pretrain", "generate-negative", "token-channel-train", "build-entropy-profile"]
```

- [ ] **Step 4: Add `--stages` argument in `wfcllm/cli/arguments.py`**

After the `--checkpoint` argument block (around line 60-61, before the "# Encoder 参数" comment), insert:

```python
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["encoder", "lexical"],
        default=None,
        help="（pretrain 专用）选择运行哪些 stage，默认两个都跑",
    )
```

- [ ] **Step 5: Register `run_pretrain` in `wfcllm/cli/entry.py`**

Find the import block:

```python
from wfcllm.cli.runners import (
    is_compare_only_mode,
    validate_compare_only_mode,
    run_build_entropy_profile,
    run_encoder,
    run_extract,
    run_generate_negative,
    run_token_channel_train,
    run_watermark,
)
```

Add `run_pretrain` import (alphabetic order):

```python
from wfcllm.cli.runners import (
    is_compare_only_mode,
    validate_compare_only_mode,
    run_build_entropy_profile,
    run_encoder,
    run_extract,
    run_generate_negative,
    run_token_channel_train,
    run_watermark,
)
from wfcllm.pretrain.runner import run_pretrain
```

Find the `_populate_phase_registry` function:

```python
def _populate_phase_registry(reg: PhaseRegistry) -> None:
    reg.register("encoder", run_encoder)
    reg.register("watermark", run_watermark)
    reg.register("extract", run_extract)
    reg.register("generate-negative", run_generate_negative)
    reg.register("token-channel-train", run_token_channel_train)
    reg.register("build-entropy-profile", run_build_entropy_profile)
```

Replace with:

```python
def _populate_phase_registry(reg: PhaseRegistry) -> None:
    reg.register("encoder", run_encoder)
    reg.register("watermark", run_watermark)
    reg.register("extract", run_extract)
    reg.register("generate-negative", run_generate_negative)
    reg.register("token-channel-train", run_token_channel_train)
    reg.register("build-entropy-profile", run_build_entropy_profile)
    reg.register("pretrain", run_pretrain)
```

- [ ] **Step 6: Register pretrain in `wfcllm/cli/runners.py` dispatch dict**

Find in `wfcllm/cli/runners.py` (around line 100-108):

```python
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
        "generate-negative": run_generate_negative,
        "token-channel-train": run_token_channel_train,
    }
```

Replace with:

```python
    from wfcllm.pretrain.runner import run_pretrain  # local import to avoid circular dep
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
        "generate-negative": run_generate_negative,
        "token-channel-train": run_token_channel_train,
        "pretrain": run_pretrain,
    }
```

(Note: this dispatch dict is only used by code paths that don't go through the orchestrator; the orchestrator uses `PhaseRegistry` populated in Step 5. We register `pretrain` in both for symmetry. `build-entropy-profile` is deliberately not added to this dict — its existing routing via the orchestrator is preserved unchanged.)

- [ ] **Step 7: Run CLI tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_pretrain_phase.py -v`
Expected: all passed (config + pipeline + cli_phase tests = 8 tests total).

- [ ] **Step 8: Verify `--phase pretrain --status` works (smoke test)**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status`
Expected: prints status table including the new `pretrain` row.

- [ ] **Step 9: Commit**

```bash
git add wfcllm/orchestration/state.py \
        wfcllm/cli/arguments.py \
        wfcllm/cli/entry.py \
        wfcllm/cli/runners.py \
        tests/integration/test_pretrain_phase.py
git commit -m "feat: register pretrain phase in CLI (--phase pretrain --stages encoder|lexical)"
```

---

## Task 15: Final scoped verification + cleanup

**Files:**
- Verify only — no changes

- [ ] **Step 1: Run scoped Phase 5 verification suite**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/integration/test_token_channel_layout_shims.py \
  tests/integration/test_pretrain_phase.py \
  tests/integration/test_extract_calibration_shims.py \
  tests/integration/test_calibrate_threshold_runner.py \
  tests/integration/test_calibrate_threshold_cli.py \
  tests/integration/test_cli_resolution.py \
  tests/watermark/token_channel/ \
  tests/watermark/test_config.py \
  tests/watermark/test_context.py \
  tests/watermark/test_generator.py \
  tests/extract/test_token_channel.py \
  tests/extract/test_negative_corpus.py \
  tests/extract/test_config.py \
  -v
```

Expected: all green. The pre-existing baseline failures (`tests/watermark/test_pipeline.py` 6 mock-pollution, `tests/extract/test_joint_detection.py` 1, `tests/test_run_config.py` 4 SimpleNamespace `min_blocks`) are deliberately NOT included in this scoped run because none of them touch Phase 5 changes.

- [ ] **Step 2: Verify no pyc files leak from old paths**

Run:

```bash
find wfcllm/watermark/token_channel/ -name "*.pyc" | xargs rm -f
find wfcllm/watermark/token_channel/__pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
from wfcllm.watermark.token_channel.core.protocol import build_partition
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.core.model import TokenChannelModel
from wfcllm.watermark.token_channel.runtime.injector import TokenChannelRuntime
from wfcllm.watermark.token_channel.training.trainer import train_one_epoch
from wfcllm.watermark.token_channel.training.corpus import build_training_rows
from wfcllm.watermark.token_channel.training.corpus_streaming import stream_training_cache
from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows
from wfcllm.watermark.token_channel.training.workflow import run_token_channel_train_workflow
from wfcllm.pretrain import PretrainConfig, PretrainPipeline, run_pretrain
print('all canonical imports ok')
"
```

Expected: prints `all canonical imports ok`.

- [ ] **Step 3: Confirm no first-party caller (excluding shims) imports a deprecated path**

Run:

```bash
grep -rn "from wfcllm.watermark.token_channel\." wfcllm/ scripts/ 2>/dev/null \
  | grep -v __pycache__ \
  | grep -v "wfcllm/watermark/token_channel/" \
  | grep -v "core\.\|training\.\|runtime\."
```

Expected: no output. (The first `grep -v` excludes the deprecation shim files themselves, since they legitimately bridge old paths to new.)

- [ ] **Step 4: Final commit if any cleanup needed (otherwise skip)**

Only commit if Step 2 or Step 3 found stale state worth fixing. Otherwise:

```bash
git log --oneline -20
```

to confirm the Phase 5 commit chain is clean.

---

## Self-Review Notes

**Spec coverage check (§4.1 + §10 step 5):**

| Spec requirement | Implementing task |
|---|---|
| Move `config.py` → `core/config.py` | Task 2 |
| Move `protocol.py` → `core/protocol.py` | Task 3 |
| Move `features.py` → `core/features.py` | Task 4 |
| Move `model.py` → `core/model.py` | Task 5 |
| Move `runtime.py` → `runtime/injector.py` | Task 6 |
| Move `teacher.py` → `training/teacher.py` | Task 7 |
| Move `train.py` → `training/trainer.py` | Task 8 |
| Move `train_corpus.py` → `training/corpus.py` | Task 9 |
| Move `train_corpus_streaming.py` → `training/corpus_streaming.py` | Task 10 |
| Move `train_workflow.py` → `training/workflow.py` | Task 11 |
| Add `wfcllm/pretrain/config.py` with `PretrainConfig` + `stages` | Task 13 |
| Add `wfcllm/pretrain/pipeline.py` with `PretrainPipeline.run` | Task 13 |
| CLI `--phase pretrain --stages encoder|lexical` | Task 14 |
| Stages execute in fixed canonical order regardless of input order | Task 13 step 3 (`PretrainConfig.__post_init__`) |
| Lexical fail-fast when encoder not done | Task 13 step 7 (`PretrainPipeline.run`) |
| State_dict compatibility (class names + key paths unchanged) | Architecture note + Task 5 verbatim copy |

**Out-of-scope items (deferred per spec §10 ordering):**

- `run_state.json` nested-key migration (`pretrain.encoder` / `pretrain.lexical`) — spec §8.3 belongs to a later phase or shim cleanup.
- `config_resolver` legacy schema merge — spec §7 / §8.3 deferred.

**Why no full-suite pytest in verification:** per user preference (in CLAUDE.md context + earlier session feedback), full pytest is too slow; Task 15 step 1 enumerates the exact subset that exercises Phase 5 changes.
