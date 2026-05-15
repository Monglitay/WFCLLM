# Phase 6: lang/ + datasets/ Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce two pluggable subpackages — `wfcllm/lang/` (`LanguageAdapter` ABC + `PythonAdapter`) and `wfcllm/datasets/` (`DatasetAdapter` ABC + `HumanEvalAdapter` + `MBPPAdapter` + `HumanEvalPackAdapter` skeleton) — built on the Phase 1 `Registry[T]` container. Existing logic (`wfcllm/common/ast_parser.py`, `wfcllm/common/transform/`, `wfcllm/common/dataset_loader.py`) is moved verbatim under the new adapters; old import paths remain working via deprecation shims so no caller breaks.

**Architecture:** Pure code-move + thin ABC wrapper, mirroring Phases 2–5. The adapters wrap existing module-level functions; the adapters themselves carry no algorithmic logic. `LanguageAdapter.extract_blocks` delegates to the verbatim-moved `extract_statement_blocks`; `LanguageAdapter.positive_rules` / `.negative_rules` delegate to the verbatim-moved `get_all_*_rules`; `DatasetAdapter.iter_samples` wraps the verbatim-moved `load_prompts` / `load_reference_solutions`. Spec §5.1 ("today only implement PythonAdapter; lang/java, lang/cpp, lang/js are empty placeholders") and §5.2 ("HumanEvalPack: implement only `__init__` + `NotImplementedError`") are honoured strictly.

**Tech Stack:** Python 3.10+, pytest, dataclasses, `tree_sitter_python` (already a project dep), `datasets` (already a project dep), no new external deps.

**Spec reference:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §5.1 + §5.2 + §10 step 6.

**Out of scope (explicitly NOT in this phase):**

- Adding `language: str = "python"` / `dataset: str = "humaneval"` fields to phase dataclasses (`EncoderConfig`, `WatermarkConfig`, `ExtractConfig`, etc.) — spec §5.1/§5.2 mentions config-level injection, but this couples to the still-pending §7 schema rewrite. Defer to a later phase. Today's adapters are usable from new call sites that pick the language/dataset at construction time; existing callers keep working via shims.
- `config_resolver` `(dataset, language)` cross-validation (spec §5.2 "联合校验") — defer to the same future phase.
- Implementing real `HumanEvalPack` data loading — spec §5.2 explicitly defers this. We only ship the skeleton + `NotImplementedError`.
- `lang/java/`, `lang/cpp/`, `lang/js/` actual implementations — spec §5.1 says these are empty placeholder packages today.
- Refactoring `common/ast_parser.py` to remove the singleton `PythonParser` or to make the parser language-agnostic — pure move, no algorithm change.
- Modifying call sites under `tests/common/`, `tests/extract/`, `tests/encoder/`, `tests/watermark/` etc. — they keep importing from the old `wfcllm.common.*` paths to exercise the shims (same convention as Phases 2–5).
- `experiment/` and `tools/` directories — not touched per project rules.

---

## File Structure

**New canonical files (verbatim moves from `wfcllm/common/`):**

| Old path | New canonical path | Move kind |
|---|---|---|
| `wfcllm/common/ast_parser.py` | `wfcllm/lang/python/parser.py` | verbatim copy (file body kept identical except the file is now under a Python-specific package) |
| `wfcllm/common/ast_parser.py` (the `SIMPLE_STATEMENT_TYPES` / `COMPOUND_STATEMENT_TYPES` / `STATEMENT_TYPES` constants only) | `wfcllm/lang/python/statement_types.py` | the three frozensets are re-exported through `parser.py` for back-compat — see Task 3 |
| `wfcllm/common/transform/base.py` | `wfcllm/lang/python/transform/base.py` | verbatim |
| `wfcllm/common/transform/engine.py` | `wfcllm/lang/python/transform/engine.py` | verbatim (rewrite intra-package `from wfcllm.common.transform.base` → `from wfcllm.lang.python.transform.base`) |
| `wfcllm/common/transform/positive/*.py` | `wfcllm/lang/python/transform/positive/*.py` | verbatim (rewrite intra-package imports) |
| `wfcllm/common/transform/negative/*.py` | `wfcllm/lang/python/transform/negative/*.py` | verbatim (rewrite intra-package imports) |
| `wfcllm/common/dataset_loader.py` | `wfcllm/datasets/loaders/local.py` | verbatim (functions reused by adapters; raw module kept for back-compat shim) |

**New ABC + Registry files:**

- `wfcllm/lang/__init__.py` — exports `LanguageAdapter`, `StatementTypes`, `register`, `get`, `names`, and triggers `PythonAdapter` registration (side-effect import of `wfcllm.lang.python`).
- `wfcllm/lang/adapter.py` — `LanguageAdapter` ABC + `StatementTypes` dataclass.
- `wfcllm/lang/registry.py` — module-level `Registry[LanguageAdapter]` plus `register(name)` decorator and `get(name)` / `names()` helpers. Built on top of `wfcllm.common.registry.Registry`.
- `wfcllm/lang/python/__init__.py` — side-effect: imports `adapter` so registration runs; exports `PythonAdapter`.
- `wfcllm/lang/python/adapter.py` — `@register("python") class PythonAdapter(LanguageAdapter)`; delegates each ABC method to the moved-in implementations under `parser.py` / `transform/`.
- `wfcllm/lang/java/__init__.py`, `wfcllm/lang/cpp/__init__.py`, `wfcllm/lang/js/__init__.py` — empty placeholder packages (docstring only).
- `wfcllm/datasets/__init__.py` — exports `DatasetAdapter`, `CodeSample`, `register`, `get`, `names`; triggers `HumanEvalAdapter`/`MBPPAdapter`/`HumanEvalPackAdapter` registration.
- `wfcllm/datasets/adapter.py` — `DatasetAdapter` ABC + `CodeSample` dataclass.
- `wfcllm/datasets/registry.py` — module-level `Registry[DatasetAdapter]` + `register(name)` decorator + `get(name)`.
- `wfcllm/datasets/humaneval.py` — `@register("humaneval") class HumanEvalAdapter(DatasetAdapter)` delegating to `wfcllm.datasets.loaders.local`.
- `wfcllm/datasets/mbpp.py` — `@register("mbpp") class MBPPAdapter(DatasetAdapter)` delegating to `wfcllm.datasets.loaders.local`.
- `wfcllm/datasets/humanevalpack.py` — `@register("humanevalpack") class HumanEvalPackAdapter(DatasetAdapter)`; `iter_samples` raises `NotImplementedError` with the spec-mandated message.
- `wfcllm/datasets/loaders/__init__.py` — empty docstring.
- `wfcllm/datasets/loaders/local.py` — verbatim body of current `wfcllm/common/dataset_loader.py`.

**New tests:**

- `tests/lang/__init__.py` — empty.
- `tests/lang/test_python_adapter.py` — contract test suite for `PythonAdapter` (spec §5.1 "契约测试"): statement-type completeness, `extract_blocks` matches old `extract_statement_blocks`, rule counts equal old `get_all_*_rules` counts.
- `tests/lang/test_registry.py` — `Registry[LanguageAdapter]` behaviours (register / get / KeyError on missing / duplicate-name error).
- `tests/datasets/__init__.py` — empty.
- `tests/datasets/test_adapters.py` — round-trip tests for `HumanEvalAdapter` + `MBPPAdapter` (call `get("humaneval")`, iterate samples, compare to `load_prompts`); `HumanEvalPackAdapter` raises `NotImplementedError` from `iter_samples`.
- `tests/integration/test_common_lang_shims.py` — shim tests: every `wfcllm.common.ast_parser`, `wfcllm.common.transform.*`, `wfcllm.common.dataset_loader` public symbol resolvable from old path, emits `DeprecationWarning`, identity-equal to canonical symbol.

**Modified existing files (shims only — no logic change):**

- `wfcllm/common/ast_parser.py` — replaced with deprecation shim re-exporting from `wfcllm.lang.python.parser`.
- `wfcllm/common/transform/__init__.py` — replaced with deprecation shim re-exporting from `wfcllm.lang.python.transform`.
- `wfcllm/common/transform/base.py` — deprecation shim re-exporting from `wfcllm.lang.python.transform.base`.
- `wfcllm/common/transform/engine.py` — deprecation shim re-exporting from `wfcllm.lang.python.transform.engine`.
- `wfcllm/common/transform/positive/__init__.py` — deprecation shim re-exporting from `wfcllm.lang.python.transform.positive`.
- `wfcllm/common/transform/negative/__init__.py` — deprecation shim re-exporting from `wfcllm.lang.python.transform.negative`.
- `wfcllm/common/transform/positive/<each-rule-file>.py` — deprecation shim re-exporting from `wfcllm.lang.python.transform.positive.<same-name>`.
- `wfcllm/common/transform/negative/<each-rule-file>.py` — deprecation shim re-exporting from `wfcllm.lang.python.transform.negative.<same-name>`.
- `wfcllm/common/dataset_loader.py` — deprecation shim re-exporting from `wfcllm.datasets.loaders.local`.
- `wfcllm/common/__init__.py` — keep current re-exports but route through canonical paths (single-line import-source change; no public API change).

**Unchanged (do NOT touch):**

- All call-site files under `wfcllm/` listed in the audit below — they keep importing from `wfcllm.common.*` to exercise the shims. (Side benefit: catches any shim regression.)
- All test files under `tests/common/`, `tests/extract/`, `tests/encoder/`, `tests/watermark/` — already using old paths and stay that way.
- `wfcllm/common/block_contract.py`, `wfcllm/common/checkpoint.py`, `wfcllm/common/offline_code_eval.py`, `wfcllm/common/registry.py` — out of scope.
- `experiment/`, `tools/`, `scripts/` — out of scope.

**First-party caller audit (these MUST continue importing from old paths — they exercise the shims):**

```
wfcllm/encoder/train.py:18-21        # ast_parser + transform engine/positive/negative
wfcllm/extract/calibration/negative_corpus.py:14   # dataset_loader
wfcllm/extract/calibration/threshold.py:7          # ast_parser
wfcllm/extract/detector.py:8                       # ast_parser
wfcllm/extract/dp_selector.py:7                    # ast_parser
wfcllm/extract/scorer.py:5                         # ast_parser
wfcllm/watermark/adaptive_gamma/entropy.py:11      # ast_parser
wfcllm/watermark/generator.py:12                   # ast_parser
wfcllm/watermark/interceptor.py:10                 # ast_parser
wfcllm/watermark/pipeline.py:22                    # dataset_loader
wfcllm/watermark/token_channel/training/corpus.py:15-16     # transform engine + positive
wfcllm/watermark/token_channel/training/workflow.py:16      # dataset_loader
wfcllm/evaluation/dual_channel.py:23               # dataset_loader
```

This list is the source of truth for Task 14's grep verification.

---

## Conventions

**Deprecation shim template** (matches Phase 2/3/4/5 pattern):

```python
"""Deprecated import path. Use wfcllm.<NEW> instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.<OLD> is deprecated; use wfcllm.<NEW>",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.<NEW> import *  # noqa: E402, F401, F403
from wfcllm.<NEW> import (  # noqa: E402, F401
    <SYMBOLS_USED_BY_TESTS_OR_MONKEYPATCHES>,
)
```

`from … import *` covers ordinary consumers; explicit names preserve `monkeypatch.setattr("wfcllm.common.ast_parser.PythonParser", …)`-style tests.

**Singleton & state preservation:** `PythonParser` in `wfcllm/common/ast_parser.py` is a process-level singleton (the `_instance` class variable). To preserve identity across the shim, the canonical file at `wfcllm/lang/python/parser.py` must declare the singleton, and the shim must re-export `PythonParser` by name (not redefine). This is automatic for `from … import *` + explicit named re-export.

**Test command:** `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest <path> -v`

**Commit message format:** `<type>: <description>` per `CLAUDE.md`. Each task ends with one commit.

**Pre-existing baseline failures (NOT introduced by Phase 6 — exclude from "no new failures" criterion):**

- `tests/watermark/test_pipeline.py` — 6 `MagicMock` mock-pollution failures around `_config.min_blocks`.
- `tests/extract/test_joint_detection.py` — 1 detection failure.
- `tests/test_run_config.py` — 4 `SimpleNamespace` fixture missing `min_blocks` failures.

These existed on `main` before Phase 6 began. If verification shows exactly these failures and nothing else, treat the task as passing.

---

## Task 1: Create `wfcllm/lang/` package skeleton + first failing test

**Files:**
- Create: `wfcllm/lang/__init__.py`
- Create: `wfcllm/lang/adapter.py`
- Create: `wfcllm/lang/registry.py`
- Create: `wfcllm/lang/python/__init__.py`
- Create: `wfcllm/lang/java/__init__.py`
- Create: `wfcllm/lang/cpp/__init__.py`
- Create: `wfcllm/lang/js/__init__.py`
- Create: `tests/lang/__init__.py`
- Create: `tests/lang/test_python_adapter.py`

- [ ] **Step 1: Create `wfcllm/lang/adapter.py`**

```python
"""Language adapter ABC for the WFCLLM pluggable language layer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StatementTypes:
    """Per-language partition of AST node types into simple / compound statement blocks."""

    simple: frozenset[str]
    compound: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        return self.simple | self.compound


class LanguageAdapter(ABC):
    """Pluggable per-language strategy used by encoder, watermark, and extract phases."""

    name: str = ""

    @abstractmethod
    def statement_types(self) -> StatementTypes: ...

    @abstractmethod
    def extract_blocks(self, source: str) -> list:
        """Parse `source` and return a flat list of statement blocks."""

    @abstractmethod
    def positive_rules(self) -> list:
        """Return the canonical list of positive (semantic-equivalent) transform rules."""

    @abstractmethod
    def negative_rules(self) -> list:
        """Return the canonical list of negative (semantic-breaking) transform rules."""
```

- [ ] **Step 2: Create `wfcllm/lang/registry.py`**

```python
"""Language-adapter registry (module-level singleton)."""
from __future__ import annotations

from wfcllm.common.registry import Registry
from wfcllm.lang.adapter import LanguageAdapter

_REGISTRY: Registry[LanguageAdapter] = Registry("language")


def register(name: str):
    """Decorator: register a LanguageAdapter subclass under `name`."""

    def deco(cls: type[LanguageAdapter]) -> type[LanguageAdapter]:
        _REGISTRY.register(name, cls)
        return cls

    return deco


def get(name: str) -> LanguageAdapter:
    return _REGISTRY.get(name)


def names() -> list[str]:
    return _REGISTRY.names()


def __contains__(name: str) -> bool:  # noqa: PLW3201 - module-level dunder is intentional
    return name in _REGISTRY
```

- [ ] **Step 3: Create empty placeholder packages**

Write `wfcllm/lang/python/__init__.py`:

```python
"""Python language adapter."""
# adapter.py will be created in Task 4; importing it triggers @register("python")
```

Write `wfcllm/lang/java/__init__.py`:

```python
"""Java language adapter — placeholder for future implementation (spec §5.1)."""
```

Same one-line docstring (with `cpp` / `js` substituted) for `wfcllm/lang/cpp/__init__.py` and `wfcllm/lang/js/__init__.py`.

- [ ] **Step 4: Create `wfcllm/lang/__init__.py`**

```python
"""WFCLLM language-adapter layer (spec §5.1)."""
from wfcllm.lang.adapter import LanguageAdapter, StatementTypes
from wfcllm.lang.registry import get, names, register

# Importing the python subpackage triggers @register("python")
from wfcllm.lang import python  # noqa: F401, E402

__all__ = ["LanguageAdapter", "StatementTypes", "get", "names", "register"]
```

- [ ] **Step 5: Write the first failing contract test**

Write `tests/lang/__init__.py` as an empty file.

Write `tests/lang/test_python_adapter.py`:

```python
"""Contract tests for PythonAdapter (spec §5.1 "契约测试")."""
from __future__ import annotations


def test_python_adapter_registered():
    from wfcllm import lang
    assert "python" in lang.names()


def test_python_adapter_name_field():
    from wfcllm import lang
    assert lang.get("python").name == "python"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/lang/test_python_adapter.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.lang'` (or `'python'` not registered).

- [ ] **Step 7: Commit**

```bash
git add wfcllm/lang/__init__.py \
        wfcllm/lang/adapter.py \
        wfcllm/lang/registry.py \
        wfcllm/lang/python/__init__.py \
        wfcllm/lang/java/__init__.py \
        wfcllm/lang/cpp/__init__.py \
        wfcllm/lang/js/__init__.py \
        tests/lang/__init__.py \
        tests/lang/test_python_adapter.py
git commit -m "feat: scaffold wfcllm/lang package with LanguageAdapter ABC + registry"
```

---

## Task 2: Move `common/ast_parser.py` → `lang/python/parser.py` (verbatim)

**Files:**
- Create: `wfcllm/lang/python/parser.py`
- Modify: nothing else yet (shim added in Task 3)

- [ ] **Step 1: Copy `ast_parser.py` body to `lang/python/parser.py` verbatim**

Run:

```bash
cp wfcllm/common/ast_parser.py wfcllm/lang/python/parser.py
```

The file has no intra-package imports under `wfcllm.common.*`, only `tree_sitter` / `tree_sitter_python` / stdlib. No rewrite needed. Verify:

```bash
grep -n "wfcllm" wfcllm/lang/python/parser.py
```

Expected: no output.

- [ ] **Step 2: Sanity-check that the new module imports cleanly**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "from wfcllm.lang.python.parser import PythonParser, extract_statement_blocks, StatementBlock; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add wfcllm/lang/python/parser.py
git commit -m "refactor: move common/ast_parser.py to lang/python/parser.py (verbatim)"
```

---

## Task 3: Replace `common/ast_parser.py` with a deprecation shim

**Files:**
- Modify: `wfcllm/common/ast_parser.py` (replaced with shim)
- Modify: `tests/integration/test_common_lang_shims.py` (created here with first 3 tests)

- [ ] **Step 1: Create `tests/integration/test_common_lang_shims.py` with a failing shim test**

```python
"""Tests that wfcllm.common.* deprecated import paths still work via shims (Phase 6)."""
from __future__ import annotations

import warnings


def test_old_ast_parser_emits_deprecation_warning():
    import importlib
    import sys

    sys.modules.pop("wfcllm.common.ast_parser", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.common.ast_parser")
    msgs = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert msgs, f"expected DeprecationWarning, got: {[w.category for w in caught]}"


def test_old_ast_parser_symbol_identity():
    """Old path must re-export the SAME object as the new path (preserves singletons)."""
    import wfcllm.common.ast_parser as old
    import wfcllm.lang.python.parser as new
    assert old.PythonParser is new.PythonParser
    assert old.extract_statement_blocks is new.extract_statement_blocks
    assert old.StatementBlock is new.StatementBlock


def test_old_ast_parser_constants_round_trip():
    import wfcllm.common.ast_parser as old
    import wfcllm.lang.python.parser as new
    assert old.SIMPLE_STATEMENT_TYPES == new.SIMPLE_STATEMENT_TYPES
    assert old.COMPOUND_STATEMENT_TYPES == new.COMPOUND_STATEMENT_TYPES
    assert old.STATEMENT_TYPES == new.STATEMENT_TYPES
```

- [ ] **Step 2: Run to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py -v`

Expected: FAIL on `test_old_ast_parser_emits_deprecation_warning` (no warning emitted yet) and `test_old_ast_parser_symbol_identity` (objects are not the same — they are duplicated definitions).

- [ ] **Step 3: Replace `wfcllm/common/ast_parser.py` with the shim**

Write `wfcllm/common/ast_parser.py`:

```python
"""Deprecated import path. Use wfcllm.lang.python.parser instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.ast_parser is deprecated; use wfcllm.lang.python.parser",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.parser import *  # noqa: E402, F401, F403
from wfcllm.lang.python.parser import (  # noqa: E402, F401
    PY_LANGUAGE,
    SIMPLE_STATEMENT_TYPES,
    COMPOUND_STATEMENT_TYPES,
    STATEMENT_TYPES,
    PythonParser,
    StatementBlock,
    extract_statement_blocks,
)
```

- [ ] **Step 4: Run the 3 shim tests and confirm they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py -v`

Expected: 3 passed.

- [ ] **Step 5: Run the *existing* `tests/common/test_ast_parser.py` to confirm no regression on real callers**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/common/test_ast_parser.py -v`

Expected: all pre-existing pass count unchanged. No new failures.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/common/ast_parser.py tests/integration/test_common_lang_shims.py
git commit -m "refactor: collapse common/ast_parser.py to deprecation shim"
```

---

## Task 4: Implement `PythonAdapter` + green the registration tests

**Files:**
- Create: `wfcllm/lang/python/adapter.py`
- Modify: `tests/lang/test_python_adapter.py` (append contract tests)

- [ ] **Step 1: Create `wfcllm/lang/python/adapter.py`**

```python
"""PythonAdapter — wraps the existing tree-sitter Python parsing + transform rules."""
from __future__ import annotations

from wfcllm.lang.adapter import LanguageAdapter, StatementTypes
from wfcllm.lang.python.parser import (
    COMPOUND_STATEMENT_TYPES,
    SIMPLE_STATEMENT_TYPES,
    StatementBlock,
    extract_statement_blocks,
)
from wfcllm.lang.registry import register


_STATEMENT_TYPES = StatementTypes(
    simple=SIMPLE_STATEMENT_TYPES,
    compound=COMPOUND_STATEMENT_TYPES,
)


@register("python")
class PythonAdapter(LanguageAdapter):
    """Tree-sitter-based Python adapter."""

    name = "python"

    def statement_types(self) -> StatementTypes:
        return _STATEMENT_TYPES

    def extract_blocks(self, source: str) -> list[StatementBlock]:
        return extract_statement_blocks(source)

    def positive_rules(self) -> list:
        # Local import: lang.python.transform doesn't exist until Task 7.
        # The methods are defined now to satisfy the ABC, but should only be
        # CALLED after Task 7. Task 4 only exercises statement_types / extract_blocks.
        from wfcllm.lang.python.transform.positive import get_all_positive_rules
        return get_all_positive_rules()

    def negative_rules(self) -> list:
        from wfcllm.lang.python.transform.negative import get_all_negative_rules
        return get_all_negative_rules()
```

- [ ] **Step 2: Wire registration: update `wfcllm/lang/python/__init__.py`**

Replace contents:

```python
"""Python language adapter."""
from wfcllm.lang.python.adapter import PythonAdapter  # noqa: F401  (side-effect: @register)
```

- [ ] **Step 3: Append contract tests for `PythonAdapter.extract_blocks` + `statement_types`**

Append to `tests/lang/test_python_adapter.py`:

```python
def test_python_adapter_statement_types_match_old_constants():
    """LanguageAdapter.statement_types must equal the legacy frozensets exactly."""
    from wfcllm.lang import get
    from wfcllm.lang.python.parser import (
        SIMPLE_STATEMENT_TYPES,
        COMPOUND_STATEMENT_TYPES,
    )
    types = get("python").statement_types()
    assert types.simple == SIMPLE_STATEMENT_TYPES
    assert types.compound == COMPOUND_STATEMENT_TYPES
    assert types.all == SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES


def test_python_adapter_extract_blocks_matches_legacy():
    """LanguageAdapter.extract_blocks must produce the same blocks as legacy extract_statement_blocks."""
    from wfcllm.lang import get
    from wfcllm.lang.python.parser import extract_statement_blocks

    source = "def f(x):\n    return x + 1\n\nclass C:\n    pass\n"
    new = get("python").extract_blocks(source)
    old = extract_statement_blocks(source)
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert (a.block_id, a.block_type, a.node_type, a.source) == (
            b.block_id, b.block_type, b.node_type, b.source,
        )


def test_unknown_language_raises_keyerror():
    from wfcllm import lang
    import pytest
    with pytest.raises(KeyError, match="unknown language"):
        lang.get("klingon")
```

- [ ] **Step 4: Run the contract tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/lang/test_python_adapter.py -v`

Expected: 5 passed (the 2 originals from Task 1 + the 3 new ones). The `positive_rules` / `negative_rules` methods are not invoked yet — they would fail since `wfcllm.lang.python.transform` doesn't exist; that is covered in Tasks 5–7.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/lang/python/adapter.py wfcllm/lang/python/__init__.py tests/lang/test_python_adapter.py
git commit -m "feat: add PythonAdapter implementing LanguageAdapter contract"
```

---

## Task 5: Add `Registry[LanguageAdapter]` behaviour test

**Files:**
- Create: `tests/lang/test_registry.py`

- [ ] **Step 1: Write the test file**

```python
"""Behaviours of the wfcllm.lang module-level registry."""
from __future__ import annotations

import pytest


def test_register_then_get_returns_instance():
    from wfcllm import lang
    instance = lang.get("python")
    assert instance.name == "python"
    # get() must return a fresh instance each call (per Registry.get contract).
    instance2 = lang.get("python")
    assert instance is not instance2


def test_get_unknown_raises_keyerror_with_helpful_message():
    from wfcllm import lang
    with pytest.raises(KeyError) as exc:
        lang.get("does-not-exist")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    assert "python" in msg  # registered names listed


def test_duplicate_register_raises():
    from wfcllm.lang import register
    from wfcllm.lang.adapter import LanguageAdapter, StatementTypes

    class _Dup(LanguageAdapter):
        name = "python"
        def statement_types(self): return StatementTypes(frozenset(), frozenset())
        def extract_blocks(self, source): return []
        def positive_rules(self): return []
        def negative_rules(self): return []

    with pytest.raises(ValueError, match="already registered"):
        register("python")(_Dup)


def test_names_includes_python():
    from wfcllm import lang
    assert "python" in lang.names()
```

- [ ] **Step 2: Run the tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/lang/test_registry.py -v`

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/lang/test_registry.py
git commit -m "test: cover wfcllm.lang registry register/get/duplicate/names"
```

---

## Task 6: Move `common/transform/base.py` + `engine.py` → `lang/python/transform/` (verbatim) with shims

**Files:**
- Create: `wfcllm/lang/python/transform/__init__.py`
- Create: `wfcllm/lang/python/transform/base.py`
- Create: `wfcllm/lang/python/transform/engine.py`
- Modify: `wfcllm/common/transform/base.py` (replace with shim)
- Modify: `wfcllm/common/transform/engine.py` (replace with shim)
- Modify: `tests/integration/test_common_lang_shims.py` (append shim tests)

- [ ] **Step 1: Create the package + copy `base.py`**

Run:

```bash
mkdir -p wfcllm/lang/python/transform
```

Write `wfcllm/lang/python/transform/__init__.py`:

```python
"""Python-specific code transformation rules (moved from wfcllm.common.transform)."""

from wfcllm.lang.python.transform.base import Match, Rule, parse_code
from wfcllm.lang.python.transform.engine import TransformEngine

__all__ = ["Match", "Rule", "parse_code", "TransformEngine"]
```

Copy `base.py` verbatim:

```bash
cp wfcllm/common/transform/base.py wfcllm/lang/python/transform/base.py
```

No intra-package imports under `wfcllm.common.transform` exist in `base.py` — verify:

```bash
grep -n "wfcllm" wfcllm/lang/python/transform/base.py
```

Expected: no output.

- [ ] **Step 2: Copy `engine.py` and rewrite its single intra-package import**

Run:

```bash
cp wfcllm/common/transform/engine.py wfcllm/lang/python/transform/engine.py
```

`engine.py` line 7 imports `from wfcllm.common.transform.base import Rule, parse_code`. Edit `wfcllm/lang/python/transform/engine.py` to switch the source:

```python
from wfcllm.lang.python.transform.base import Rule, parse_code
```

Verify by grep:

```bash
grep -n "wfcllm" wfcllm/lang/python/transform/engine.py
```

Expected: exactly one line — `from wfcllm.lang.python.transform.base import Rule, parse_code`.

- [ ] **Step 3: Append shim equivalence tests**

Append to `tests/integration/test_common_lang_shims.py`:

```python
def test_old_transform_base_symbol_identity():
    import wfcllm.common.transform.base as old
    import wfcllm.lang.python.transform.base as new
    assert old.Rule is new.Rule
    assert old.Match is new.Match
    assert old.parse_code is new.parse_code


def test_old_transform_engine_symbol_identity():
    import wfcllm.common.transform.engine as old
    import wfcllm.lang.python.transform.engine as new
    assert old.TransformEngine is new.TransformEngine
```

- [ ] **Step 4: Run to verify the new tests fail** (no shim yet)

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py -v`

Expected: 2 new tests FAIL (`Rule` / `TransformEngine` are duplicated, not identity-equal).

- [ ] **Step 5: Replace `common/transform/base.py` with shim**

Write `wfcllm/common/transform/base.py`:

```python
"""Deprecated import path. Use wfcllm.lang.python.transform.base instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.base is deprecated; use wfcllm.lang.python.transform.base",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.base import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.base import (  # noqa: E402, F401
    PY_LANGUAGE,
    Match,
    Rule,
    parse_code,
)
```

- [ ] **Step 6: Replace `common/transform/engine.py` with shim**

Write `wfcllm/common/transform/engine.py`:

```python
"""Deprecated import path. Use wfcllm.lang.python.transform.engine instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.engine is deprecated; use wfcllm.lang.python.transform.engine",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.engine import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.engine import TransformEngine  # noqa: E402, F401
```

- [ ] **Step 7: Run the shim tests + existing transform tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py tests/common/transform/test_base.py tests/common/transform/test_engine.py -v
```

Expected: all pass. (The shim emits `DeprecationWarning` but the existing tests don't filter on it.)

- [ ] **Step 8: Commit**

```bash
git add wfcllm/lang/python/transform/__init__.py \
        wfcllm/lang/python/transform/base.py \
        wfcllm/lang/python/transform/engine.py \
        wfcllm/common/transform/base.py \
        wfcllm/common/transform/engine.py \
        tests/integration/test_common_lang_shims.py
git commit -m "refactor: move common/transform/{base,engine}.py to lang/python/transform/ with shims"
```

---

## Task 7: Move `common/transform/positive/` → `lang/python/transform/positive/` (verbatim) with shims

**Files:**
- Create: `wfcllm/lang/python/transform/positive/__init__.py`
- Create: `wfcllm/lang/python/transform/positive/api_calls.py`
- Create: `wfcllm/lang/python/transform/positive/control_flow.py`
- Create: `wfcllm/lang/python/transform/positive/expression_logic.py`
- Create: `wfcllm/lang/python/transform/positive/formatting.py`
- Create: `wfcllm/lang/python/transform/positive/identifier.py`
- Create: `wfcllm/lang/python/transform/positive/syntax_init.py`
- Modify: each old `wfcllm/common/transform/positive/<file>.py` → shim
- Modify: `tests/integration/test_common_lang_shims.py` (append shim tests)

- [ ] **Step 1: Copy each positive rule file verbatim, rewriting intra-package imports**

Run:

```bash
mkdir -p wfcllm/lang/python/transform/positive
for f in api_calls control_flow expression_logic formatting identifier syntax_init; do
  cp wfcllm/common/transform/positive/$f.py wfcllm/lang/python/transform/positive/$f.py
done
```

Each file imports `from wfcllm.common.transform.base import ...`. Rewrite all six with `sed` then verify:

```bash
sed -i 's|from wfcllm.common.transform.base|from wfcllm.lang.python.transform.base|g' \
    wfcllm/lang/python/transform/positive/*.py

grep -rn "wfcllm.common" wfcllm/lang/python/transform/positive/
```

Expected: no output from the grep.

- [ ] **Step 2: Create the positive `__init__.py` (verbatim aggregator)**

Write `wfcllm/lang/python/transform/positive/__init__.py` verbatim from the old one with `wfcllm.common.transform` → `wfcllm.lang.python.transform`:

```python
"""Positive (semantic-equivalent) transformation rules registry."""

from wfcllm.lang.python.transform.base import Rule
from wfcllm.lang.python.transform.positive.api_calls import (
    ExplicitDefaultPrint, ExplicitDefaultRange, ExplicitDefaultOpen,
    ExplicitDefaultSorted, ExplicitDefaultMinMax, ExplicitDefaultZip,
    ExplicitDefaultRandomSeed, ExplicitDefaultHtmlEscape,
    ExplicitDefaultRound, ExplicitDefaultJsonDump,
    LibraryAliasReplace, ThirdPartyFuncReplace,
)
from wfcllm.lang.python.transform.positive.syntax_init import ListInit, DictInit, TypeCheck, StringFormat
from wfcllm.lang.python.transform.positive.control_flow import LoopConvert, IterationConvert, ComprehensionConvert, BranchFlip
from wfcllm.lang.python.transform.positive.expression_logic import (
    OperandSwap, ComparisonFlip, UnarySimplify, DeMorgan, ArithmeticAssociativity,
)
from wfcllm.lang.python.transform.positive.identifier import VariableRename, NameObfuscation
from wfcllm.lang.python.transform.positive.formatting import FixSpacing, FixCommentSymbols

_ALL_POSITIVE_RULES: list[Rule] = [
    ExplicitDefaultPrint(), ExplicitDefaultRange(), ExplicitDefaultOpen(),
    ExplicitDefaultSorted(), ExplicitDefaultMinMax(), ExplicitDefaultZip(),
    ExplicitDefaultRandomSeed(), ExplicitDefaultHtmlEscape(),
    ExplicitDefaultRound(), ExplicitDefaultJsonDump(),
    LibraryAliasReplace(), ThirdPartyFuncReplace(),
    ListInit(), DictInit(), TypeCheck(), StringFormat(),
    LoopConvert(), IterationConvert(), ComprehensionConvert(), BranchFlip(),
    OperandSwap(), ComparisonFlip(), UnarySimplify(), DeMorgan(), ArithmeticAssociativity(),
    VariableRename(), NameObfuscation(),
    FixSpacing(), FixCommentSymbols(),
]


def get_all_positive_rules() -> list[Rule]:
    return list(_ALL_POSITIVE_RULES)
```

- [ ] **Step 3: Sanity-check the new module imports + has the right count**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "from wfcllm.lang.python.transform.positive import get_all_positive_rules; rules = get_all_positive_rules(); print(len(rules))"
```

Expected: `29` (cross-check by counting entries in `_ALL_POSITIVE_RULES` — must match exactly).

- [ ] **Step 4: Append a shim-equivalence test for the rules**

Append to `tests/integration/test_common_lang_shims.py`:

```python
def test_old_positive_rules_symbol_identity():
    """Every public rule class re-exported from the old positive aggregator must be identity-equal to the new one."""
    import wfcllm.common.transform.positive as old
    import wfcllm.lang.python.transform.positive as new
    assert old.get_all_positive_rules.__code__ is new.get_all_positive_rules.__code__ \
        or old.get_all_positive_rules is new.get_all_positive_rules
    # Spot-check a few representative classes — full equivalence is enforced by
    # tests/common/transform/test_positive_rules.py.
    for sym in ("VariableRename", "FixSpacing", "ListInit", "BranchFlip", "DeMorgan"):
        assert getattr(old, sym) is getattr(new, sym), sym


def test_old_positive_rules_count_equals_new():
    import wfcllm.common.transform.positive as old
    import wfcllm.lang.python.transform.positive as new
    assert len(old.get_all_positive_rules()) == len(new.get_all_positive_rules())
```

- [ ] **Step 5: Run to confirm the new tests fail before the shim is in place**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py::test_old_positive_rules_symbol_identity tests/integration/test_common_lang_shims.py::test_old_positive_rules_count_equals_new -v`

Expected: FAIL — duplicated definitions, identity check breaks.

- [ ] **Step 6: Replace each old positive file with a shim**

For each `wfcllm/common/transform/positive/<f>.py` (excluding `__init__.py`), overwrite with:

```python
"""Deprecated import path. Use wfcllm.lang.python.transform.positive.<f> instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.positive.<f> is deprecated; "
    "use wfcllm.lang.python.transform.positive.<f>",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.positive.<f> import *  # noqa: E402, F401, F403
```

Substitute `<f>` for each module name (`api_calls`, `control_flow`, `expression_logic`, `formatting`, `identifier`, `syntax_init`). No `import (X, Y, …)` named-list is needed for these files because the existing aggregator pulls names through `__init__.py`, and `from … import *` covers any direct callers.

- [ ] **Step 7: Replace `wfcllm/common/transform/positive/__init__.py` with a shim**

```python
"""Deprecated import path. Use wfcllm.lang.python.transform.positive instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.positive is deprecated; "
    "use wfcllm.lang.python.transform.positive",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.positive import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.positive import (  # noqa: E402, F401
    get_all_positive_rules,
)
```

- [ ] **Step 8: Run the shim tests + existing positive rule tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py tests/common/transform/test_positive_rules.py -v
```

Expected: all pass (modulo pre-existing baseline failures, which positive rule tests should not have — verify by line count).

- [ ] **Step 9: Commit**

```bash
git add wfcllm/lang/python/transform/positive/ wfcllm/common/transform/positive/ tests/integration/test_common_lang_shims.py
git commit -m "refactor: move common/transform/positive/ to lang/python/transform/positive/ with shims"
```

---

## Task 8: Move `common/transform/negative/` → `lang/python/transform/negative/` (verbatim) with shims

**Files:**
- Create: `wfcllm/lang/python/transform/negative/__init__.py`
- Create: `wfcllm/lang/python/transform/negative/api_calls.py`
- Create: `wfcllm/lang/python/transform/negative/control_flow.py`
- Create: `wfcllm/lang/python/transform/negative/data_structure.py`
- Create: `wfcllm/lang/python/transform/negative/exception.py`
- Create: `wfcllm/lang/python/transform/negative/expression_logic.py`
- Create: `wfcllm/lang/python/transform/negative/identifier.py`
- Create: `wfcllm/lang/python/transform/negative/system.py`
- Modify: each old `wfcllm/common/transform/negative/<file>.py` → shim
- Modify: `tests/integration/test_common_lang_shims.py` (append shim tests)

- [ ] **Step 1: Mechanical copy + sed**

```bash
mkdir -p wfcllm/lang/python/transform/negative
for f in api_calls control_flow data_structure exception expression_logic identifier system; do
  cp wfcllm/common/transform/negative/$f.py wfcllm/lang/python/transform/negative/$f.py
done
sed -i 's|from wfcllm.common.transform.base|from wfcllm.lang.python.transform.base|g' \
    wfcllm/lang/python/transform/negative/*.py
grep -rn "wfcllm.common" wfcllm/lang/python/transform/negative/
```

Expected: grep returns no output.

- [ ] **Step 2: Create the negative `__init__.py` aggregator** (verbatim from old with `common.transform` → `lang.python.transform`)

```python
"""Negative (semantic-breaking) transformation rules registry."""

from wfcllm.lang.python.transform.base import Rule
from wfcllm.lang.python.transform.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from wfcllm.lang.python.transform.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)
from wfcllm.lang.python.transform.negative.expression_logic import (
    EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip,
)
from wfcllm.lang.python.transform.negative.identifier import ScopeVarCorrupt
from wfcllm.lang.python.transform.negative.data_structure import SliceStepFlip, DictViewSwap
from wfcllm.lang.python.transform.negative.exception import ExceptionSwallow
from wfcllm.lang.python.transform.negative.system import SysExitFlip


def get_all_negative_rules() -> list[Rule]:
    return [
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
        EqNeqFlip(), ArithmeticOpReplace(), AndOrSwap(), BoundsNarrow(), AugAssignCorrupt(), ShiftFlip(),
        ScopeVarCorrupt(),
        SliceStepFlip(), DictViewSwap(),
        ExceptionSwallow(),
        SysExitFlip(),
    ]
```

- [ ] **Step 3: Append shim-equivalence tests**

Append to `tests/integration/test_common_lang_shims.py`:

```python
def test_old_negative_rules_symbol_identity():
    import wfcllm.common.transform.negative as old
    import wfcllm.lang.python.transform.negative as new
    for sym in ("MinMaxFlip", "OffByOne", "EqNeqFlip", "SliceStepFlip", "ExceptionSwallow", "SysExitFlip"):
        assert getattr(old, sym) is getattr(new, sym), sym


def test_old_negative_rules_count_equals_new():
    import wfcllm.common.transform.negative as old
    import wfcllm.lang.python.transform.negative as new
    assert len(old.get_all_negative_rules()) == len(new.get_all_negative_rules())
```

- [ ] **Step 4: Run to confirm the new tests fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py::test_old_negative_rules_symbol_identity tests/integration/test_common_lang_shims.py::test_old_negative_rules_count_equals_new -v`

Expected: FAIL.

- [ ] **Step 5: Replace each old negative file with a shim** (same template as Task 7 Step 6, substituting `negative.<f>`)

- [ ] **Step 6: Replace `wfcllm/common/transform/negative/__init__.py` with a shim**

```python
"""Deprecated import path. Use wfcllm.lang.python.transform.negative instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform.negative is deprecated; "
    "use wfcllm.lang.python.transform.negative",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform.negative import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform.negative import (  # noqa: E402, F401
    get_all_negative_rules,
)
```

- [ ] **Step 7: Run shim tests + existing negative rule tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py tests/common/transform/test_negative_rules.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add wfcllm/lang/python/transform/negative/ wfcllm/common/transform/negative/ tests/integration/test_common_lang_shims.py
git commit -m "refactor: move common/transform/negative/ to lang/python/transform/negative/ with shims"
```

---

## Task 9: Collapse `common/transform/__init__.py` to a shim

**Files:**
- Modify: `wfcllm/common/transform/__init__.py` (replace with shim)
- Modify: `tests/integration/test_common_lang_shims.py` (append shim test)

- [ ] **Step 1: Append shim-equivalence test for the package-level aggregator**

```python
def test_old_transform_package_symbol_identity():
    import wfcllm.common.transform as old
    import wfcllm.lang.python.transform as new
    assert old.Rule is new.Rule
    assert old.Match is new.Match
    assert old.parse_code is new.parse_code
    assert old.TransformEngine is new.TransformEngine
```

- [ ] **Step 2: Run — expect FAIL**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py::test_old_transform_package_symbol_identity -v`

Expected: FAIL.

- [ ] **Step 3: Replace `wfcllm/common/transform/__init__.py` with shim**

```python
"""Deprecated import path. Use wfcllm.lang.python.transform instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.transform is deprecated; use wfcllm.lang.python.transform",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.transform import *  # noqa: E402, F401, F403
from wfcllm.lang.python.transform import (  # noqa: E402, F401
    Match,
    Rule,
    TransformEngine,
    parse_code,
)
```

- [ ] **Step 4: Run shim tests + existing top-level transform tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py tests/common/transform/ -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/common/transform/__init__.py tests/integration/test_common_lang_shims.py
git commit -m "refactor: collapse common/transform/__init__.py to lang/python/transform shim"
```

---

## Task 10: Scaffold `wfcllm/datasets/` package + `DatasetAdapter` ABC + first failing test

**Files:**
- Create: `wfcllm/datasets/__init__.py`
- Create: `wfcllm/datasets/adapter.py`
- Create: `wfcllm/datasets/registry.py`
- Create: `tests/datasets/__init__.py`
- Create: `tests/datasets/test_adapters.py`

- [ ] **Step 1: Create `wfcllm/datasets/adapter.py`**

```python
"""Dataset adapter ABC (spec §5.2)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class CodeSample:
    """A code-generation benchmark sample, normalised across datasets."""

    task_id: str
    prompt: str
    canonical_solution: str
    language: str
    metadata: dict = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Pluggable per-dataset strategy for loading code-generation benchmarks."""

    name: str = ""
    supported_languages: tuple[str, ...] = ()

    @abstractmethod
    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]: ...

    @abstractmethod
    def get_sample(self, task_id: str) -> CodeSample: ...

    def task_ids(self, language: str | None = None) -> list[str]:
        return [s.task_id for s in self.iter_samples(language)]

    def supports(self, language: str) -> bool:
        return language in self.supported_languages
```

- [ ] **Step 2: Create `wfcllm/datasets/registry.py`**

```python
"""Dataset-adapter registry (module-level singleton)."""
from __future__ import annotations

from wfcllm.common.registry import Registry
from wfcllm.datasets.adapter import DatasetAdapter

_REGISTRY: Registry[DatasetAdapter] = Registry("dataset")


def register(name: str):
    def deco(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        _REGISTRY.register(name, cls)
        return cls
    return deco


def get(name: str) -> DatasetAdapter:
    return _REGISTRY.get(name)


def names() -> list[str]:
    return _REGISTRY.names()
```

- [ ] **Step 3: Create `wfcllm/datasets/__init__.py`** (registers no adapters yet — done in Tasks 11–13)

```python
"""WFCLLM dataset-adapter layer (spec §5.2)."""
from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import get, names, register

__all__ = ["CodeSample", "DatasetAdapter", "get", "names", "register"]
```

- [ ] **Step 4: Write the failing test**

Write `tests/datasets/__init__.py` as an empty file.

Write `tests/datasets/test_adapters.py`:

```python
"""Tests for DatasetAdapter implementations and registry."""
from __future__ import annotations

import pytest


def test_humaneval_registered():
    from wfcllm import datasets
    assert "humaneval" in datasets.names()


def test_mbpp_registered():
    from wfcllm import datasets
    assert "mbpp" in datasets.names()


def test_humanevalpack_registered():
    from wfcllm import datasets
    assert "humanevalpack" in datasets.names()


def test_unknown_dataset_raises():
    from wfcllm import datasets
    with pytest.raises(KeyError, match="unknown dataset"):
        datasets.get("does-not-exist")
```

- [ ] **Step 5: Run to confirm 3 of 4 fail (none registered yet)**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/datasets/test_adapters.py -v`

Expected: `test_unknown_dataset_raises` passes; the three "registered" tests FAIL.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/datasets/__init__.py wfcllm/datasets/adapter.py wfcllm/datasets/registry.py \
        tests/datasets/__init__.py tests/datasets/test_adapters.py
git commit -m "feat: scaffold wfcllm/datasets package with DatasetAdapter ABC + registry"
```

---

## Task 11: Move `common/dataset_loader.py` → `datasets/loaders/local.py` (verbatim) with shim

**Files:**
- Create: `wfcllm/datasets/loaders/__init__.py`
- Create: `wfcllm/datasets/loaders/local.py`
- Modify: `wfcllm/common/dataset_loader.py` (replace with shim)
- Modify: `tests/integration/test_common_lang_shims.py` (append shim tests)

- [ ] **Step 1: Copy `dataset_loader.py` verbatim**

```bash
mkdir -p wfcllm/datasets/loaders
echo '"""Concrete loader functions backing the DatasetAdapter implementations."""' \
    > wfcllm/datasets/loaders/__init__.py
cp wfcllm/common/dataset_loader.py wfcllm/datasets/loaders/local.py
grep -n "wfcllm" wfcllm/datasets/loaders/local.py
```

Expected: no output (the file has no intra-package imports).

- [ ] **Step 2: Append shim test**

Append to `tests/integration/test_common_lang_shims.py`:

```python
def test_old_dataset_loader_symbol_identity():
    import wfcllm.common.dataset_loader as old
    import wfcllm.datasets.loaders.local as new
    assert old.load_prompts is new.load_prompts
    assert old.load_reference_solutions is new.load_reference_solutions
    assert old.SUPPORTED_DATASETS == new.SUPPORTED_DATASETS
```

- [ ] **Step 3: Run — expect FAIL**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py::test_old_dataset_loader_symbol_identity -v`

Expected: FAIL (duplicated definitions).

- [ ] **Step 4: Replace `wfcllm/common/dataset_loader.py` with a shim**

```python
"""Deprecated import path. Use wfcllm.datasets.loaders.local instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.dataset_loader is deprecated; use wfcllm.datasets.loaders.local",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.datasets.loaders.local import *  # noqa: E402, F401, F403
from wfcllm.datasets.loaders.local import (  # noqa: E402, F401
    SUPPORTED_DATASETS,
    load_prompts,
    load_reference_solutions,
)
```

- [ ] **Step 5: Check `wfcllm/common/__init__.py`**

Currently:

```python
from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts
```

Switch this single import line to the canonical path (the shim still works for outside callers; this is a first-party fix):

```python
from wfcllm.datasets.loaders.local import SUPPORTED_DATASETS, load_prompts  # noqa: F401
```

- [ ] **Step 6: Run the shim tests + existing dataset_loader tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_common_lang_shims.py tests/common/test_dataset_loader.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add wfcllm/datasets/loaders/__init__.py wfcllm/datasets/loaders/local.py \
        wfcllm/common/dataset_loader.py wfcllm/common/__init__.py \
        tests/integration/test_common_lang_shims.py
git commit -m "refactor: move common/dataset_loader.py to datasets/loaders/local.py with shim"
```

---

## Task 12: Implement `HumanEvalAdapter` + `MBPPAdapter`

**Files:**
- Create: `wfcllm/datasets/humaneval.py`
- Create: `wfcllm/datasets/mbpp.py`
- Modify: `wfcllm/datasets/__init__.py` (trigger registration via side-effect imports)
- Modify: `tests/datasets/test_adapters.py` (append behaviour tests)

- [ ] **Step 1: Create `wfcllm/datasets/humaneval.py`**

```python
"""HumanEvalAdapter — wraps wfcllm.datasets.loaders.local.load_prompts."""
from __future__ import annotations

from typing import Iterator

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.loaders.local import load_prompts, load_reference_solutions
from wfcllm.datasets.registry import register


@register("humaneval")
class HumanEvalAdapter(DatasetAdapter):
    name = "humaneval"
    supported_languages: tuple[str, ...] = ("python",)

    def __init__(self, dataset_path: str = "data/datasets") -> None:
        self._dataset_path = dataset_path

    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]:
        if language is not None and language not in self.supported_languages:
            raise ValueError(
                f"HumanEvalAdapter does not support language={language!r}; "
                f"supported={self.supported_languages}"
            )
        rows = load_reference_solutions("humaneval", self._dataset_path)
        for row in rows:
            yield CodeSample(
                task_id=row["id"],
                prompt=row["prompt"],
                canonical_solution=row["generated_code"],
                language="python",
                metadata={},
            )

    def get_sample(self, task_id: str) -> CodeSample:
        for sample in self.iter_samples():
            if sample.task_id == task_id:
                return sample
        raise KeyError(f"HumanEval task_id not found: {task_id!r}")
```

- [ ] **Step 2: Create `wfcllm/datasets/mbpp.py`** (same shape, `name = "mbpp"`)

```python
"""MBPPAdapter — wraps wfcllm.datasets.loaders.local.load_prompts."""
from __future__ import annotations

from typing import Iterator

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.loaders.local import load_reference_solutions
from wfcllm.datasets.registry import register


@register("mbpp")
class MBPPAdapter(DatasetAdapter):
    name = "mbpp"
    supported_languages: tuple[str, ...] = ("python",)

    def __init__(self, dataset_path: str = "data/datasets") -> None:
        self._dataset_path = dataset_path

    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]:
        if language is not None and language not in self.supported_languages:
            raise ValueError(
                f"MBPPAdapter does not support language={language!r}; "
                f"supported={self.supported_languages}"
            )
        rows = load_reference_solutions("mbpp", self._dataset_path)
        for row in rows:
            yield CodeSample(
                task_id=row["id"],
                prompt=row["prompt"],
                canonical_solution=row["generated_code"],
                language="python",
                metadata={},
            )

    def get_sample(self, task_id: str) -> CodeSample:
        for sample in self.iter_samples():
            if sample.task_id == task_id:
                return sample
        raise KeyError(f"MBPP task_id not found: {task_id!r}")
```

- [ ] **Step 3: Trigger registration from `wfcllm/datasets/__init__.py`**

Add at the end (after the `__all__` line):

```python
# Side-effect imports trigger @register decorators
from wfcllm.datasets import humaneval as _humaneval  # noqa: F401, E402
from wfcllm.datasets import mbpp as _mbpp  # noqa: F401, E402
```

- [ ] **Step 4: Append behaviour tests**

Append to `tests/datasets/test_adapters.py`:

```python
def test_humaneval_adapter_supports_python_only():
    from wfcllm import datasets
    adapter = datasets.get("humaneval")
    assert adapter.supports("python")
    assert not adapter.supports("java")


def test_humaneval_adapter_rejects_unsupported_language_at_iter():
    from wfcllm import datasets
    adapter = datasets.get("humaneval")
    with pytest.raises(ValueError, match="java"):
        next(adapter.iter_samples(language="java"))


def test_mbpp_adapter_name():
    from wfcllm import datasets
    assert datasets.get("mbpp").name == "mbpp"
```

(Round-trip tests against the real dataset cache are intentionally NOT included — they require local data files; the contract is already exercised by `tests/common/test_dataset_loader.py`.)

- [ ] **Step 5: Run the dataset adapter tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/datasets/test_adapters.py -v`

Expected: all pass (6 tests now — the original 4 plus 3 new — minus the now-redundant pieces, however many sum out).

- [ ] **Step 6: Commit**

```bash
git add wfcllm/datasets/humaneval.py wfcllm/datasets/mbpp.py wfcllm/datasets/__init__.py \
        tests/datasets/test_adapters.py
git commit -m "feat: add HumanEvalAdapter and MBPPAdapter (DatasetAdapter implementations)"
```

---

## Task 13: Implement `HumanEvalPackAdapter` skeleton

**Files:**
- Create: `wfcllm/datasets/humanevalpack.py`
- Modify: `wfcllm/datasets/__init__.py` (side-effect import)
- Modify: `tests/datasets/test_adapters.py` (append NotImplementedError test)

- [ ] **Step 1: Create `wfcllm/datasets/humanevalpack.py`** (spec §5.2 specifies the exact error message)

```python
"""HumanEvalPackAdapter — skeleton-only; spec §5.2 defers data loading to a multi-language phase."""
from __future__ import annotations

from typing import Iterator

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import register


@register("humanevalpack")
class HumanEvalPackAdapter(DatasetAdapter):
    name = "humanevalpack"
    supported_languages: tuple[str, ...] = ("python", "java", "cpp", "js")

    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]:
        raise NotImplementedError(
            "HumanEvalPack Adapter 留作接口；多语言阶段实现"
        )

    def get_sample(self, task_id: str) -> CodeSample:
        raise NotImplementedError(
            "HumanEvalPack Adapter 留作接口；多语言阶段实现"
        )
```

- [ ] **Step 2: Wire registration**

Append to `wfcllm/datasets/__init__.py`:

```python
from wfcllm.datasets import humanevalpack as _humanevalpack  # noqa: F401, E402
```

- [ ] **Step 3: Append test**

```python
def test_humanevalpack_iter_raises_not_implemented():
    from wfcllm import datasets
    adapter = datasets.get("humanevalpack")
    with pytest.raises(NotImplementedError, match="HumanEvalPack"):
        next(adapter.iter_samples())


def test_humanevalpack_supports_four_languages():
    from wfcllm import datasets
    adapter = datasets.get("humanevalpack")
    assert set(adapter.supported_languages) == {"python", "java", "cpp", "js"}
```

- [ ] **Step 4: Run tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/datasets/test_adapters.py -v`

Expected: all tests pass; the new two tests confirm the skeleton behaviour.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/datasets/humanevalpack.py wfcllm/datasets/__init__.py tests/datasets/test_adapters.py
git commit -m "feat: add HumanEvalPackAdapter skeleton (NotImplementedError per spec §5.2)"
```

---

## Task 14: First-party caller sanity grep + targeted regression run

**Files:** none modified — this is a verification task.

- [ ] **Step 1: Confirm no first-party file under `wfcllm/` has switched to canonical paths against plan**

Run:

```bash
grep -rn "from wfcllm.lang\|from wfcllm.datasets\." wfcllm/ 2>/dev/null | grep -v __pycache__
```

Expected output: only the files we created in this plan (`wfcllm/lang/*`, `wfcllm/datasets/*`, and `wfcllm/common/__init__.py:from wfcllm.datasets.loaders.local …` from Task 11 Step 5). Any caller under `wfcllm/encoder/`, `wfcllm/extract/`, `wfcllm/watermark/`, `wfcllm/evaluation/` must still be using the old `wfcllm.common.*` paths. If a caller has been changed by mistake, revert it.

- [ ] **Step 2: Run the targeted regression suite**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/lang/ \
  tests/datasets/ \
  tests/integration/test_common_lang_shims.py \
  tests/common/ \
  tests/encoder/ \
  tests/extract/ \
  tests/watermark/ \
  -v 2>&1 | tail -80
```

Expected: only the documented baseline failures (`tests/watermark/test_pipeline.py` ×6, `tests/extract/test_joint_detection.py` ×1). No new failures. (The 4 `test_run_config.py` baseline failures are not included in this run since we didn't select them; that suite stays clean of Phase 6 impact by construction.)

- [ ] **Step 3: Capture exact failure list for the verification artefact**

Pipe the summary into the commit body or report so the reviewer can diff against the baseline.

- [ ] **Step 4: Commit (none — verification only)**

No commit needed unless verification turned up a regression. If a regression appeared, STOP and report per the project's "测试失败立刻停下来报告" rule.

---

## Task 15: Final smoke test — `run.py --status` + import smoke

**Files:** none modified — this is a verification task.

- [ ] **Step 1: Confirm `run.py --status` still works**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py --status
```

Expected: clean status table; no `ModuleNotFoundError`, no `DeprecationWarning` printed (warnings are not raised to surface in the default filter when imports are eager — they fire once per location).

- [ ] **Step 2: Import smoke — canonical paths**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "
from wfcllm import lang, datasets
from wfcllm.lang.python.parser import PythonParser, extract_statement_blocks
from wfcllm.lang.python.transform import TransformEngine, Rule
from wfcllm.lang.python.transform.positive import get_all_positive_rules
from wfcllm.lang.python.transform.negative import get_all_negative_rules
from wfcllm.datasets.loaders.local import load_prompts, load_reference_solutions
print('lang names:', lang.names())
print('dataset names:', datasets.names())
print('positive rule count:', len(get_all_positive_rules()))
print('negative rule count:', len(get_all_negative_rules()))
"
```

Expected: `lang names: ['python']`, `dataset names: ['humaneval', 'humanevalpack', 'mbpp']` (sorted), positive count = `29`, negative count = `23` (cross-checked against the new aggregator entry counts).

- [ ] **Step 3: Import smoke — deprecated paths still work**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -W error::DeprecationWarning -c "
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
from wfcllm.common.ast_parser import PythonParser, extract_statement_blocks
from wfcllm.common.transform.engine import TransformEngine
from wfcllm.common.transform.positive import get_all_positive_rules
from wfcllm.common.transform.negative import get_all_negative_rules
from wfcllm.common.dataset_loader import load_prompts, load_reference_solutions
print('ok')
"
```

Expected: `ok` (with the warning filter disabling the `-W error::DeprecationWarning` for our `warnings.warn(...)` calls). If `-W error::DeprecationWarning` is not silenced and the imports succeed, that also confirms shim behaviour at runtime — leave the filter off and verify the prints succeed.

- [ ] **Step 4: Commit (none — verification only)**

No new commit. The Phase 6 chain is complete at this point.

---

## Phase 6 done-criteria checklist

- [ ] All 13 commits in Phase 6 land on `main` with `<type>: <description>` messages.
- [ ] `pytest tests/lang/ tests/datasets/ tests/integration/test_common_lang_shims.py` is green.
- [ ] `pytest tests/common/ tests/encoder/ tests/extract/ tests/watermark/` shows the exact pre-existing baseline failure list — no new failures.
- [ ] `run.py --status` runs cleanly.
- [ ] First-party callers under `wfcllm/encoder/`, `wfcllm/extract/`, `wfcllm/watermark/`, `wfcllm/evaluation/` are **unchanged** (still imports from the old `wfcllm.common.*` paths) — Task 14 Step 1 grep matches expectation.
- [ ] No file under `experiment/` or `tools/` is touched.
- [ ] No algorithmic change: the verbatim file copies + sed rewrites in Tasks 6–8, 11 are the only delta to the bodies of moved files.
- [ ] State-dict / weight files are unaffected — Phase 6 never touches `wfcllm/encoder/` or `wfcllm/watermark/token_channel/core/model.py`.
