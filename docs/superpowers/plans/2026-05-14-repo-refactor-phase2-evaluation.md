# Repo Refactor Phase 2: Evaluation Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the three scattered evaluation entry points (`wfcllm/common/offline_code_eval.py`, `wfcllm/extract/offline_analysis.py`, `scripts/evaluate_dual_channel.py`) under a new top-level `wfcllm/evaluation/` package, leave deprecation shims at the old paths, and expose a unified `scripts/evaluate.py exec|detection|dual` CLI.

**Architecture:** Pure code-organization refactor — no algorithm changes. Each old file is copied verbatim into its new location (rename only). Old paths become 4-line shims that emit `DeprecationWarning` and re-export via `from <new> import *`. `scripts/evaluate_dual_channel.py` keeps its shape (subprocess driver) but its core moves into `wfcllm/evaluation/dual_channel.py`; the script file collapses to a thin shim plus path bootstrap so legacy invocations keep working. `scripts/evaluate.py` is the new umbrella entry: three argparse subcommands dispatch into `wfcllm.evaluation.code_execution`, `wfcllm.evaluation.detection_report`, and `wfcllm.evaluation.dual_channel`.

**Tech Stack:** Python 3.11, pytest, argparse, `warnings` stdlib. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §4.4, §8.2, §10 step 2.
**Builds on:** `docs/superpowers/plans/2026-05-14-repo-refactor-phase1-infrastructure.md` (already landed: `wfcllm/common/registry.py`, `wfcllm/orchestration/*`, `wfcllm/cli/*`).

**Pre-flight:**

1. On a clean working tree (Phase 1 already merged to `main`).
2. Confirm baseline is green:

   ```
   HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_offline_analysis.py tests/extract/test_joint_detection.py tests/test_run.py -v
   ```

   Expected: all pass. These are the "before" suites that this phase must preserve.

3. Snapshot the line counts you're working with:

   ```
   wc -l wfcllm/common/offline_code_eval.py wfcllm/extract/offline_analysis.py scripts/evaluate_dual_channel.py
   ```

   Expected: 360 / 687 / 483 lines respectively. After this plan, each old path is ≤10 lines.

4. Confirm the current callers of these modules:

   ```
   grep -rn "offline_code_eval\|offline_analysis\|evaluate_dual_channel" --include="*.py"
   ```

   You should see hits in: `wfcllm/cli/runners.py`, `scripts/evaluate_dual_channel.py`, `tests/test_run.py`, `tests/extract/test_offline_analysis.py`, `tests/extract/test_joint_detection.py`. Phase 2 updates the first two; tests stay on old paths and are exercised through the shim (a dedicated shim test confirms this works).

---

## Task 1: `wfcllm/evaluation/` package skeleton

**Files:**
- Create: `wfcllm/evaluation/__init__.py`

Empty package marker so subsequent tasks can import.

- [ ] **Step 1: Create the directory and `__init__.py`**

```
mkdir -p wfcllm/evaluation
```

Create `wfcllm/evaluation/__init__.py` with this single line:

```python
"""Cross-phase evaluation: pass@k, ROC/AUC, detection regression, dual-channel sweep."""
```

- [ ] **Step 2: Verify the package is importable**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "import wfcllm.evaluation; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```
git add wfcllm/evaluation/__init__.py
git commit -m "chore: add wfcllm/evaluation/ package skeleton"
```

---

## Task 2: Move `offline_code_eval.py` → `evaluation/code_execution.py`

**Files:**
- Create: `wfcllm/evaluation/code_execution.py` (copied verbatim from `wfcllm/common/offline_code_eval.py`)
- Modify: `wfcllm/common/offline_code_eval.py` → becomes deprecation shim
- Create: `tests/integration/test_evaluation_shims.py`

Source file is 360 lines: 13 public functions (`load_jsonl_records`, `write_jsonl_records`, `annotate_correctness_from_references`, `compute_pass_at_k`, `compute_retry_rate`, `compute_average_latency`, `compute_roc_auc`, `compute_tpr_at_fpr`, `mode_score_field`, `extract_scores`, `mean_or_zero`, `relative_delta`, `build_perturbation_corpus`, `apply_perturbation`, `normalize_code`) + several private helpers. Pure-functional module; no internal `wfcllm.*` imports. Copy is byte-identical.

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_evaluation_shims.py`:

```python
"""Tests for wfcllm/evaluation/* migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import warnings

import pytest


# --- code_execution: new path works ---

def test_code_execution_new_path_importable_and_callable():
    from wfcllm.evaluation.code_execution import (
        compute_pass_at_k,
        compute_roc_auc,
        compute_tpr_at_fpr,
        load_jsonl_records,
        write_jsonl_records,
    )

    records = [
        {"task_id": "t1", "is_correct": True},
        {"task_id": "t1", "is_correct": False},
        {"task_id": "t2", "is_correct": True},
    ]
    assert compute_pass_at_k(records, k=1) == pytest.approx(0.75)


# --- code_execution: old path is deprecated shim ---

def test_offline_code_eval_old_path_emits_deprecation_warning():
    """Old wfcllm.common.offline_code_eval should warn but still expose public API."""
    # Force a fresh import so the warning fires deterministically.
    import importlib
    import sys
    sys.modules.pop("wfcllm.common.offline_code_eval", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.common.offline_code_eval")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.evaluation.code_execution" in str(w.message)
        for w in caught
    )
    # Public symbols are still resolvable from the old path.
    assert callable(module.compute_pass_at_k)
    assert callable(module.compute_roc_auc)
    assert callable(module.annotate_correctness_from_references)


def test_offline_code_eval_old_and_new_paths_share_symbols():
    """Symbols re-exported from the shim refer to the same callable as the new path."""
    import importlib
    import sys
    sys.modules.pop("wfcllm.common.offline_code_eval", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.common.offline_code_eval")
    new = importlib.import_module("wfcllm.evaluation.code_execution")
    for name in (
        "compute_pass_at_k",
        "compute_roc_auc",
        "annotate_correctness_from_references",
        "load_jsonl_records",
        "write_jsonl_records",
    ):
        assert getattr(old, name) is getattr(new, name), name
```

- [ ] **Step 2: Run tests to verify they fail**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluation_shims.py -v
```

Expected: 3 tests fail with `ModuleNotFoundError: No module named 'wfcllm.evaluation.code_execution'`.

- [ ] **Step 3: Copy the file content verbatim**

Use `git mv` to preserve history, then we'll replace the old path's body with a shim:

```
git mv wfcllm/common/offline_code_eval.py wfcllm/evaluation/code_execution.py
```

**Important — do not modify any function body, docstring, or import.** The file is pure-functional and has no `wfcllm.*` imports, so byte-identical relocation is correct.

Update the top-line docstring of `wfcllm/evaluation/code_execution.py` from:

```python
"""Offline evaluation helpers for dual-channel code-watermark experiments."""
```

to:

```python
"""Offline evaluation helpers for dual-channel code-watermark experiments.

(Phase 2 refactor: moved from wfcllm.common.offline_code_eval.)
"""
```

No other edits.

- [ ] **Step 4: Write the shim at the old path**

Create `wfcllm/common/offline_code_eval.py` with exactly:

```python
"""Deprecated: re-exports for backwards compatibility.

Moved to ``wfcllm.evaluation.code_execution`` in the Phase 2 refactor
(see docs/superpowers/specs/2026-05-14-repo-refactor-design.md §4.4, §8.2).
"""
import warnings

warnings.warn(
    "wfcllm.common.offline_code_eval is deprecated; "
    "use wfcllm.evaluation.code_execution instead.",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.evaluation.code_execution import *  # noqa: F401,F403,E402
```

- [ ] **Step 5: Run the new shim tests**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluation_shims.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Run the existing test suites that use the old path**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_joint_detection.py -v
```

Expected: all pass. The tests import from `wfcllm.common.offline_code_eval` and now go through the shim — symbols still resolve, but a `DeprecationWarning` is emitted (pytest will show `PytestUnraisableExceptionWarning`-style summaries; that is expected and benign for this transition).

- [ ] **Step 7: Commit**

```
git add wfcllm/evaluation/code_execution.py wfcllm/common/offline_code_eval.py tests/integration/test_evaluation_shims.py
git commit -m "refactor: move offline_code_eval into evaluation.code_execution + shim old path"
```

---

## Task 3: Move `offline_analysis.py` → `evaluation/detection_report.py`

**Files:**
- Create: `wfcllm/evaluation/detection_report.py` (copied verbatim from `wfcllm/extract/offline_analysis.py`)
- Modify: `wfcllm/extract/offline_analysis.py` → becomes deprecation shim
- Modify: `tests/integration/test_evaluation_shims.py` (append tests)

Source file is 687 lines: 9 public dataclasses (`SummaryArtifact`, `DetailArtifact`, `WatermarkedArtifact`, `RunParameterComparison`, `ArtifactCompatibility`, `DetailDelta`, `DetailDeltaReport`, `EmbeddingDelta`) + 8 public functions (`load_summary_artifact`, `load_detail_artifact`, `load_watermarked_artifact`, `compare_run_parameters`, `check_artifact_compatibility`, `build_detail_delta_report`, `build_offline_regression_report`, `write_offline_regression_report`) + private helpers. No internal `wfcllm.*` imports. Byte-identical relocation.

- [ ] **Step 1: Append failing tests**

Append to `tests/integration/test_evaluation_shims.py`:

```python
# --- detection_report: new path works ---

def test_detection_report_new_path_importable_and_callable():
    from wfcllm.evaluation.detection_report import (
        build_offline_regression_report,
        load_detail_artifact,
        load_summary_artifact,
        load_watermarked_artifact,
        write_offline_regression_report,
    )

    assert callable(build_offline_regression_report)
    assert callable(load_detail_artifact)


# --- detection_report: old path is deprecated shim ---

def test_offline_analysis_old_path_emits_deprecation_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.extract.offline_analysis", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.extract.offline_analysis")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.evaluation.detection_report" in str(w.message)
        for w in caught
    )
    assert callable(module.build_offline_regression_report)
    assert callable(module.load_summary_artifact)


def test_offline_analysis_old_and_new_paths_share_symbols():
    import importlib
    import sys
    sys.modules.pop("wfcllm.extract.offline_analysis", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.extract.offline_analysis")
    new = importlib.import_module("wfcllm.evaluation.detection_report")
    for name in (
        "build_offline_regression_report",
        "load_summary_artifact",
        "load_detail_artifact",
        "load_watermarked_artifact",
        "write_offline_regression_report",
        "SummaryArtifact",
        "DetailArtifact",
        "WatermarkedArtifact",
        "DetailDelta",
        "DetailDeltaReport",
    ):
        assert getattr(old, name) is getattr(new, name), name
```

- [ ] **Step 2: Run tests to verify they fail**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluation_shims.py -v
```

Expected: 3 new tests fail with `ModuleNotFoundError: No module named 'wfcllm.evaluation.detection_report'`.

- [ ] **Step 3: Copy the file content verbatim**

```
git mv wfcllm/extract/offline_analysis.py wfcllm/evaluation/detection_report.py
```

Update only the top-line docstring of `wfcllm/evaluation/detection_report.py` from:

```python
"""Offline helpers for comparing saved watermark extraction artifacts."""
```

to:

```python
"""Offline helpers for comparing saved watermark extraction artifacts.

(Phase 2 refactor: moved from wfcllm.extract.offline_analysis.)
"""
```

No other edits.

- [ ] **Step 4: Write the shim at the old path**

Create `wfcllm/extract/offline_analysis.py` with exactly:

```python
"""Deprecated: re-exports for backwards compatibility.

Moved to ``wfcllm.evaluation.detection_report`` in the Phase 2 refactor
(see docs/superpowers/specs/2026-05-14-repo-refactor-design.md §4.4, §8.2).
"""
import warnings

warnings.warn(
    "wfcllm.extract.offline_analysis is deprecated; "
    "use wfcllm.evaluation.detection_report instead.",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.evaluation.detection_report import *  # noqa: F401,F403,E402
```

- [ ] **Step 5: Run the new shim tests**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluation_shims.py -v
```

Expected: 6 tests pass total (3 from Task 2 + 3 new).

- [ ] **Step 6: Run the existing test suites that use the old path**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_offline_analysis.py -v
```

Expected: all pass. These tests load the module via `importlib.import_module("wfcllm.extract.offline_analysis")` so the shim path is exercised end-to-end (DeprecationWarning emitted; symbols resolved).

- [ ] **Step 7: Commit**

```
git add wfcllm/evaluation/detection_report.py wfcllm/extract/offline_analysis.py tests/integration/test_evaluation_shims.py
git commit -m "refactor: move offline_analysis into evaluation.detection_report + shim old path"
```

---

## Task 4: Update `wfcllm/cli/runners.py` to import from the new path

**Files:**
- Modify: `wfcllm/cli/runners.py:330-337`

The only first-party non-test caller of `wfcllm.extract.offline_analysis` is `run_offline_analysis` inside `wfcllm/cli/runners.py`. Update it to use the canonical new path so internal code does not trigger its own deprecation warning. Test files keep their old-path imports (those exercise the shim — that's intentional and removed in the final shim-cleanup commit, not here).

- [ ] **Step 1: Locate the import in `wfcllm/cli/runners.py`**

```
grep -n "offline_analysis" wfcllm/cli/runners.py
```

Expected: one hit around line 331 inside `run_offline_analysis`.

- [ ] **Step 2: Replace the import**

Open `wfcllm/cli/runners.py`. In the function `run_offline_analysis`, replace:

```python
def run_offline_analysis(args: argparse.Namespace) -> int:
    from wfcllm.extract.offline_analysis import (
        build_offline_regression_report,
        load_detail_artifact,
        load_summary_artifact,
        load_watermarked_artifact,
        write_offline_regression_report,
    )
```

with:

```python
def run_offline_analysis(args: argparse.Namespace) -> int:
    from wfcllm.evaluation.detection_report import (
        build_offline_regression_report,
        load_detail_artifact,
        load_summary_artifact,
        load_watermarked_artifact,
        write_offline_regression_report,
    )
```

(Only the module path on the `from ... import` line changes; the imported names are identical.)

- [ ] **Step 3: Run the affected tests**

The tests in `tests/test_run.py` use function-level definitions (e.g.
`test_run_offline_analysis_writes_json_report` at `tests/test_run.py:1286`),
not class-level grouping. Run them by keyword:

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -v -k "offline_analysis or compare"
```

Expected: all matching tests pass.

- [ ] **Step 4: Run the full integration suite to confirm no regression**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/ tests/test_run.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add wfcllm/cli/runners.py
git commit -m "refactor: route run_offline_analysis through evaluation.detection_report"
```

---

## Task 5: Move `evaluate_dual_channel.py` core → `wfcllm/evaluation/dual_channel.py`

**Files:**
- Create: `wfcllm/evaluation/dual_channel.py`
- Modify: `tests/integration/test_evaluation_shims.py` (append a smoke test)

`scripts/evaluate_dual_channel.py` is 483 lines. Map of top-level defs (verified):

```
line  45  class CommandRunResult
line  57  def run_evaluation
line 232  def build_argument_parser
line 253  def main
line 265  def _resolve_reference_records
line 279  def _ensure_negative_corpus
line 304  def _run_watermark_phase
line 333  def _run_extract_phase
line 360  def _run_perturbation_checks
line 397  def _build_threshold_report
line 438  def _mode_cli_args
line 448  def _newest_artifact
line 457  def _require_success
line 463  def _run_command
```

Plus module-level constants `MODES`, `PERTURBATIONS`, `TARGET_FPR`, type alias
`CommandRunner`, and a `REPO_ROOT = Path(__file__).resolve().parents[1]` line that
exists only to support the `sys.path` bootstrap. Inside the package, the constant
becomes `parents[2]` (since the new file is two levels deeper than scripts/).

Note: `scripts/evaluate_dual_channel.py:25` does `from wfcllm.common.dataset_loader
import load_reference_solutions`. That import stays as-is (Phase 6 will retarget
`dataset_loader`).

**Important — this is a copy, not a move.** `scripts/evaluate_dual_channel.py`
must continue to exist after this task (Task 6 collapses it to a shim that re-exports
from the new module). Use `cp`, not `git mv`.

- [ ] **Step 1: Append a failing smoke test**

Append to `tests/integration/test_evaluation_shims.py`:

```python
# --- dual_channel: new module ---

def test_dual_channel_new_module_exposes_public_api():
    from wfcllm.evaluation import dual_channel
    assert callable(dual_channel.run_evaluation)
    assert callable(dual_channel.build_argument_parser)
    assert callable(dual_channel.main)
    assert hasattr(dual_channel, "CommandRunResult")
    assert dual_channel.MODES == ("semantic-only", "lexical-only", "dual-channel")
    assert dual_channel.PERTURBATIONS == ("formatting", "comments", "rename", "light-rewrite")
    assert dual_channel.TARGET_FPR == pytest.approx(0.01)
```

- [ ] **Step 2: Run the test to confirm it fails**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluation_shims.py::test_dual_channel_new_module_exposes_public_api -v
```

Expected: `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 3: Copy file content into the new module**

Copy the source to the new location:

```
cp scripts/evaluate_dual_channel.py wfcllm/evaluation/dual_channel.py
```

Now apply exactly four surgical edits to `wfcllm/evaluation/dual_channel.py`:

**Edit 3a** — replace the module docstring at lines 1-5 of the copy:

```python
#!/usr/bin/env python
"""Offline evaluation harness for semantic, lexical, and dual-channel modes.

Correctness stays offline: pass@k is derived from generated code grouped per task and
matched against local reference solutions via normalized AST equality.
"""
```

with:

```python
"""Offline dual-channel evaluation harness for semantic, lexical, and dual-channel modes.

(Phase 2 refactor: moved from scripts/evaluate_dual_channel.py. Correctness stays offline:
pass@k is derived from generated code grouped per task and matched against local reference
solutions via normalized AST equality.)
"""
```

(Drop the `#!/usr/bin/env python` shebang — it's only meaningful for executable scripts.)

**Edit 3b** — delete the `sys.path` bootstrap at lines 21-23 of the copy:

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

There is no replacement; just remove the three lines plus the surrounding blank line.
The `sys` and `Path` imports stay (they're still used by other code in the file).

**Edit 3c** — retarget the 12 `from wfcllm.common.offline_code_eval import ...` lines
(lines 26-38 of the copy) to the new evaluation module:

```python
from wfcllm.common.offline_code_eval import annotate_correctness_from_references
from wfcllm.common.offline_code_eval import build_perturbation_corpus
from wfcllm.common.offline_code_eval import compute_average_latency
from wfcllm.common.offline_code_eval import compute_pass_at_k
from wfcllm.common.offline_code_eval import compute_retry_rate
from wfcllm.common.offline_code_eval import compute_roc_auc
from wfcllm.common.offline_code_eval import compute_tpr_at_fpr
from wfcllm.common.offline_code_eval import extract_scores
from wfcllm.common.offline_code_eval import load_jsonl_records
from wfcllm.common.offline_code_eval import mean_or_zero
from wfcllm.common.offline_code_eval import relative_delta
from wfcllm.common.offline_code_eval import write_jsonl_records
```

becomes:

```python
from wfcllm.evaluation.code_execution import annotate_correctness_from_references
from wfcllm.evaluation.code_execution import build_perturbation_corpus
from wfcllm.evaluation.code_execution import compute_average_latency
from wfcllm.evaluation.code_execution import compute_pass_at_k
from wfcllm.evaluation.code_execution import compute_retry_rate
from wfcllm.evaluation.code_execution import compute_roc_auc
from wfcllm.evaluation.code_execution import compute_tpr_at_fpr
from wfcllm.evaluation.code_execution import extract_scores
from wfcllm.evaluation.code_execution import load_jsonl_records
from wfcllm.evaluation.code_execution import mean_or_zero
from wfcllm.evaluation.code_execution import relative_delta
from wfcllm.evaluation.code_execution import write_jsonl_records
```

(Mechanical: same 12 names, same order; only the module path changes.)

The `from wfcllm.common.dataset_loader import load_reference_solutions` line at
the top of the import block stays exactly as it is.

**Edit 3d** — every other line (constants, class, all 13 function bodies) is
**byte-identical**. Do not modify anything else.

After the edits, the file should:
- start with the new module docstring
- have stdlib imports unchanged
- import `load_reference_solutions` from `wfcllm.common.dataset_loader` (unchanged)
- import 12 names from `wfcllm.evaluation.code_execution` (changed)
- contain `MODES`, `PERTURBATIONS`, `TARGET_FPR`, `CommandRunResult`, `CommandRunner`,
  `run_evaluation`, `build_argument_parser`, `main`, and all 9 underscore-prefixed
  helpers — each function body byte-identical to the original

- [ ] **Step 4: Verify the module imports cleanly and the smoke test passes**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "from wfcllm.evaluation import dual_channel; print(dual_channel.MODES)"
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluation_shims.py::test_dual_channel_new_module_exposes_public_api -v
```

Expected: prints `('semantic-only', 'lexical-only', 'dual-channel')`; smoke test passes.

- [ ] **Step 5: Spot-check function-body parity**

Pick one mid-sized helper and confirm it's identical between old and new:

```
diff <(sed -n '397,435p' scripts/evaluate_dual_channel.py) <(awk '/^def _build_threshold_report/,/^def _mode_cli_args/' wfcllm/evaluation/dual_channel.py | head -39)
```

Expected: empty diff (the function body is the same; the surrounding awk/sed window
isolates lines 397-435 of the original). If the diff shows differences, you typed
something instead of copying — fix it.

- [ ] **Step 6: Commit**

```
git add wfcllm/evaluation/dual_channel.py tests/integration/test_evaluation_shims.py
git commit -m "feat: add wfcllm/evaluation/dual_channel.py (moved from scripts/)"
```

---

## Task 6: Collapse `scripts/evaluate_dual_channel.py` to a thin shim

**Files:**
- Modify: `scripts/evaluate_dual_channel.py` (replace ~483 lines with a short shim)

The existing script is still importable from `tests/extract/test_joint_detection.py` (`from scripts.evaluate_dual_channel import CommandRunResult, run_evaluation`) and may still be invoked directly by users. Keep both contracts: re-export the public API, emit a deprecation warning, and keep `if __name__ == "__main__"` working when called as a script.

- [ ] **Step 1: Replace the entire file with the shim**

Open `scripts/evaluate_dual_channel.py` and replace its full contents with:

```python
#!/usr/bin/env python
"""Deprecated wrapper around wfcllm.evaluation.dual_channel.

Use ``scripts/evaluate.py dual ...`` (Phase 2 refactor) or import from
``wfcllm.evaluation.dual_channel`` directly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Repo-root path bootstrap so this script keeps working when invoked as
# ``python scripts/evaluate_dual_channel.py`` (i.e. without ``-m``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.warn(
    "scripts/evaluate_dual_channel.py is deprecated; "
    "use 'python scripts/evaluate.py dual ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.evaluation.dual_channel import (  # noqa: E402,F401
    CommandRunResult,
    CommandRunner,
    MODES,
    PERTURBATIONS,
    TARGET_FPR,
    build_argument_parser,
    main,
    run_evaluation,
)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the shim still satisfies test_joint_detection.py imports**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_joint_detection.py -v
```

Expected: all tests pass (including the heavyweight `test_evaluate_dual_channel_builds_all_three_modes`).

- [ ] **Step 3: Verify the script can still be invoked directly (help-only smoke test)**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate_dual_channel.py --help
```

Expected: prints the existing CLI help text, including `--dataset {humaneval,mbpp}`, `--config`, `--output-dir`, `--num-candidates`. Exits 0. A `DeprecationWarning` is printed to stderr (that's the point of the shim).

- [ ] **Step 4: Commit**

```
git add scripts/evaluate_dual_channel.py
git commit -m "refactor: collapse scripts/evaluate_dual_channel.py to deprecation shim"
```

---

## Task 7: Create unified `scripts/evaluate.py` subcommand entry

**Files:**
- Create: `scripts/evaluate.py`
- Create: `tests/integration/test_evaluate_cli.py`

Spec §4.4 shows the target CLI:

```
python scripts/evaluate.py exec       <jsonl> [<jsonl>...] --metric pass_at_1 [--k 1] [--reference <path>]
python scripts/evaluate.py detection  --left-summary X --left-details Y --right-summary Z --right-details W --output report.json [--left-watermarked ...] [--right-watermarked ...]
python scripts/evaluate.py dual       --dataset humaneval --config configs/base_config.json --output-dir data/eval/dual_channel [--num-candidates N]
```

Each subcommand is a thin argparse wrapper around the functions already present in `wfcllm.evaluation.*`. No new business logic.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/integration/test_evaluate_cli.py`:

```python
"""Tests for the unified scripts/evaluate.py CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the script's main() via runpy-style attribute access so we don't shell out.
import importlib.util


def _load_evaluate_module():
    spec = importlib.util.spec_from_file_location(
        "_scripts_evaluate", REPO_ROOT / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- exec subcommand ---

def test_evaluate_cli_exec_computes_pass_at_k(tmp_path, capsys):
    input_path = tmp_path / "candidates.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"task_id": "t1", "is_correct": True},
                {"task_id": "t1", "is_correct": False},
                {"task_id": "t2", "is_correct": True},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evaluate = _load_evaluate_module()
    rc = evaluate.main(["exec", str(input_path), "--metric", "pass_at_1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == "pass_at_1"
    assert out["value"] == pytest.approx(0.75)
    assert out["sample_count"] == 3


def test_evaluate_cli_exec_with_reference_annotates_correctness(tmp_path, capsys):
    candidates_path = tmp_path / "cands.jsonl"
    reference_path = tmp_path / "refs.jsonl"
    candidates_path.write_text(
        json.dumps({"task_id": "t1", "generated_code": "def f(x):\n    return x + 1\n"}) + "\n",
        encoding="utf-8",
    )
    reference_path.write_text(
        json.dumps({"id": "t1", "generated_code": "def f(x):\n    return x + 1\n"}) + "\n",
        encoding="utf-8",
    )
    evaluate = _load_evaluate_module()
    rc = evaluate.main([
        "exec", str(candidates_path),
        "--metric", "pass_at_1",
        "--reference", str(reference_path),
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == pytest.approx(1.0)


# --- detection subcommand ---

def test_evaluate_cli_detection_writes_report(tmp_path, capsys):
    summary_payload = {
        "dataset": "HumanEval",
        "watermark_params": {"lsh_d": 3},
        "summary": {"watermark_rate": 0.5},
    }
    detail_row = {
        "id": "HumanEval/0",
        "is_watermarked": True,
        "z_score": 4.0,
        "p_value": 0.0001,
        "independent_blocks": 5,
        "hits": 4,
    }
    left_summary = tmp_path / "ls.json"; left_summary.write_text(json.dumps(summary_payload))
    right_summary = tmp_path / "rs.json"; right_summary.write_text(json.dumps(summary_payload))
    left_details = tmp_path / "ld.jsonl"; left_details.write_text(json.dumps(detail_row) + "\n")
    right_details = tmp_path / "rd.jsonl"; right_details.write_text(json.dumps(detail_row) + "\n")
    output = tmp_path / "report.json"

    evaluate = _load_evaluate_module()
    rc = evaluate.main([
        "detection",
        "--left-summary", str(left_summary),
        "--left-details", str(left_details),
        "--right-summary", str(right_summary),
        "--right-details", str(right_details),
        "--output", str(output),
    ])
    assert rc == 0
    assert output.exists()
    report = json.loads(output.read_text())
    assert "compatibility" in report
    assert "detail_delta" in report
    assert report["compatibility"]["same_id_set"] is True


# --- dual subcommand routing ---

def test_evaluate_cli_dual_dispatches_to_dual_channel(tmp_path, monkeypatch):
    """Verify the dual subcommand calls wfcllm.evaluation.dual_channel.run_evaluation."""
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"watermark": {}, "extract": {}}))

    captured: dict = {}

    def fake_run_evaluation(*, dataset, config_path, output_dir, candidate_count, **kw):
        captured["dataset"] = dataset
        captured["config_path"] = config_path
        captured["output_dir"] = output_dir
        captured["candidate_count"] = candidate_count
        return {"dataset": dataset, "modes": {}}

    import wfcllm.evaluation.dual_channel as dc
    monkeypatch.setattr(dc, "run_evaluation", fake_run_evaluation)

    evaluate = _load_evaluate_module()
    rc = evaluate.main([
        "dual",
        "--dataset", "humaneval",
        "--config", str(config_path),
        "--output-dir", str(tmp_path / "out"),
        "--num-candidates", "1",
    ])
    assert rc == 0
    assert captured["dataset"] == "humaneval"
    assert captured["candidate_count"] == 1


def test_evaluate_cli_unknown_subcommand_exits_nonzero(capsys):
    evaluate = _load_evaluate_module()
    with pytest.raises(SystemExit):
        evaluate.main(["frobnicate"])
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluate_cli.py -v
```

Expected: tests fail because `scripts/evaluate.py` doesn't exist yet.

- [ ] **Step 3: Create `scripts/evaluate.py`**

```python
#!/usr/bin/env python
"""Unified offline evaluation entry point.

Three subcommands:
  exec        compute pass@k (and friends) over JSONL candidate rows
  detection   build the offline regression report from saved summary + details
  dual        run the dual-channel end-to-end harness (semantic / lexical / dual)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.evaluation.code_execution import (  # noqa: E402
    annotate_correctness_from_references,
    compute_pass_at_k,
    load_jsonl_records,
)
from wfcllm.evaluation.detection_report import (  # noqa: E402
    build_offline_regression_report,
    load_detail_artifact,
    load_summary_artifact,
    load_watermarked_artifact,
    write_offline_regression_report,
)
from wfcllm.evaluation import dual_channel as dual_channel_module  # noqa: E402


def _cmd_exec(args: argparse.Namespace) -> int:
    records: list[dict] = []
    for path in args.inputs:
        records.extend(load_jsonl_records(path))

    if args.reference is not None:
        reference_records = load_jsonl_records(args.reference)
        records = annotate_correctness_from_references(records, reference_records)

    if args.metric == "pass_at_k":
        k = args.k
    elif args.metric == "pass_at_1":
        k = 1
    elif args.metric == "pass_at_10":
        k = 10
    else:
        raise ValueError(f"unsupported metric: {args.metric}")

    value = compute_pass_at_k(records, k=k)
    print(json.dumps({
        "metric": args.metric,
        "k": k,
        "value": value,
        "sample_count": len(records),
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_detection(args: argparse.Namespace) -> int:
    left_watermarked = (
        load_watermarked_artifact(args.left_watermarked) if args.left_watermarked else None
    )
    right_watermarked = (
        load_watermarked_artifact(args.right_watermarked) if args.right_watermarked else None
    )
    report = build_offline_regression_report(
        left_summary=load_summary_artifact(args.left_summary),
        left_details=load_detail_artifact(args.left_details),
        left_watermarked=left_watermarked,
        right_summary=load_summary_artifact(args.right_summary),
        right_details=load_detail_artifact(args.right_details),
        right_watermarked=right_watermarked,
    )
    output_path = write_offline_regression_report(args.output, report)
    print(f"[完成] 离线回归报告已保存至 {output_path}")
    return 0


def _cmd_dual(args: argparse.Namespace) -> int:
    result = dual_channel_module.run_evaluation(
        dataset=args.dataset,
        config_path=args.config,
        output_dir=args.output_dir,
        candidate_count=args.num_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WFCLLM unified offline evaluation entry point.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    exec_parser = subparsers.add_parser("exec", help="pass@k from JSONL candidate rows")
    exec_parser.add_argument("inputs", nargs="+", help="one or more JSONL files of candidates")
    exec_parser.add_argument(
        "--metric",
        choices=["pass_at_1", "pass_at_10", "pass_at_k"],
        default="pass_at_1",
    )
    exec_parser.add_argument("--k", type=int, default=1, help="k for --metric pass_at_k")
    exec_parser.add_argument(
        "--reference",
        default=None,
        help="optional reference JSONL; if given, candidate rows are re-annotated for correctness",
    )
    exec_parser.set_defaults(func=_cmd_exec)

    det_parser = subparsers.add_parser(
        "detection",
        help="offline regression report from saved summary + details artifacts",
    )
    det_parser.add_argument("--left-summary", required=True)
    det_parser.add_argument("--left-details", required=True)
    det_parser.add_argument("--right-summary", required=True)
    det_parser.add_argument("--right-details", required=True)
    det_parser.add_argument("--left-watermarked", default=None)
    det_parser.add_argument("--right-watermarked", default=None)
    det_parser.add_argument("--output", required=True, help="report JSON output path")
    det_parser.set_defaults(func=_cmd_detection)

    dual_parser = subparsers.add_parser(
        "dual",
        help="end-to-end dual-channel evaluation harness",
    )
    dual_parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    dual_parser.add_argument("--config", default="configs/base_config.json")
    dual_parser.add_argument("--output-dir", default="data/eval/dual_channel")
    dual_parser.add_argument("--num-candidates", type=int, default=10)
    dual_parser.set_defaults(func=_cmd_dual)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI tests**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_evaluate_cli.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Manually smoke-test `--help`**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate.py --help
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate.py exec --help
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate.py detection --help
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate.py dual --help
```

Expected: each prints its respective help text and exits 0.

- [ ] **Step 6: Commit**

```
git add scripts/evaluate.py tests/integration/test_evaluate_cli.py
git commit -m "feat: add scripts/evaluate.py unified CLI (exec/detection/dual subcommands)"
```

---

## Task 8: Final verification — full test suite & line-count audit

**Files:**
- (No file changes; verification only.)

- [ ] **Step 1: Run the full test suite**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

Expected: all tests pass, including:
- `tests/integration/test_evaluation_shims.py` (7 tests added in this phase: 3 code_execution + 3 detection_report + 1 dual_channel smoke)
- `tests/integration/test_evaluate_cli.py` (5 tests added)
- `tests/extract/test_offline_analysis.py` (untouched; exercises the shim)
- `tests/extract/test_joint_detection.py` (untouched; exercises both shims)
- `tests/test_run.py` (run_offline_analysis path now via new module)

If any test fails, do NOT proceed to commit. Diff against the pre-Phase-2 commit for the file in question and check for a copy-paste typo.

- [ ] **Step 2: Verify the old files are now thin shims**

```
wc -l wfcllm/common/offline_code_eval.py wfcllm/extract/offline_analysis.py scripts/evaluate_dual_channel.py
```

Expected line counts (approximate):
- `wfcllm/common/offline_code_eval.py`: ~15 lines (was 360)
- `wfcllm/extract/offline_analysis.py`: ~15 lines (was 687)
- `scripts/evaluate_dual_channel.py`: ~35 lines (was 483)

- [ ] **Step 3: Verify the new files have the bulk of the content**

```
wc -l wfcllm/evaluation/*.py scripts/evaluate.py
```

Expected:
- `wfcllm/evaluation/__init__.py`: 1 line
- `wfcllm/evaluation/code_execution.py`: ~360 lines
- `wfcllm/evaluation/detection_report.py`: ~690 lines
- `wfcllm/evaluation/dual_channel.py`: ~475 lines (slightly shorter than the 483-line original because the `sys.path` block was dropped)
- `scripts/evaluate.py`: ~130 lines

- [ ] **Step 4: Confirm git log shows clean commits for this phase**

```
git log --oneline main..HEAD
```

Expected: 7 well-scoped commits since Phase 1 closed — one per task (Tasks 2–7) plus optional skeleton commit from Task 1.

- [ ] **Step 5: (Optional) Spot-check the public CLI**

If a small `data/datasets/humaneval/` reference fixture is available locally, run a sanity check of the new `exec` subcommand to confirm it still parses real data; otherwise skip.

- [ ] **Step 6: Close-out — no commit needed if everything above is green.**

---

## Phase 2 Summary (after all tasks)

**Files created (5):**

- `wfcllm/evaluation/__init__.py`
- `wfcllm/evaluation/code_execution.py` (moved from `wfcllm/common/offline_code_eval.py`)
- `wfcllm/evaluation/detection_report.py` (moved from `wfcllm/extract/offline_analysis.py`)
- `wfcllm/evaluation/dual_channel.py` (moved from `scripts/evaluate_dual_channel.py` core)
- `scripts/evaluate.py` (new unified CLI)
- `tests/integration/test_evaluation_shims.py`
- `tests/integration/test_evaluate_cli.py`

**Files modified:**

- `wfcllm/common/offline_code_eval.py` (360 → ~15 lines: deprecation shim)
- `wfcllm/extract/offline_analysis.py` (687 → ~15 lines: deprecation shim)
- `scripts/evaluate_dual_channel.py` (483 → ~35 lines: deprecation shim + path bootstrap + main passthrough)
- `wfcllm/cli/runners.py` (one-line import path change inside `run_offline_analysis`)

**Files deleted:**

- None. Old paths are kept as shims for the ≤1-month migration window (spec §8.2). A final phase will batch-delete them.

**Behavior preserved:**

- All existing tests pass without modification.
- Public function signatures unchanged.
- `scripts/evaluate_dual_channel.py --dataset X ...` still works (with a one-line DeprecationWarning printed to stderr).
- `tests/extract/test_joint_detection.py` imports from `scripts.evaluate_dual_channel` continue to resolve.
- `tests/extract/test_offline_analysis.py` imports from `wfcllm.extract.offline_analysis` continue to resolve.

**New behaviors:**

- Canonical home for cross-stage evaluation: `wfcllm/evaluation/`.
- `scripts/evaluate.py` is the documented entry point going forward.
- Internal first-party caller (`wfcllm.cli.runners.run_offline_analysis`) now uses canonical paths.
- Shim tests provide a single place to verify the deprecation-window contract.

**What this unlocks:**

- Phase 3 (`adaptive_gamma/` subpackage) can follow the same shim pattern.
- Phase 4 (`extract/calibration/`) can follow the same shim pattern.
- Phase 7 (`ablation/`) can register evaluation metric collectors against `wfcllm.evaluation.*`.

---

## Notes for the implementer

- **Do not refactor while moving.** Even if a function looks ugly or its docstring could be improved, leave it byte-identical. Polish in a separate commit later.
- **Do not delete the shims in this phase.** Spec §8.2 keeps the migration window open for ≤1 month; shims are removed in one final batch commit at the end of the overall refactor.
- **Do not migrate `tests/extract/test_offline_analysis.py` or `tests/extract/test_joint_detection.py`.** They intentionally exercise the old paths and let us verify the shims work. The reorganization to `tests/unit/evaluation/` belongs to a later phase (spec §10 step 9).
- **`scripts/evaluate_dual_channel.py` must keep working when called as a script** (not just as a module). The `REPO_ROOT` `sys.path` insertion at the top of the shim is intentional — drop it from the `wfcllm.evaluation.dual_channel` module body (it doesn't need it) but keep it in the script shim.
- **Watch for `DeprecationWarning` noise during pytest.** Old-path test imports will emit warnings; this is expected and shows up as informational lines in the pytest tail. If the suite ever upgrades to `-W error::DeprecationWarning`, that's a Phase 9 concern, not Phase 2.
- **The placeholder `# ... paste lines NN-MM verbatim ...` comments in Task 5 must NOT survive into the committed file** — they are scaffolding for the implementer only.
