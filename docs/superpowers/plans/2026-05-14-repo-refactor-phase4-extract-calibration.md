# Phase 4: extract/calibration 子包 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize FPR-threshold calibration and negative-corpus generation into a cohesive `wfcllm/extract/calibration/` subpackage; rename `scripts/calibrate.py calibrate-threshold` → `scripts/calibrate_threshold.py`; preserve all old import paths and CLI invocations via deprecation shims.

**Architecture:** Pure code-move + thin shim, mirroring Phase 2 (`evaluation/`) and Phase 3 (`watermark/adaptive_gamma/`). Logic stays bit-for-bit identical; only physical location changes. The CLI core (currently inlined in `scripts/calibrate.py:_calibrate_threshold`) is lifted into `wfcllm/extract/calibration/runner.py` so the new `scripts/calibrate_threshold.py` becomes a thin argparse shell — same pattern as `scripts/build_entropy_profile.py` ↔ `wfcllm/watermark/adaptive_gamma/calibrate.py`.

**Tech Stack:** Python 3.10+, pytest, transformers (model loading in calibration runner only), no new external deps.

**Spec reference:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §4.3 + §10 step 4.

**Out of scope (explicitly NOT in this phase):**

- Registering `calibrate-threshold` as an orchestration phase (§6.2 / §6.3 — defer to a later phase). `wfcllm/orchestration/state.py:OPTIONAL_PHASES` is **not** modified by this plan.
- Auto-triggering calibration as a prereq for `extract` (§6.3 — defer).
- Changing `generate-negative` orchestration registration (already exists from earlier phases — leave untouched).

---

## File Structure

**New files (canonical locations):**

- `wfcllm/extract/calibration/__init__.py` — re-exports `ThresholdCalibrator`, `NegativeCorpusConfig`, `NegativeCorpusGenerator`, and the `runner` module.
- `wfcllm/extract/calibration/threshold.py` — body of current `wfcllm/extract/calibrator.py` (verbatim).
- `wfcllm/extract/calibration/negative_corpus.py` — body of current `wfcllm/extract/negative_corpus.py` (verbatim).
- `wfcllm/extract/calibration/runner.py` — `calibrate_threshold_from_corpus(...)` lifted from `scripts/calibrate.py:_calibrate_threshold` (CLI-args replaced by typed kwargs).
- `scripts/calibrate_threshold.py` — new thin CLI shell that calls `calibrate_threshold_from_corpus`.
- `tests/integration/test_extract_calibration_shims.py` — new path / old path / shim-equivalence tests.
- `tests/integration/test_calibrate_threshold_runner.py` — runner unit tests using a fake scorer.

**Modified files:**

- `wfcllm/extract/calibrator.py` — collapse to deprecation shim re-exporting from `wfcllm.extract.calibration.threshold`.
- `wfcllm/extract/negative_corpus.py` — collapse to deprecation shim re-exporting from `wfcllm.extract.calibration.negative_corpus`.
- `wfcllm/extract/__init__.py` — switch imports to `wfcllm.extract.calibration.*` canonical paths.
- `wfcllm/cli/runners.py` — switch internal imports (`run_extract`, `run_generate_negative`) to canonical paths.
- `scripts/generate_negative_corpus.py` — switch its import to the canonical path; keep file name (per spec §4.3).
- `scripts/calibrate.py` — collapse `calibrate-threshold` to a deprecation proxy that calls into `wfcllm.extract.calibration.runner` (mirrors how Phase 3 collapsed `build-entropy-profile` to a proxy).
- `tests/extract/test_negative_corpus.py` — three `patch(...)` call sites repointed from the old shim path to the canonical `wfcllm.extract.calibration.negative_corpus` path (see Task 6 step 6 for rationale).

**Unchanged (do NOT touch):**

- `wfcllm/extract/calibrator.py` and `wfcllm/extract/negative_corpus.py` are *replaced* with shims, but their public symbols and signatures stay identical so existing unit tests under `tests/extract/test_calibrator.py` and `tests/extract/test_negative_corpus.py` keep passing without modification.
- `experiment/`, `tools/` directories.
- All algorithm logic — pure code move.

---

## Conventions

**Deprecation shim template** (matches Phase 2/3 pattern in `wfcllm/watermark/entropy.py` etc.):

```python
"""Deprecated import path. Use wfcllm.extract.calibration.<NEW> instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.extract.<OLD> is deprecated; use wfcllm.extract.calibration.<NEW>",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.extract.calibration.<NEW> import *  # noqa: E402, F401, F403
from wfcllm.extract.calibration.<NEW> import (  # noqa: E402
    <SYMBOLS_USED_BY_TESTS>,
)
```

The `from ... import *` covers consumers; the explicit named import preserves attribute access for `monkeypatch.setattr("wfcllm.extract.calibrator.ThresholdCalibrator", ...)` style tests in `tests/test_run_config.py` (lines 231, 681, 816).

**Test command:** `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest <path> -v`

**Commit message format:** `<type>: <description>` per `CLAUDE.md` (types: feat / fix / refactor / test / docs / chore).

---

## Task 1: Create empty `calibration/` package + first failing shim test

**Files:**
- Create: `wfcllm/extract/calibration/__init__.py`
- Create: `tests/integration/test_extract_calibration_shims.py`

- [ ] **Step 1: Create empty package `__init__.py`**

Write `wfcllm/extract/calibration/__init__.py`:

```python
"""Calibration subpackage: FPR threshold + negative corpus generation."""
```

- [ ] **Step 2: Write the first failing shim test (canonical path importable)**

Write `tests/integration/test_extract_calibration_shims.py`:

```python
"""Tests for wfcllm/extract/calibration/* migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import importlib
import sys
import warnings


# --- threshold: new path works ---

def test_threshold_new_path_importable():
    from wfcllm.extract.calibration.threshold import ThresholdCalibrator
    assert callable(ThresholdCalibrator)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_extract_calibration_shims.py::test_threshold_new_path_importable -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.extract.calibration.threshold'`.

- [ ] **Step 4: Commit the package skeleton + failing test**

```bash
git add wfcllm/extract/calibration/__init__.py tests/integration/test_extract_calibration_shims.py
git commit -m "test: scaffold extract/calibration shim test (Phase 4)"
```

---

## Task 2: Move `extract/calibrator.py` → `extract/calibration/threshold.py` + shim

**Files:**
- Create: `wfcllm/extract/calibration/threshold.py`
- Modify: `wfcllm/extract/calibrator.py` (becomes deprecation shim)
- Modify: `tests/integration/test_extract_calibration_shims.py` (add shim tests)

- [ ] **Step 1: Copy current `calibrator.py` body verbatim to new path**

Read the entire content of `wfcllm/extract/calibrator.py` (118 lines) and write that content unmodified into `wfcllm/extract/calibration/threshold.py`. The file already has its own module docstring and imports — no edits to internals.

Internal imports to be aware of (already absolute, do NOT change):
- `from wfcllm.common.ast_parser import extract_statement_blocks`
- `from wfcllm.extract import hypothesis`
- `from wfcllm.extract.config import BlockScore`
- `from wfcllm.extract.scorer import BlockScorer`

- [ ] **Step 2: Replace old `wfcllm/extract/calibrator.py` with a deprecation shim**

Overwrite `wfcllm/extract/calibrator.py` with:

```python
"""Deprecated import path. Use wfcllm.extract.calibration.threshold instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.extract.calibrator is deprecated; use wfcllm.extract.calibration.threshold",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.extract.calibration.threshold import *  # noqa: E402, F401, F403
from wfcllm.extract.calibration.threshold import ThresholdCalibrator  # noqa: E402, F401
```

- [ ] **Step 3: Add shim tests**

Append to `tests/integration/test_extract_calibration_shims.py`:

```python
# --- threshold: old path is a deprecated shim ---

def test_threshold_old_path_emits_deprecation_warning():
    sys.modules.pop("wfcllm.extract.calibrator", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.extract.calibrator")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.extract.calibration.threshold" in str(w.message)
        for w in caught
    )
    assert callable(module.ThresholdCalibrator)


def test_threshold_old_and_new_paths_share_symbol():
    sys.modules.pop("wfcllm.extract.calibrator", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.extract.calibrator")
    new = importlib.import_module("wfcllm.extract.calibration.threshold")
    assert old.ThresholdCalibrator is new.ThresholdCalibrator
```

- [ ] **Step 4: Run shim tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_extract_calibration_shims.py -v`
Expected: 3 PASS (the new-path test from Task 1 plus the two added here).

- [ ] **Step 5: Run existing unit tests to confirm shim is backwards-compatible**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_calibrator.py tests/extract/test_pipeline.py -v`
Expected: ALL PASS (these tests still import from `wfcllm.extract.calibrator` — the shim must keep them green).

- [ ] **Step 6: Commit**

```bash
git add wfcllm/extract/calibration/threshold.py wfcllm/extract/calibrator.py tests/integration/test_extract_calibration_shims.py
git commit -m "refactor: move ThresholdCalibrator to extract/calibration/threshold + shim"
```

---

## Task 3: Move `extract/negative_corpus.py` → `extract/calibration/negative_corpus.py` + shim

**Files:**
- Create: `wfcllm/extract/calibration/negative_corpus.py`
- Modify: `wfcllm/extract/negative_corpus.py` (becomes deprecation shim)
- Modify: `tests/integration/test_extract_calibration_shims.py` (add shim tests)

- [ ] **Step 1: Copy current `negative_corpus.py` body verbatim to new path**

Read the entire content of `wfcllm/extract/negative_corpus.py` (185 lines) and write that content unmodified into `wfcllm/extract/calibration/negative_corpus.py`.

Internal imports to be aware of (already absolute, do NOT change):
- `from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts, load_reference_solutions`

Note: `NegativeCorpusGenerator.__init__` lazily imports `transformers`. The shim (Step 2) is module-level, but the lazy import inside the class stays inside the new file unchanged.

- [ ] **Step 2: Replace old `wfcllm/extract/negative_corpus.py` with a deprecation shim**

Overwrite `wfcllm/extract/negative_corpus.py` with:

```python
"""Deprecated import path. Use wfcllm.extract.calibration.negative_corpus instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.extract.negative_corpus is deprecated; use wfcllm.extract.calibration.negative_corpus",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.extract.calibration.negative_corpus import *  # noqa: E402, F401, F403
from wfcllm.extract.calibration.negative_corpus import (  # noqa: E402, F401
    NegativeCorpusConfig,
    NegativeCorpusGenerator,
)
```

Note on `load_prompts` / `load_reference_solutions`: the existing test suite at `tests/extract/test_negative_corpus.py` lines 110/136/169 currently patches them through the old module path (`patch("wfcllm.extract.negative_corpus.load_prompts", ...)`). After the move, those globals live on the *new* module (`wfcllm.extract.calibration.negative_corpus`), and `NegativeCorpusGenerator.run()` resolves them via that module's globals — so patching the shim's bindings would have no effect on the consumer. Therefore:

- The shim deliberately does **not** re-export `load_prompts` / `load_reference_solutions`.
- The patch sites in `tests/extract/test_negative_corpus.py` are updated in Task 6 step 6 to point at the canonical module.

- [ ] **Step 3: Add shim tests**

Append to `tests/integration/test_extract_calibration_shims.py`:

```python
# --- negative_corpus: new path works ---

def test_negative_corpus_new_path_importable():
    from wfcllm.extract.calibration.negative_corpus import (
        NegativeCorpusConfig,
        NegativeCorpusGenerator,
    )
    cfg = NegativeCorpusConfig(lm_model_path="/tmp/fake", output_path="/tmp/out.jsonl")
    assert cfg.dataset == "humaneval"
    assert callable(NegativeCorpusGenerator)


# --- negative_corpus: old path is a deprecated shim ---

def test_negative_corpus_old_path_emits_deprecation_warning():
    sys.modules.pop("wfcllm.extract.negative_corpus", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.extract.negative_corpus")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.extract.calibration.negative_corpus" in str(w.message)
        for w in caught
    )
    assert callable(module.NegativeCorpusGenerator)
    assert callable(module.NegativeCorpusConfig)


def test_negative_corpus_old_and_new_paths_share_symbols():
    sys.modules.pop("wfcllm.extract.negative_corpus", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.extract.negative_corpus")
    new = importlib.import_module("wfcllm.extract.calibration.negative_corpus")
    assert old.NegativeCorpusGenerator is new.NegativeCorpusGenerator
    assert old.NegativeCorpusConfig is new.NegativeCorpusConfig
```

- [ ] **Step 4: Run shim tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_extract_calibration_shims.py -v`
Expected: 6 PASS (3 from earlier tasks + 3 added here).

- [ ] **Step 5: Verify the unit tests that DON'T touch the patched globals still pass**

The three test methods that currently patch `wfcllm.extract.negative_corpus.load_prompts` / `load_reference_solutions` will not pass yet — those patches will become effective only after Task 6 step 6 redirects them to the canonical module path. Confirm the rest still pass:

Run:
```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_negative_corpus.py -v \
  --deselect tests/extract/test_negative_corpus.py::TestNegativeCorpusGeneratorRun::test_run_reference_mode_writes_dataset_solutions \
  --deselect tests/extract/test_negative_corpus.py::TestNegativeCorpusGeneratorRun::test_run_writes_jsonl \
  --deselect tests/extract/test_negative_corpus.py::TestNegativeCorpusGeneratorRun::test_run_respects_limit
```

Expected: ALL remaining tests PASS. The deselect list anchors the fact that those three are knowingly broken between Task 3 and Task 6 step 6 — do not "fix" them by re-adding `load_prompts` / `load_reference_solutions` to the shim, because that re-export would be ineffective (see the explanatory note above the shim definition: the consumer resolves them via the *new* module's globals, not the shim's).

**Sanity check:** before running, `grep -n "def test_run_reference_mode_writes_dataset_solutions\|def test_run_writes_jsonl\|def test_run_respects_limit" tests/extract/test_negative_corpus.py` should yield three matches. If any test name has drifted, update the deselect command accordingly.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/extract/calibration/negative_corpus.py wfcllm/extract/negative_corpus.py tests/integration/test_extract_calibration_shims.py
git commit -m "refactor: move NegativeCorpusGenerator to extract/calibration/negative_corpus + shim"
```

---

## Task 4: Lift `_calibrate_threshold` core into `extract/calibration/runner.py`

**Files:**
- Create: `wfcllm/extract/calibration/runner.py`
- Create: `tests/integration/test_calibrate_threshold_runner.py`

This task extracts the body of `scripts/calibrate.py:_calibrate_threshold` (lines 100–138) into a typed, importable function so the new CLI shell (Task 5) and any future orchestration phase can call it with kwargs instead of an `argparse.Namespace`. The translation is mechanical:

| `args.<field>`     | function kwarg           |
| ------------------ | ------------------------ |
| `args.input`       | `input: str \| Path`     |
| `args.output`      | `output: str \| Path`    |
| `args.fpr`         | `fpr: float = 0.01`      |
| `args.secret_key`  | `secret_key: str`        |
| `args.model`       | `model: str`             |
| `args.device`      | `device: str = "cuda"`   |
| `args.embed_dim`   | `embed_dim: int = 128`   |
| `args.lsh_d`       | `lsh_d: int = 3`         |
| `args.gamma`       | `gamma: float = 0.5`     |

- [ ] **Step 1: Write the failing test (corpus → threshold result)**

Write `tests/integration/test_calibrate_threshold_runner.py`:

```python
"""Tests for wfcllm.extract.calibration.runner.calibrate_threshold_from_corpus."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_calibrate_threshold_from_corpus_writes_result_json(tmp_path):
    from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus

    input_path = tmp_path / "corpus.jsonl"
    output_path = tmp_path / "threshold.json"
    _write_jsonl(input_path, [{"generated_code": "x = 1"} for _ in range(8)])

    fake_calibrator = MagicMock()
    fake_calibrator.calibrate.return_value = {
        "fpr": 0.01,
        "fpr_threshold": 1.234,
        "n_samples": 8,
    }

    with (
        patch("wfcllm.extract.calibration.runner.AutoTokenizer") as mock_tok,
        patch("wfcllm.extract.calibration.runner.AutoModel") as mock_model,
        patch("wfcllm.extract.calibration.runner.LSHSpace"),
        patch("wfcllm.extract.calibration.runner.WatermarkKeying"),
        patch("wfcllm.extract.calibration.runner.ProjectionVerifier"),
        patch("wfcllm.extract.calibration.runner.BlockScorer"),
        patch(
            "wfcllm.extract.calibration.runner.ThresholdCalibrator",
            return_value=fake_calibrator,
        ),
    ):
        mock_tok.from_pretrained.return_value = MagicMock()
        encoder_instance = MagicMock()
        encoder_instance.to.return_value = encoder_instance
        encoder_instance.eval.return_value = encoder_instance
        mock_model.from_pretrained.return_value = encoder_instance

        result_path = calibrate_threshold_from_corpus(
            input=str(input_path),
            output=str(output_path),
            secret_key="k",
            model="data/models/codet5-base",
            device="cpu",
            fpr=0.01,
        )

    assert Path(result_path) == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["fpr"] == 0.01
    assert payload["fpr_threshold"] == pytest.approx(1.234)
    assert payload["n_samples"] == 8
    fake_calibrator.calibrate.assert_called_once()
    args_passed, kwargs_passed = fake_calibrator.calibrate.call_args
    assert kwargs_passed.get("fpr") == 0.01 or (len(args_passed) >= 2 and args_passed[1] == 0.01)
    assert len(args_passed[0]) == 8


def test_calibrate_threshold_from_corpus_returns_path_object(tmp_path):
    from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus

    input_path = tmp_path / "corpus.jsonl"
    output_path = tmp_path / "nested" / "threshold.json"
    _write_jsonl(input_path, [{"generated_code": "x = 1"}])

    fake_calibrator = MagicMock()
    fake_calibrator.calibrate.return_value = {"fpr": 0.05, "fpr_threshold": 0.5, "n_samples": 1}

    with (
        patch("wfcllm.extract.calibration.runner.AutoTokenizer"),
        patch("wfcllm.extract.calibration.runner.AutoModel") as mock_model,
        patch("wfcllm.extract.calibration.runner.LSHSpace"),
        patch("wfcllm.extract.calibration.runner.WatermarkKeying"),
        patch("wfcllm.extract.calibration.runner.ProjectionVerifier"),
        patch("wfcllm.extract.calibration.runner.BlockScorer"),
        patch(
            "wfcllm.extract.calibration.runner.ThresholdCalibrator",
            return_value=fake_calibrator,
        ),
    ):
        encoder_instance = MagicMock()
        encoder_instance.to.return_value = encoder_instance
        encoder_instance.eval.return_value = encoder_instance
        mock_model.from_pretrained.return_value = encoder_instance

        result = calibrate_threshold_from_corpus(
            input=input_path,
            output=output_path,
            secret_key="k",
            model="any",
            device="cpu",
        )

    assert isinstance(result, Path)
    assert result.parent.exists()  # parent dir auto-created
    assert result == output_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_calibrate_threshold_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.extract.calibration.runner'`.

- [ ] **Step 3: Write `runner.py` with the lifted implementation**

Write `wfcllm/extract/calibration/runner.py`:

```python
"""Calibrate FPR-based watermark detection threshold from a negative corpus.

Logic moved from scripts/calibrate.py:_calibrate_threshold (Phase 4 refactor).
The CLI shell that wraps this lives in scripts/calibrate_threshold.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from wfcllm.extract.calibration.threshold import ThresholdCalibrator
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.verifier import ProjectionVerifier


def _load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def calibrate_threshold_from_corpus(
    *,
    input: str | Path,
    output: str | Path,
    secret_key: str,
    model: str,
    device: str = "cuda",
    fpr: float = 0.01,
    embed_dim: int = 128,
    lsh_d: int = 3,
    gamma: float = 0.5,
) -> Path:
    """Calibrate the FPR detection threshold and persist the result JSON.

    Returns the resolved output path.
    """
    print(f"Loading model from {model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model)
    encoder = AutoModel.from_pretrained(model).to(device)
    encoder.eval()

    lsh_space = LSHSpace(secret_key, embed_dim, lsh_d)
    keying = WatermarkKeying(secret_key, lsh_d, gamma)
    verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=device)
    scorer = BlockScorer(keying, verifier)

    print(f"Loading corpus from {input} ...", file=sys.stderr)
    corpus = _load_jsonl(input)
    print(f"  {len(corpus)} samples loaded.", file=sys.stderr)

    calibrator = ThresholdCalibrator(scorer, gamma=gamma)
    result = calibrator.calibrate(corpus, fpr=fpr)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Calibration complete:\n"
        f"  FPR target    : {result['fpr']}\n"
        f"  M_r threshold : {result['fpr_threshold']:.4f}\n"
        f"  Samples used  : {result['n_samples']}\n"
        f"  Output        : {output}",
        file=sys.stderr,
    )
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_calibrate_threshold_runner.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/extract/calibration/runner.py tests/integration/test_calibrate_threshold_runner.py
git commit -m "feat: add extract/calibration/runner.calibrate_threshold_from_corpus"
```

---

## Task 5: Add `scripts/calibrate_threshold.py` + collapse `scripts/calibrate.py` to proxy

**Files:**
- Create: `scripts/calibrate_threshold.py`
- Modify: `scripts/calibrate.py` (full rewrite — both subcommands become deprecation proxies)
- Create: `tests/integration/test_calibrate_threshold_cli.py`

This task introduces the new CLI (`scripts/calibrate_threshold.py`) and reduces `scripts/calibrate.py` to two thin proxies that print a deprecation notice and forward to the canonical entry points. This mirrors how Phase 3 already collapsed `build-entropy-profile` to a proxy in `scripts/calibrate.py:_proxy_build_entropy_profile` (lines 65–86 of the current file).

- [ ] **Step 1: Write the failing CLI smoke test**

Write `tests/integration/test_calibrate_threshold_cli.py`:

```python
"""Smoke test for scripts/calibrate_threshold.py: argparse layer wires args correctly."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_calibrate_threshold_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "calibrate_threshold.py"
    spec = importlib.util.spec_from_file_location("calibrate_threshold_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibrate_threshold_cli_passes_args_to_runner(tmp_path, monkeypatch):
    module = _load_calibrate_threshold_module()
    input_path = tmp_path / "corpus.jsonl"
    input_path.write_text('{"generated_code": "x"}\n', encoding="utf-8")
    output_path = tmp_path / "thr.json"

    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fpr": kwargs["fpr"]}), encoding="utf-8")
        return out

    monkeypatch.setattr(module, "calibrate_threshold_from_corpus", fake_runner)
    monkeypatch.setattr(
        sys, "argv",
        [
            "calibrate_threshold.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--secret-key", "k",
            "--model", "data/models/codet5-base",
            "--device", "cpu",
            "--fpr", "0.02",
            "--gamma", "0.5",
            "--embed-dim", "128",
            "--lsh-d", "3",
        ],
    )
    rc = module.main()
    assert rc == 0
    assert captured["secret_key"] == "k"
    assert captured["fpr"] == 0.02
    assert captured["device"] == "cpu"
    assert Path(captured["output"]) == output_path
    assert output_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_calibrate_threshold_cli.py -v`
Expected: FAIL — `scripts/calibrate_threshold.py` does not yet exist (`FileNotFoundError` from `spec_from_file_location` resolving to None, surfaced as `AttributeError` on `spec.loader`).

- [ ] **Step 3: Write the new CLI shell**

Write `scripts/calibrate_threshold.py`:

```python
#!/usr/bin/env python
"""Calibrate FPR-based watermark detection threshold from a negative corpus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate FPR-based watermark detection threshold.",
    )
    parser.add_argument("--input", required=True, help="negative corpus JSONL")
    parser.add_argument("--output", required=True, help="threshold result JSON")
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--model", required=True, help="encoder model path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lsh-d", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    calibrate_threshold_from_corpus(
        input=args.input,
        output=args.output,
        secret_key=args.secret_key,
        model=args.model,
        device=args.device,
        fpr=args.fpr,
        embed_dim=args.embed_dim,
        lsh_d=args.lsh_d,
        gamma=args.gamma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_calibrate_threshold_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Collapse `scripts/calibrate.py` to a deprecation proxy for both subcommands**

Overwrite `scripts/calibrate.py` with:

```python
#!/usr/bin/env python
"""Deprecated wrapper. Use:

  scripts/build_entropy_profile.py   for build-entropy-profile
  scripts/calibrate_threshold.py     for calibrate-threshold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watermark calibration utilities (deprecated wrapper).",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_profile = subparsers.add_parser(
        "build-entropy-profile",
        help="DEPRECATED: use scripts/build_entropy_profile.py",
    )
    build_profile.add_argument("--input-log", required=True)
    build_profile.add_argument("--output", required=True)
    build_profile.add_argument("--language", required=True)
    build_profile.add_argument("--model-family", required=True)
    build_profile.add_argument("--strategy", default="piecewise_quantile")
    build_profile.add_argument("--profile-id", default=None)

    calibrate = subparsers.add_parser(
        "calibrate-threshold",
        help="DEPRECATED: use scripts/calibrate_threshold.py",
    )
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--fpr", type=float, default=0.01)
    calibrate.add_argument("--secret-key", required=True)
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--embed-dim", type=int, default=128)
    calibrate.add_argument("--lsh-d", type=int, default=3)
    calibrate.add_argument("--gamma", type=float, default=0.5)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help(sys.stderr)
        parser.exit(2)
    if argv[0] not in {"build-entropy-profile", "calibrate-threshold", "-h", "--help"}:
        argv = ["calibrate-threshold", *argv]
    return parser.parse_args(argv)


def _proxy_build_entropy_profile(args: argparse.Namespace) -> int:
    print(
        "[deprecated] scripts/calibrate.py build-entropy-profile is deprecated; "
        "use scripts/build_entropy_profile.py instead.",
        file=sys.stderr,
    )
    from wfcllm.watermark.adaptive_gamma.calibrate import (
        build_entropy_profile_from_log,
    )
    try:
        build_entropy_profile_from_log(
            input_log=args.input_log,
            output=args.output,
            language=args.language,
            model_family=args.model_family,
            strategy=args.strategy,
            profile_id=args.profile_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    return 0


def _proxy_calibrate_threshold(args: argparse.Namespace) -> int:
    print(
        "[deprecated] scripts/calibrate.py calibrate-threshold is deprecated; "
        "use scripts/calibrate_threshold.py instead.",
        file=sys.stderr,
    )
    from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus
    calibrate_threshold_from_corpus(
        input=args.input,
        output=args.output,
        secret_key=args.secret_key,
        model=args.model,
        device=args.device,
        fpr=args.fpr,
        embed_dim=args.embed_dim,
        lsh_d=args.lsh_d,
        gamma=args.gamma,
    )
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "build-entropy-profile":
        return _proxy_build_entropy_profile(args)
    if args.command == "calibrate-threshold":
        return _proxy_calibrate_threshold(args)
    raise SystemExit(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Re-run the runner + CLI test suite to confirm everything still works**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_calibrate_threshold_runner.py tests/integration/test_calibrate_threshold_cli.py tests/integration/test_extract_calibration_shims.py -v`
Expected: ALL PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/calibrate_threshold.py scripts/calibrate.py tests/integration/test_calibrate_threshold_cli.py
git commit -m "feat: add scripts/calibrate_threshold.py + proxy old scripts/calibrate.py"
```

---

## Task 6: Switch internal callers to canonical paths

**Files:**
- Modify: `wfcllm/extract/__init__.py`
- Modify: `wfcllm/cli/runners.py:495` (single import inside `run_extract`)
- Modify: `wfcllm/cli/runners.py:607` (single import inside `run_generate_negative`)
- Modify: `scripts/generate_negative_corpus.py:51` (single import inside `main`)
- Modify: `wfcllm/extract/calibration/__init__.py` (populate re-exports)

This task switches every first-party caller off the deprecated paths. Test code under `tests/` is intentionally left alone — its job is to keep exercising the shims.

- [ ] **Step 1: Update `wfcllm/extract/calibration/__init__.py` to re-export the public surface**

Overwrite `wfcllm/extract/calibration/__init__.py` with:

```python
"""Calibration subpackage: FPR threshold + negative corpus generation."""

from wfcllm.extract.calibration.negative_corpus import (
    NegativeCorpusConfig,
    NegativeCorpusGenerator,
)
from wfcllm.extract.calibration.threshold import ThresholdCalibrator

__all__ = [
    "ThresholdCalibrator",
    "NegativeCorpusConfig",
    "NegativeCorpusGenerator",
]
```

- [ ] **Step 2: Update `wfcllm/extract/__init__.py` imports**

Apply this exact edit to `wfcllm/extract/__init__.py`:

Replace:
```python
from wfcllm.extract.calibrator import ThresholdCalibrator
```
with:
```python
from wfcllm.extract.calibration.threshold import ThresholdCalibrator
```

Replace:
```python
from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator
```
with:
```python
from wfcllm.extract.calibration.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator
```

The `__all__` list and other imports stay unchanged.

- [ ] **Step 3: Update `wfcllm/cli/runners.py` line 495**

Inside `run_extract`, replace:
```python
        from wfcllm.extract.calibrator import ThresholdCalibrator
```
with:
```python
        from wfcllm.extract.calibration.threshold import ThresholdCalibrator
```

- [ ] **Step 4: Update `wfcllm/cli/runners.py` line 607**

Inside `run_generate_negative`, replace:
```python
    from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator
```
with:
```python
    from wfcllm.extract.calibration.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator
```

- [ ] **Step 5: Update `scripts/generate_negative_corpus.py` line 51**

Inside `main`, replace:
```python
    from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator
```
with:
```python
    from wfcllm.extract.calibration.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator
```

- [ ] **Step 6: Repoint `tests/extract/test_negative_corpus.py` patches to the canonical module**

The three `patch(...)` calls at lines 110, 136, 169 currently target the old path, where `load_prompts` / `load_reference_solutions` no longer live as module-level globals after the move. Update all three:

Replace:
```python
patch("wfcllm.extract.negative_corpus.load_reference_solutions", return_value=references)
```
with:
```python
patch("wfcllm.extract.calibration.negative_corpus.load_reference_solutions", return_value=references)
```

Replace (two occurrences):
```python
patch("wfcllm.extract.negative_corpus.load_prompts", return_value=prompts)
```
with:
```python
patch("wfcllm.extract.calibration.negative_corpus.load_prompts", return_value=prompts)
```

The test file's `from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator` at line 11 stays unchanged — it intentionally exercises the shim, and the `*` re-export covers it.

- [ ] **Step 7: Confirm no remaining first-party imports of the old paths**

Run:
```bash
grep -rn "from wfcllm.extract.calibrator\|from wfcllm.extract.negative_corpus\|wfcllm.extract.calibrator\|wfcllm.extract.negative_corpus" wfcllm/ scripts/
```
Expected output: only the two shim files themselves (`wfcllm/extract/calibrator.py`, `wfcllm/extract/negative_corpus.py`) printing their own internal `from wfcllm.extract.calibration.<NEW> import ...` lines. No other matches under `wfcllm/` or `scripts/`.

- [ ] **Step 8: Run extract + cli tests to confirm canonical path imports work**

Run:
```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ tests/integration/test_extract_calibration_shims.py tests/integration/test_calibrate_threshold_runner.py tests/integration/test_calibrate_threshold_cli.py tests/integration/test_cli_resolution.py -v
```
Expected: ALL PASS — including the three previously-deselected `TestNegativeCorpusGeneratorRun` tests, now that their patch targets are correct.

- [ ] **Step 9: Commit**

```bash
git add wfcllm/extract/calibration/__init__.py wfcllm/extract/__init__.py wfcllm/cli/runners.py scripts/generate_negative_corpus.py tests/extract/test_negative_corpus.py
git commit -m "refactor: switch first-party callers to wfcllm.extract.calibration canonical paths"
```

---

## Task 7: Final verification — full test suite + Phase 4 acceptance criteria

**Files:** none (verification-only)

- [ ] **Step 1: Run the full test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -q`
Expected: ALL PASS, no `DeprecationWarning` triggered as an error (warnings are emitted by the shim tests but pytest doesn't treat `DeprecationWarning` as failure under the project's default config).

- [ ] **Step 2: Confirm spec §4.3 file layout**

Run:
```bash
ls -1 wfcllm/extract/calibration/
```
Expected output:
```
__init__.py
negative_corpus.py
runner.py
threshold.py
```

- [ ] **Step 3: Confirm `scripts/` rename intent**

Run:
```bash
ls -1 scripts/calibrate*.py scripts/generate_negative_corpus.py
```
Expected output:
```
scripts/calibrate.py
scripts/calibrate_threshold.py
scripts/generate_negative_corpus.py
```

`scripts/calibrate.py` survives as a deprecation proxy (per spec §8.2 transition policy); `scripts/generate_negative_corpus.py` keeps its name (per spec §4.3 — "保留旧名 (CLI 习惯)").

- [ ] **Step 4: Confirm no first-party caller of old import paths remains**

Run:
```bash
grep -rn "from wfcllm.extract.calibrator import\|from wfcllm.extract.negative_corpus import" wfcllm/ scripts/ | grep -v "^wfcllm/extract/calibrator.py:\|^wfcllm/extract/negative_corpus.py:"
```
Expected output: empty.

- [ ] **Step 5: Phase 4 done — no commit**

This task only verifies; nothing to add to git.

---

## Self-Review

This section is the spec coverage / placeholder / type consistency check the writing-plans skill mandates. It is part of the plan document so reviewers can re-run it.

**Spec §4.3 coverage:**

| Spec requirement | Implemented in |
| ---------------- | -------------- |
| `wfcllm/extract/calibration/negative_corpus.py` ← `extract/negative_corpus.py` | Task 3 |
| `wfcllm/extract/calibration/threshold.py` ← `extract/calibrator.py` | Task 2 |
| `wfcllm/extract/calibration/runner.py` ← `scripts/generate_negative_corpus.py` 核心 | **Note:** spec line 273 says "scripts/generate_negative_corpus.py 核心". The Phase-3-style precedent (`scripts/build_entropy_profile.py` ↔ `wfcllm/watermark/adaptive_gamma/calibrate.py`) lifts the *new CLI's* core, which for Phase 4 is `scripts/calibrate.py:_calibrate_threshold` (the existing `scripts/generate_negative_corpus.py` is already a thin wrapper around `NegativeCorpusGenerator`, leaving no separate core to lift). Task 4 lifts the threshold-calibration core into `runner.py`. Negative-corpus generation core remains in `negative_corpus.py:NegativeCorpusGenerator` — no separate runner needed because `scripts/generate_negative_corpus.py` and `wfcllm/cli/runners.py:run_generate_negative` already share that class directly. |
| `scripts/generate_negative_corpus.py` keeps name | Task 6 step 5 only changes its imports |
| `scripts/calibrate_threshold.py` new name | Task 5 |
| Deprecation shims | Tasks 2, 3, 5 |

**Spec §10 step 4 coverage:** "搬负语料 + 阈值校准 + scripts 改名，加 shim" — all four covered: Tasks 3 (负语料 move), 2 (阈值校准 move), 5 (scripts rename), 2+3 (shim).

**Out-of-scope items intentionally excluded** (reasons given in plan header): orchestration phase registration (§6.2), prereq auto-trigger (§6.3) — both depend on later phases not yet in scope.

**Placeholder scan:** No `TBD`, `TODO`, `implement later`, or `Similar to Task N` strings. Every step that mutates code shows the full final content. The `<NEW>` / `<OLD>` / `<SYMBOLS_USED_BY_TESTS>` placeholders in the *Conventions* section are template documentation for the reader, not steps to be executed — every actual shim in Tasks 2 and 3 spells out the concrete content.

**Type consistency:** `calibrate_threshold_from_corpus` is defined in Task 4 with kwargs `input, output, secret_key, model, device, fpr, embed_dim, lsh_d, gamma`. Task 5 step 3 calls it with the same kwargs. Task 5 step 5 (the proxy) calls it with the same kwargs. The runner test in Task 4 uses the same kwargs. `ThresholdCalibrator`, `NegativeCorpusConfig`, `NegativeCorpusGenerator` keep their existing public signatures (this plan moves their definitions, never edits them). The shim re-exports preserve those exact symbols.

**Backwards-compat sanity:** `tests/extract/test_calibrator.py:12` (`from wfcllm.extract.calibrator import ThresholdCalibrator`), `tests/extract/test_negative_corpus.py:11` (`from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator`), and `tests/test_run_config.py:231,681,816` (`monkeypatch.setattr("wfcllm.extract.calibrator.ThresholdCalibrator", ...)`) all keep working through the shim's explicit re-exports (Tasks 2 step 2 and 3 step 2).

**Known shim limitation, deliberately accepted:** `unittest.mock.patch("wfcllm.extract.negative_corpus.load_prompts", ...)` style patches are *not* preserved by the shim because the consumer (`NegativeCorpusGenerator.run()`) resolves these names via the new canonical module's globals, so patching the shim's binding has no effect on it. The three test sites that did this in `tests/extract/test_negative_corpus.py` are repointed to the canonical path in Task 6 step 6 — a one-character-per-site `patch(...)` target update, no logic change.
