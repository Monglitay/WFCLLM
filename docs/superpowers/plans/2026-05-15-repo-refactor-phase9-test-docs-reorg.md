# Phase 9: 测试重组 + 文档归档

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic `tests/test_run.py` (1750 lines, 52 tests) and `tests/test_run_config.py` (1007 lines, 20 tests) into focused per-phase integration test modules under `tests/integration/`, then reorganize root `docs/` files into `design/`, `history/`, `progress/`, `references/` subdirectories per spec §4.6 + §4.7.

**Architecture:** Pure file-move + split. No algorithm changes, no new logic. Each test function is copied verbatim into its target file with only the necessary imports adjusted. Shared helpers (`_write_json`, `_write_jsonl`, path constants) are extracted into `tests/integration/conftest.py`. Old files are deleted only after all tests pass from their new locations.

**Tech Stack:** Python 3.10+, pytest, git mv. No new external deps.

**Spec reference:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §4.6 (测试重组), §4.7 (文档重组), §10 step 9.

**Out of scope:**
- Creating `tests/unit/` wrapper (spec mentions it but existing `tests/encoder/`, `tests/watermark/`, `tests/extract/` already serve this role — renaming would break all CI references for zero benefit)
- Moving `tests/lang/`, `tests/datasets/`, `tests/node_entropy/`, `tests/ablation/` — already well-organized
- Touching `experiment/`, `tools/`, `scripts/`
- Changing any test logic or assertions
- Touching `docs/plans/`, `docs/superpowers/`, `docs/experiment/` — already organized

---

## File Structure

**Test split targets (from `tests/test_run.py`):**

| Source class/function | Lines | Target file |
|---|---|---|
| `TestRunState` (5 tests) | 31-76 | `tests/integration/test_orchestration.py` (append) |
| `TestRunGenerateNegative` (3 tests) | 78-174 | `tests/integration/test_generate_negative.py` (new) |
| `TestCLI` — parser tests (3 tests) | 177-269 | `tests/integration/test_cli_resolution.py` (append) |
| `TestCLI` — token-channel-train (23 tests) | 271-1260 | `tests/integration/test_token_channel_train.py` (new) |
| `TestCLI` — status/reset/unknown (3 tests) | 1261-1285 | `tests/integration/test_orchestration.py` (append) |
| `TestCLI` — offline analysis (1 test) | 1286-1391 | `tests/integration/test_orchestration.py` (append) |
| `TestCLI` — compare-only/extract (4 tests) | 1392-1585 | `tests/integration/test_extract_phase.py` (new) |
| `TestRunWatermarkConfigNoFallback` (3 tests) | 1587-1633 | `tests/integration/test_watermark_phase.py` (new) |
| Module-level `test_run_extract_*` (4 tests) | 1635-1750 | `tests/integration/test_extract_phase.py` (new) |

**Test split targets (from `tests/test_run_config.py`):**

| Source function/class | Lines | Target file |
|---|---|---|
| `test_load_config_*` (4 tests) | 17-48 | `tests/integration/test_orchestration.py` (append) |
| `test_parser_*` (3 tests) | 50-84 | `tests/integration/test_cli_resolution.py` (append) |
| `TestBaseConfigFallbackCascade` (3 tests) | 86-106 | `tests/integration/test_watermark_phase.py` |
| `test_base_config_b_restores_*` | 108-123 | `tests/integration/test_watermark_phase.py` |
| `test_run_extract_uses_resolved_gamma_*` | 125-267 | `tests/integration/test_extract_phase.py` |
| `test_run_watermark_parses_token_channel_*` | 269-361 | `tests/integration/test_watermark_phase.py` |
| `test_run_extract_parses_token_channel_*` | 363-483 | `tests/integration/test_extract_phase.py` |
| `test_run_watermark_rejects_non_object_*` | 485-530 | `tests/integration/test_watermark_phase.py` |
| `test_run_extract_uses_adaptive_calibration_*` | 532-707 | `tests/integration/test_extract_phase.py` |
| `test_run_extract_prefers_configured_fpr_*` | 709-839 | `tests/integration/test_extract_phase.py` |
| `test_run_extract_allows_extract_only_*` | 841-958 | `tests/integration/test_extract_phase.py` |
| `test_base_config_includes_adaptive_sections` | 960-987 | `tests/integration/test_watermark_phase.py` |
| `test_humaneval_subset_config_*` | 989-1003 | `tests/integration/test_watermark_phase.py` |
| `test_base_config_defaults_generate_negative_*` | 1005-1007 | `tests/integration/test_generate_negative.py` |

**Docs reorganization:**

| Source | Target | Contents |
|---|---|---|
| `docs/*.pdf` | `docs/references/` | 论文 PDF |
| `docs/*.csv` | `docs/references/` | 变换规则 CSV |
| `docs/基于语句块的语义特征的生成时代码水印方案.md` | `docs/design/` | 核心方案（活文档） |
| `docs/基于LSH的改进方案.md` | `docs/design/` | LSH 改进方案 |
| `docs/水印检测改造.md` | `docs/design/` | 检测改造方案 |
| `docs/token-channel-batch-inference.md` | `docs/design/` | 批推理方案 |
| `docs/watermark重构.md` | `docs/history/` | 已完成重构记录 |
| `docs/修复记录-Top-k-Logits-KL散度计算.md` | `docs/history/` | 已完成修复记录 |
| `docs/组会汇报-*.md` (4 files) | `docs/progress/` | 汇报文档 |

**New files:**

- `tests/integration/conftest.py` — shared helpers (`_write_json`, `_write_jsonl`, path constants)
- `tests/integration/test_generate_negative.py`
- `tests/integration/test_token_channel_train.py`
- `tests/integration/test_extract_phase.py`
- `tests/integration/test_watermark_phase.py`

**Modified files:**

- `tests/integration/test_orchestration.py` — append TestRunState + status/reset/offline-analysis tests
- `tests/integration/test_cli_resolution.py` — append parser tests

**Deleted after verification:**

- `tests/test_run.py`
- `tests/test_run_config.py`

**Reference updates:**

- `CLAUDE.md` — update `docs/基于语句块...` path to `docs/design/基于语句块...`
- `README.md` — update doc path references
- `wfcllm/watermark/token_channel/training/teacher.py` — update `token-channel-batch-inference.md` path

---

## Conventions

**Verbatim copy rule:** Every test function body is copied exactly as-is. Only the import block at the top of each new file is adjusted to pull in the correct modules.

**Shared helpers pattern** (`tests/integration/conftest.py`):

```python
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_PY = PROJECT_ROOT / "run.py"
RUNNERS_PY = PROJECT_ROOT / "wfcllm" / "cli" / "runners.py"
README_MD = PROJECT_ROOT / "README.md"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
```

**Import adjustment rule:** Each target file imports helpers from conftest (pytest auto-discovers fixtures) or uses direct imports. Module-level constants like `PROJECT_ROOT`, `RUN_PY` come from conftest via explicit import:

```python
from tests.integration.conftest import PROJECT_ROOT, RUN_PY, write_json, write_jsonl
```

**Incremental deletion:** Old files are NOT deleted until ALL tests pass from their new locations. The plan deletes them in the final task.

---

## Tasks

### Task 1: Create `tests/integration/conftest.py` with shared helpers

**Files:**
- Create: `tests/integration/conftest.py`

- [ ] **Step 1: Write conftest.py with shared constants and helpers**

```python
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_PY = PROJECT_ROOT / "run.py"
RUNNERS_PY = PROJECT_ROOT / "wfcllm" / "cli" / "runners.py"
README_MD = PROJECT_ROOT / "README.md"

sys.path.insert(0, str(PROJECT_ROOT))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
```

- [ ] **Step 2: Verify import works**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "from tests.integration.conftest import PROJECT_ROOT, write_json; print(PROJECT_ROOT)"`
Expected: prints the project root path without error.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/conftest.py
git commit -m "test: add integration conftest with shared helpers for test_run split"
```

---

### Task 2: Split `TestRunState` + orchestration tests into `test_orchestration.py`

**Files:**
- Modify: `tests/integration/test_orchestration.py` (append)
- Source: `tests/test_run.py:31-76` (TestRunState), `tests/test_run.py:1261-1391` (status/reset/offline-analysis), `tests/test_run_config.py:17-48` (test_load_config_*)

- [ ] **Step 1: Read source sections verbatim**

Read `tests/test_run.py` lines 31-76 (TestRunState), lines 1261-1391 (status/reset/offline-analysis from TestCLI), and `tests/test_run_config.py` lines 17-48 (test_load_config_*).

- [ ] **Step 2: Append TestRunState to test_orchestration.py**

Append the `TestRunState` class (verbatim) at the end of `tests/integration/test_orchestration.py`. Add necessary imports at the top:

```python
from wfcllm.orchestration.state import ALL_PHASES, PHASES, RunStateManager as RunState
from tests.integration.conftest import write_json
```

- [ ] **Step 3: Append status/reset/unknown-phase tests**

Extract `test_status_exits_zero`, `test_reset_exits_zero`, `test_unknown_phase_exits_nonzero`, `test_run_offline_analysis_writes_json_report` from TestCLI and append as standalone functions (they use `subprocess` + `RUN_PY`). Add:

```python
import subprocess
from tests.integration.conftest import PROJECT_ROOT, RUN_PY
```

- [ ] **Step 4: Append test_load_config_* tests**

Copy `test_load_config_returns_dict`, `test_load_config_missing_phase_ok`, `test_load_config_file_not_found`, `test_load_config_invalid_json` verbatim. Add:

```python
from wfcllm.cli.config import load_config
```

- [ ] **Step 5: Run new tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v --tb=short -q 2>&1 | tail -20`
Expected: all appended tests pass (existing tests still pass too).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_orchestration.py
git commit -m "test: move RunState + orchestration tests from test_run into integration"
```

---

### Task 3: Create `tests/integration/test_token_channel_train.py`

**Files:**
- Create: `tests/integration/test_token_channel_train.py`
- Source: `tests/test_run.py:271-1260` (23 token-channel-train tests from TestCLI)

- [ ] **Step 1: Read source section verbatim**

Read `tests/test_run.py` lines 271-1260.

- [ ] **Step 2: Write test_token_channel_train.py**

Create the file with all 23 token-channel-train test methods extracted from TestCLI. Convert from class methods to module-level functions (remove `self` parameter). Keep all imports verbatim:

```python
import json
import sys
from pathlib import Path

import pytest

from tests.integration.conftest import PROJECT_ROOT, RUN_PY, write_json

sys.path.insert(0, str(PROJECT_ROOT))
```

Each test function keeps its exact body — only `self` is removed and fixtures (`tmp_path`, `monkeypatch`, `capsys`) stay as parameters.

- [ ] **Step 3: Run new tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_token_channel_train.py -v --tb=short -q 2>&1 | tail -30`
Expected: all 23 tests pass (or match the Phase 8 baseline — the 2 known failures in token_channel train_workflow are acceptable).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_token_channel_train.py
git commit -m "test: move token-channel-train tests from test_run into integration"
```

---

### Task 4: Create `tests/integration/test_extract_phase.py`

**Files:**
- Create: `tests/integration/test_extract_phase.py`
- Source: `tests/test_run.py:1392-1585` (compare-only/extract from TestCLI), `tests/test_run.py:1635-1750` (module-level extract tests), `tests/test_run_config.py:125-267,363-483,532-839,841-958` (extract config tests)

- [ ] **Step 1: Read source sections verbatim**

Read the relevant line ranges from both files.

- [ ] **Step 2: Write test_extract_phase.py**

Create the file with all extract-related tests. Tests from TestCLI lose `self`; module-level tests and test_run_config tests are already standalone. Imports:

```python
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.integration.conftest import PROJECT_ROOT, RUN_PY, write_json, write_jsonl

sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 3: Run new tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_extract_phase.py -v --tb=short -q 2>&1 | tail -30`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_extract_phase.py
git commit -m "test: move extract-phase tests from test_run/test_run_config into integration"
```

---

### Task 5: Create `tests/integration/test_watermark_phase.py`

**Files:**
- Create: `tests/integration/test_watermark_phase.py`
- Source: `tests/test_run.py:1587-1633` (TestRunWatermarkConfigNoFallback), `tests/test_run_config.py:86-123,269-361,485-530,960-1003` (watermark config tests)

- [ ] **Step 1: Read source sections verbatim**

Read the relevant line ranges.

- [ ] **Step 2: Write test_watermark_phase.py**

Create the file. `TestRunWatermarkConfigNoFallback` methods lose `self`. Config tests are already standalone. Imports:

```python
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.integration.conftest import PROJECT_ROOT, write_json

sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 3: Run new tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_watermark_phase.py -v --tb=short -q 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_watermark_phase.py
git commit -m "test: move watermark-phase tests from test_run/test_run_config into integration"
```

---

### Task 6: Create `tests/integration/test_generate_negative.py` + append CLI parser tests

**Files:**
- Create: `tests/integration/test_generate_negative.py`
- Modify: `tests/integration/test_cli_resolution.py` (append)
- Source: `tests/test_run.py:78-174` (TestRunGenerateNegative), `tests/test_run_config.py:1005-1007`, `tests/test_run.py:177-269` (parser tests from TestCLI), `tests/test_run_config.py:50-84` (test_parser_*)

- [ ] **Step 1: Write test_generate_negative.py**

Create the file with `TestRunGenerateNegative` methods (lose `self`) + `test_base_config_defaults_generate_negative_to_reference_mode`. Imports:

```python
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.integration.conftest import PROJECT_ROOT, write_json

sys.path.insert(0, str(PROJECT_ROOT))
```

The fixture `neg_cfg_file` from TestRunGenerateNegative must be included verbatim.

- [ ] **Step 2: Append parser tests to test_cli_resolution.py**

Append `test_cli_subprocess_invocations_do_not_use_bare_run_py_script_name`, `test_build_parser_parses_resume_argument`, `test_build_parser_accepts_token_channel_flags`, `test_build_parser_accepts_token_channel_train_phase_and_flags` (lose `self`), plus `test_parser_default_config`, `test_parser_custom_config`, `test_parser_accepts_adaptive_watermark_and_extract_flags` from test_run_config.py. Add necessary imports.

- [ ] **Step 3: Run new tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_generate_negative.py tests/integration/test_cli_resolution.py -v --tb=short -q 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_generate_negative.py tests/integration/test_cli_resolution.py
git commit -m "test: move generate-negative + CLI parser tests into integration"
```

---

### Task 7: Delete old test files + verify full suite

**Files:**
- Delete: `tests/test_run.py`
- Delete: `tests/test_run_config.py`

- [ ] **Step 1: Run full integration suite from new locations**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/ -v --tb=short -q 2>&1 | tail -30`
Expected: all tests that were passing before still pass. Known baseline failures (3 pipeline stats + 2 token_channel train_workflow) are acceptable.

- [ ] **Step 2: Delete old files**

```bash
git rm tests/test_run.py tests/test_run_config.py
```

- [ ] **Step 3: Run full project test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --tb=short -q 2>&1 | tail -20`
Expected: 695 passed + 15 failed (exact Phase 8 baseline). If any NEW failures appear, stop and investigate — likely a missing import or helper.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove monolithic test_run.py and test_run_config.py after split"
```

---

### Task 8: Reorganize `docs/` into subdirectories

**Files:**
- Create: `docs/design/`, `docs/history/`, `docs/progress/`, `docs/references/`
- Move: see table in File Structure section above
- Modify: `CLAUDE.md` (update path reference)
- Modify: `README.md` (update path reference)
- Modify: `wfcllm/watermark/token_channel/training/teacher.py` (update comment path)

- [ ] **Step 1: Create target directories and move files**

```bash
mkdir -p docs/design docs/history docs/progress docs/references

git mv "docs/基于语句块的语义特征的生成时代码水印方案.md" docs/design/
git mv "docs/基于LSH的改进方案.md" docs/design/
git mv "docs/水印检测改造.md" docs/design/
git mv docs/token-channel-batch-inference.md docs/design/

git mv "docs/watermark重构.md" docs/history/
git mv "docs/修复记录-Top-k-Logits-KL散度计算.md" docs/history/

git mv "docs/组会汇报-从词法到结构语义的代码水印框架-汇报版.md" docs/progress/
git mv "docs/组会汇报-从词法到结构语义的代码水印框架-精简版.md" docs/progress/
git mv "docs/组会汇报-从词法到结构语义的代码水印框架-进度更新版.md" docs/progress/
git mv "docs/组会汇报-从词法到结构语义的代码水印框架.md" docs/progress/

git mv "docs/Guo和Cheng - 2025 - Practical and Effective Code Watermarking for Large Language Models.pdf" docs/references/
git mv "docs/Hou 等 - 2024 - SemStamp a semantic watermark with paraphrastic robustness for text generation.pdf" docs/references/
git mv "docs/软件工程需求分析-教学用表.pdf" docs/references/
git mv "docs/python代码变换规则.csv" docs/references/
git mv "docs/python代码负变换规则.csv" docs/references/
```

- [ ] **Step 2: Update CLAUDE.md path reference**

In `CLAUDE.md`, change:
```
> 方案文档：`docs/基于语句块的语义特征的生成时代码水印方案.md`
```
to:
```
> 方案文档：`docs/design/基于语句块的语义特征的生成时代码水印方案.md`
```

- [ ] **Step 3: Update README.md path reference**

Find and update any reference to `docs/基于语句块的语义特征的生成时代码水印方案.md` → `docs/design/基于语句块的语义特征的生成时代码水印方案.md`.

- [ ] **Step 4: Update teacher.py comment/docstring path**

In `wfcllm/watermark/token_channel/training/teacher.py`, update the reference to `docs/token-channel-batch-inference.md` → `docs/design/token-channel-batch-inference.md`.

- [ ] **Step 5: Verify no broken references**

```bash
grep -r "docs/基于语句块" . --include="*.py" --include="*.md" | grep -v "docs/design/" | grep -v ".claude/" | grep -v "docs/plans/" | grep -v "docs/superpowers/"
grep -r "docs/token-channel-batch" . --include="*.py" --include="*.md" | grep -v "docs/design/" | grep -v ".claude/" | grep -v "docs/plans/" | grep -v "docs/superpowers/"
```

Expected: no hits (all references updated). Frozen plan files under `docs/plans/` and `docs/superpowers/` are acceptable — they reference historical state.

- [ ] **Step 6: Commit**

```bash
git add -A docs/ CLAUDE.md README.md wfcllm/watermark/token_channel/training/teacher.py
git commit -m "docs: reorganize docs/ into design/ history/ progress/ references/ per spec §4.7"
```

---

## Self-Review

1. **Spec coverage:** §4.6 (测试重组) covered by Tasks 1-7. §4.7 (文档重组) covered by Task 8. §10 step 9 fully addressed.

2. **Placeholder scan:** No TBD/TODO/placeholders. All steps have concrete commands and expected outputs.

3. **Type consistency:** Helper names (`write_json`, `write_jsonl`) match across conftest and all target files. Path constants (`PROJECT_ROOT`, `RUN_PY`, `RUNNERS_PY`, `README_MD`) consistent.

4. **Risk assessment:** This phase is LOW risk — pure file moves with no algorithm changes. The only failure mode is a missing import or helper in a new file, which is caught by Step 3 (run tests) in each task. The spec's mitigation ("每拆一组测试，先在新位置跑通，再删旧文件中对应部分") is followed exactly.

5. **Baseline preservation:** Task 7 Step 3 verifies the full suite matches Phase 8 baseline (695 passed + 15 failed). Zero new regressions expected.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-repo-refactor-phase9-test-docs-reorg.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
