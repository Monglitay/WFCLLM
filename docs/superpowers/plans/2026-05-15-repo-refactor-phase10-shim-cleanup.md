# Phase 10: Deprecation Shim Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete all 37 deprecation shim files introduced in Phases 2-8, retarget every first-party caller (production + test) to the canonical import path, and remove shim-testing test files that are no longer needed. One-shot commit per spec §10 step 10: "上述全部稳定后，一次性 commit 删除所有 deprecation shim。"

**Architecture:** Pure import-path rewrite + file deletion. No algorithm changes, no new logic. Each `from wfcllm.<old_path> import X` becomes `from wfcllm.<canonical_path> import X`.

**Tech Stack:** Python 3.10+, pytest, sed (bulk retargeting). No new external deps.

**Spec reference:** `docs/superpowers/specs/2026-05-14-repo-refactor-design.md` §10 step 10, §8.2.

**Out of scope:**
- `experiment/`, `tools/`, `scripts/` — not touched per project rules
- `wfcllm/watermark/interceptor.py` line 208 (`get_pre_event_state()` deprecation) — this is a method-level deprecation inside a live file, NOT a shim file
- `wfcllm/lang/python/transform/__init__.py` line 1 comment — just a docstring noting provenance, not a shim

---

## Shim Inventory (37 files to delete)

**Group A: `wfcllm/common/` shims (21 files) → canonical `wfcllm.lang.python.*` / `wfcllm.datasets.*` / `wfcllm.evaluation.*`**

| Shim file | Canonical path |
|---|---|
| `wfcllm/common/ast_parser.py` | `wfcllm.lang.python.parser` |
| `wfcllm/common/dataset_loader.py` | `wfcllm.datasets.loaders.local` |
| `wfcllm/common/offline_code_eval.py` | `wfcllm.evaluation.code_execution` |
| `wfcllm/common/transform/__init__.py` | `wfcllm.lang.python.transform` |
| `wfcllm/common/transform/base.py` | `wfcllm.lang.python.transform.base` |
| `wfcllm/common/transform/engine.py` | `wfcllm.lang.python.transform.engine` |
| `wfcllm/common/transform/positive/__init__.py` | `wfcllm.lang.python.transform.positive` |
| `wfcllm/common/transform/positive/api_calls.py` | `wfcllm.lang.python.transform.positive.api_calls` |
| `wfcllm/common/transform/positive/control_flow.py` | `wfcllm.lang.python.transform.positive.control_flow` |
| `wfcllm/common/transform/positive/expression_logic.py` | `wfcllm.lang.python.transform.positive.expression_logic` |
| `wfcllm/common/transform/positive/formatting.py` | `wfcllm.lang.python.transform.positive.formatting` |
| `wfcllm/common/transform/positive/identifier.py` | `wfcllm.lang.python.transform.positive.identifier` |
| `wfcllm/common/transform/positive/syntax_init.py` | `wfcllm.lang.python.transform.positive.syntax_init` |
| `wfcllm/common/transform/negative/__init__.py` | `wfcllm.lang.python.transform.negative` |
| `wfcllm/common/transform/negative/api_calls.py` | `wfcllm.lang.python.transform.negative.api_calls` |
| `wfcllm/common/transform/negative/control_flow.py` | `wfcllm.lang.python.transform.negative.control_flow` |
| `wfcllm/common/transform/negative/data_structure.py` | `wfcllm.lang.python.transform.negative.data_structure` |
| `wfcllm/common/transform/negative/exception.py` | `wfcllm.lang.python.transform.negative.exception` |
| `wfcllm/common/transform/negative/expression_logic.py` | `wfcllm.lang.python.transform.negative.expression_logic` |
| `wfcllm/common/transform/negative/identifier.py` | `wfcllm.lang.python.transform.negative.identifier` |
| `wfcllm/common/transform/negative/system.py` | `wfcllm.lang.python.transform.negative.system` |

**Group B: `wfcllm/extract/` shims (3 files) → canonical `wfcllm.extract.calibration.*` / `wfcllm.evaluation.*`**

| Shim file | Canonical path |
|---|---|
| `wfcllm/extract/calibrator.py` | `wfcllm.extract.calibration.threshold` |
| `wfcllm/extract/negative_corpus.py` | `wfcllm.extract.calibration.negative_corpus` |
| `wfcllm/extract/offline_analysis.py` | `wfcllm.evaluation.detection_report` |

**Group C: `wfcllm/watermark/` adaptive-gamma shims (3 files)**

| Shim file | Canonical path |
|---|---|
| `wfcllm/watermark/entropy.py` | `wfcllm.watermark.adaptive_gamma.entropy` |
| `wfcllm/watermark/entropy_profile.py` | `wfcllm.watermark.adaptive_gamma.profile` |
| `wfcllm/watermark/gamma_schedule.py` | `wfcllm.watermark.adaptive_gamma.schedule` |

**Group D: `wfcllm/watermark/token_channel/` shims (9 files)**

| Shim file | Canonical path |
|---|---|
| `wfcllm/watermark/token_channel/config.py` | `wfcllm.watermark.token_channel.core.config` |
| `wfcllm/watermark/token_channel/features.py` | `wfcllm.watermark.token_channel.core.features` |
| `wfcllm/watermark/token_channel/model.py` | `wfcllm.watermark.token_channel.core.model` |
| `wfcllm/watermark/token_channel/protocol.py` | `wfcllm.watermark.token_channel.core.protocol` |
| `wfcllm/watermark/token_channel/teacher.py` | `wfcllm.watermark.token_channel.training.teacher` |
| `wfcllm/watermark/token_channel/train.py` | `wfcllm.watermark.token_channel.training.trainer` |
| `wfcllm/watermark/token_channel/train_corpus.py` | `wfcllm.watermark.token_channel.training.corpus` |
| `wfcllm/watermark/token_channel/train_corpus_streaming.py` | `wfcllm.watermark.token_channel.training.corpus_streaming` |
| `wfcllm/watermark/token_channel/train_workflow.py` | `wfcllm.watermark.token_channel.training.workflow` |

**Group E: `wfcllm/watermark/generator.py` (1 file)**

| Shim file | Canonical path |
|---|---|
| `wfcllm/watermark/generator.py` | `wfcllm.watermark.orchestrator` |

---

## Caller Retargeting Summary

**Production code callers (wfcllm/):**

| File | Old import | New import |
|---|---|---|
| `wfcllm/common/block_contract.py:10` | `from wfcllm.common.ast_parser import extract_statement_blocks` | `from wfcllm.lang.python.parser import extract_statement_blocks` |
| `wfcllm/encoder/train.py:18` | `from wfcllm.common.ast_parser import extract_statement_blocks` | `from wfcllm.lang.python.parser import extract_statement_blocks` |
| `wfcllm/encoder/train.py:19` | `from wfcllm.common.transform.engine import TransformEngine` | `from wfcllm.lang.python.transform.engine import TransformEngine` |
| `wfcllm/encoder/train.py:20` | `from wfcllm.common.transform.positive import get_all_positive_rules` | `from wfcllm.lang.python.transform.positive import get_all_positive_rules` |
| `wfcllm/encoder/train.py:21` | `from wfcllm.common.transform.negative import get_all_negative_rules` | `from wfcllm.lang.python.transform.negative import get_all_negative_rules` |
| `wfcllm/evaluation/dual_channel.py:23` | `from wfcllm.common.dataset_loader import load_reference_solutions` | `from wfcllm.datasets.loaders.local import load_reference_solutions` |
| `wfcllm/extract/calibration/negative_corpus.py:14` | `from wfcllm.common.dataset_loader import (...)` | `from wfcllm.datasets.loaders.local import (...)` |
| `wfcllm/extract/calibration/threshold.py:7` | `from wfcllm.common.ast_parser import extract_statement_blocks` | `from wfcllm.lang.python.parser import extract_statement_blocks` |
| `wfcllm/extract/detector.py:8` | `from wfcllm.common.ast_parser import extract_statement_blocks` | `from wfcllm.lang.python.parser import extract_statement_blocks` |
| `wfcllm/extract/dp_selector.py:7` | `from wfcllm.common.ast_parser import StatementBlock` | `from wfcllm.lang.python.parser import StatementBlock` |
| `wfcllm/extract/scorer.py:5` | `from wfcllm.common.ast_parser import StatementBlock` | `from wfcllm.lang.python.parser import StatementBlock` |
| `wfcllm/watermark/adaptive_gamma/entropy.py:11` | `from wfcllm.common.ast_parser import PythonParser` | `from wfcllm.lang.python.parser import PythonParser` |
| `wfcllm/watermark/interceptor.py:10` | `from wfcllm.common.ast_parser import (...)` | `from wfcllm.lang.python.parser import (...)` |
| `wfcllm/watermark/pipeline.py:21` | `from wfcllm.watermark.generator import WatermarkGenerator` | `from wfcllm.watermark.orchestrator import WatermarkGenerator` |
| `wfcllm/watermark/pipeline.py:22` | `from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts` | `from wfcllm.datasets.loaders.local import SUPPORTED_DATASETS, load_prompts` |
| `wfcllm/watermark/semantic_channel.py:13` | `from wfcllm.common.ast_parser import extract_statement_blocks` | `from wfcllm.lang.python.parser import extract_statement_blocks` |
| `wfcllm/watermark/token_channel/training/corpus.py:15` | `from wfcllm.common.transform.engine import TransformEngine` | `from wfcllm.lang.python.transform.engine import TransformEngine` |
| `wfcllm/watermark/token_channel/training/corpus.py:16` | `from wfcllm.common.transform.positive import get_all_positive_rules` | `from wfcllm.lang.python.transform.positive import get_all_positive_rules` |
| `wfcllm/watermark/token_channel/training/workflow.py:16` | `from wfcllm.common.dataset_loader import load_reference_solutions` | `from wfcllm.datasets.loaders.local import load_reference_solutions` |
| `wfcllm/watermark/token_channel/training/teacher.py:21` | `from wfcllm.watermark.token_channel.teacher import batch_extract_teacher_rows` | (self-import via shim — delete these lines, they're inside the canonical file) |
| `wfcllm/watermark/token_channel/training/teacher.py:42` | `from wfcllm.watermark.token_channel.teacher import extract_teacher_rows` | (same — delete) |
| `wfcllm/cli/runners.py:212` | `from wfcllm.watermark.generator import WatermarkGenerator` | `from wfcllm.watermark.orchestrator import WatermarkGenerator` |

**Test files to delete (shim-testing tests no longer needed):**

- `tests/watermark/test_generator_shim.py` — tests the generator.py shim
- `tests/integration/test_token_channel_layout_shims.py` — tests token_channel shims
- `tests/integration/test_adaptive_gamma_shims.py` — tests adaptive_gamma shims
- `tests/integration/test_evaluation_shims.py` — tests evaluation shims
- `tests/integration/test_extract_calibration_shims.py` — tests extract/calibration shims
- `tests/integration/test_common_lang_shims.py` — tests common→lang shims

---

## Tasks

### Task 1: Retarget production code callers of `common/` shims

**Files:**
- Modify: `wfcllm/common/block_contract.py:10`
- Modify: `wfcllm/encoder/train.py:18-21`
- Modify: `wfcllm/evaluation/dual_channel.py:23`
- Modify: `wfcllm/extract/calibration/negative_corpus.py:14`
- Modify: `wfcllm/extract/calibration/threshold.py:7`
- Modify: `wfcllm/extract/detector.py:8`
- Modify: `wfcllm/extract/dp_selector.py:7`
- Modify: `wfcllm/extract/scorer.py:5`
- Modify: `wfcllm/watermark/adaptive_gamma/entropy.py:11`
- Modify: `wfcllm/watermark/interceptor.py:10`
- Modify: `wfcllm/watermark/pipeline.py:22`
- Modify: `wfcllm/watermark/semantic_channel.py:13`
- Modify: `wfcllm/watermark/token_channel/training/corpus.py:15-16`
- Modify: `wfcllm/watermark/token_channel/training/workflow.py:16`

- [ ] **Step 1: Bulk sed retarget**

```bash
find wfcllm/ -name "*.py" -exec grep -l "from wfcllm.common.ast_parser" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.ast_parser/from wfcllm.lang.python.parser/g'

find wfcllm/ -name "*.py" -exec grep -l "from wfcllm.common.dataset_loader" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.dataset_loader/from wfcllm.datasets.loaders.local/g'

find wfcllm/ -name "*.py" -exec grep -l "from wfcllm.common.offline_code_eval" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.offline_code_eval/from wfcllm.evaluation.code_execution/g'

find wfcllm/ -name "*.py" -exec grep -l "from wfcllm.common.transform" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.transform/from wfcllm.lang.python.transform/g'
```

Exclude the shim files themselves (they'll be deleted later).

- [ ] **Step 2: Verify no remaining references in production code**

```bash
grep -rn "from wfcllm.common.ast_parser\|from wfcllm.common.dataset_loader\|from wfcllm.common.offline_code_eval\|from wfcllm.common.transform" wfcllm/ --include="*.py" | grep -v "wfcllm/common/"
```

Expected: no output.

- [ ] **Step 3: Quick smoke test**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "from wfcllm.watermark.pipeline import WatermarkPipeline; from wfcllm.encoder.train import ContrastiveTrainer; print('OK')"`
Expected: prints "OK" without import errors.

- [ ] **Step 4: Commit**

```bash
git add wfcllm/
git commit -m "refactor: retarget production code from common/ shims to canonical paths"
```

---

### Task 2: Retarget test callers of `common/` shims

**Files:** All test files under `tests/` that import from `wfcllm.common.ast_parser`, `wfcllm.common.dataset_loader`, `wfcllm.common.offline_code_eval`, or `wfcllm.common.transform.*`.

- [ ] **Step 1: Bulk sed retarget**

```bash
find tests/ -name "*.py" -exec grep -l "from wfcllm.common.ast_parser" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.ast_parser/from wfcllm.lang.python.parser/g'

find tests/ -name "*.py" -exec grep -l "from wfcllm.common.dataset_loader" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.dataset_loader/from wfcllm.datasets.loaders.local/g'

find tests/ -name "*.py" -exec grep -l "from wfcllm.common.offline_code_eval" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.offline_code_eval/from wfcllm.evaluation.code_execution/g'

find tests/ -name "*.py" -exec grep -l "from wfcllm.common.transform" {} \; | \
  xargs sed -i 's/from wfcllm\.common\.transform/from wfcllm.lang.python.transform/g'
```

- [ ] **Step 2: Verify no remaining references**

```bash
grep -rn "from wfcllm.common.ast_parser\|from wfcllm.common.dataset_loader\|from wfcllm.common.offline_code_eval\|from wfcllm.common.transform" tests/ --include="*.py"
```

Expected: no output.

- [ ] **Step 3: Run affected test suites**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/common/ tests/extract/ tests/watermark/test_cascade.py tests/watermark/test_generator_integration.py tests/watermark/test_rollback_scenarios.py --tb=short -q 2>&1 | tail -10`
Expected: same pass/fail counts as Phase 9 baseline.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: retarget test imports from common/ shims to canonical paths"
```

---

### Task 3: Retarget callers of `extract/` and `watermark/` adaptive-gamma shims

**Files:**
- Modify: `tests/extract/test_calibrator.py:11-12`
- Modify: `tests/extract/test_negative_corpus.py:11`
- Modify: `tests/extract/test_pipeline.py:17`
- Modify: `tests/common/test_block_contract.py:5`
- Modify: `tests/extract/test_adaptive_roundtrip.py:14-15`
- Modify: `tests/extract/test_alignment.py:10-11`
- Modify: `tests/watermark/test_entropy.py:4`
- Modify: `tests/watermark/test_entropy_profile.py:7`
- Modify: `tests/watermark/test_gamma_schedule.py:3-4`
- Modify: `tests/watermark/test_retry_loop.py:9`

- [ ] **Step 1: Retarget extract/ shim callers**

```bash
find tests/ -name "*.py" -exec grep -l "from wfcllm.extract.calibrator" {} \; | \
  xargs sed -i 's/from wfcllm\.extract\.calibrator/from wfcllm.extract.calibration.threshold/g'

find tests/ -name "*.py" -exec grep -l "from wfcllm.extract.negative_corpus" {} \; | \
  xargs sed -i 's/from wfcllm\.extract\.negative_corpus/from wfcllm.extract.calibration.negative_corpus/g'
```

- [ ] **Step 2: Retarget adaptive-gamma shim callers**

```bash
sed -i 's/from wfcllm\.watermark\.entropy import/from wfcllm.watermark.adaptive_gamma.entropy import/g' \
  tests/common/test_block_contract.py tests/watermark/test_entropy.py

sed -i 's/from wfcllm\.watermark\.entropy_profile/from wfcllm.watermark.adaptive_gamma.profile/g' \
  tests/extract/test_adaptive_roundtrip.py tests/extract/test_alignment.py \
  tests/watermark/test_entropy_profile.py tests/watermark/test_gamma_schedule.py

sed -i 's/from wfcllm\.watermark\.gamma_schedule/from wfcllm.watermark.adaptive_gamma.schedule/g' \
  tests/extract/test_adaptive_roundtrip.py tests/extract/test_alignment.py \
  tests/watermark/test_gamma_schedule.py tests/watermark/test_retry_loop.py
```

- [ ] **Step 3: Run affected tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_calibrator.py tests/extract/test_negative_corpus.py tests/watermark/test_entropy.py tests/watermark/test_entropy_profile.py tests/watermark/test_gamma_schedule.py tests/watermark/test_retry_loop.py --tb=short -q 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: retarget extract/ and adaptive-gamma shim imports to canonical paths"
```

---

### Task 4: Retarget callers of `token_channel/` shims

**Files:** All test files under `tests/watermark/token_channel/` and `tests/integration/` that import from old `wfcllm.watermark.token_channel.{config,features,model,protocol,teacher,train,train_corpus,train_corpus_streaming,train_workflow}` paths. Also `wfcllm/watermark/token_channel/training/teacher.py` (self-imports via shim).

- [ ] **Step 1: Retarget core/ shim callers**

```bash
find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.config import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/config.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.config import/from wfcllm.watermark.token_channel.core.config import/g'

find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.features import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/features.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.features import/from wfcllm.watermark.token_channel.core.features import/g'

find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.model import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/model.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.model import/from wfcllm.watermark.token_channel.core.model import/g'

find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.protocol import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/protocol.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.protocol import/from wfcllm.watermark.token_channel.core.protocol import/g'
```

- [ ] **Step 2: Retarget training/ shim callers**

```bash
find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.teacher import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/teacher.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.teacher import/from wfcllm.watermark.token_channel.training.teacher import/g'

find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.train import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/train.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.train import/from wfcllm.watermark.token_channel.training.trainer import/g'

find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.train_corpus import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/train_corpus.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.train_corpus import/from wfcllm.watermark.token_channel.training.corpus import/g'

find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.train_corpus_streaming import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/train_corpus_streaming.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.train_corpus_streaming import/from wfcllm.watermark.token_channel.training.corpus_streaming import/g'

find tests/ wfcllm/ -name "*.py" -exec grep -l "from wfcllm.watermark.token_channel.train_workflow import" {} \; | \
  grep -v "wfcllm/watermark/token_channel/train_workflow.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.token_channel\.train_workflow import/from wfcllm.watermark.token_channel.training.workflow import/g'
```

- [ ] **Step 3: Fix self-imports in training/teacher.py**

The canonical `wfcllm/watermark/token_channel/training/teacher.py` has lines 21 and 42 that import from the old shim path `wfcllm.watermark.token_channel.teacher`. These are self-referential imports that need to be removed or converted to relative imports. Check what they actually do and fix accordingly.

- [ ] **Step 4: Run affected tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/ tests/integration/test_token_channel_train.py --tb=short -q 2>&1 | tail -10`
Expected: same pass/fail as Phase 9 baseline.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/ tests/
git commit -m "refactor: retarget token_channel/ shim imports to canonical core/ and training/ paths"
```

---

### Task 5: Retarget callers of `watermark/generator.py` shim

**Files:**
- Modify: `wfcllm/cli/runners.py:212`
- Modify: `wfcllm/watermark/pipeline.py:21`
- Modify: `tests/watermark/test_generator.py:14,1550`
- Modify: `tests/watermark/test_generator_integration.py:9`
- Modify: `tests/watermark/test_context.py:10`
- Modify: `tests/watermark/test_embed_metadata_roundtrip.py:6`
- Modify: `tests/watermark/test_pipeline_checkpoint.py:8`
- Modify: `tests/watermark/test_pipeline.py:97,1060`

- [ ] **Step 1: Retarget all callers**

```bash
find wfcllm/ tests/ -name "*.py" -exec grep -l "from wfcllm.watermark.generator import" {} \; | \
  grep -v "wfcllm/watermark/generator.py" | \
  xargs sed -i 's/from wfcllm\.watermark\.generator import/from wfcllm.watermark.orchestrator import/g'
```

- [ ] **Step 2: Verify no remaining references**

```bash
grep -rn "from wfcllm.watermark.generator" wfcllm/ tests/ --include="*.py" | grep -v "wfcllm/watermark/generator.py"
```

Expected: no output.

- [ ] **Step 3: Run affected tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py tests/watermark/test_pipeline.py tests/watermark/test_context.py --tb=short -q 2>&1 | tail -10`
Expected: same pass/fail as Phase 9 baseline.

- [ ] **Step 4: Commit**

```bash
git add wfcllm/ tests/
git commit -m "refactor: retarget watermark/generator.py shim imports to orchestrator"
```

---

### Task 6: Delete all shim files + shim-testing test files

**Files to delete (37 shim files):**
- `wfcllm/common/ast_parser.py`
- `wfcllm/common/dataset_loader.py`
- `wfcllm/common/offline_code_eval.py`
- `wfcllm/common/transform/__init__.py`
- `wfcllm/common/transform/base.py`
- `wfcllm/common/transform/engine.py`
- `wfcllm/common/transform/positive/__init__.py`
- `wfcllm/common/transform/positive/api_calls.py`
- `wfcllm/common/transform/positive/control_flow.py`
- `wfcllm/common/transform/positive/expression_logic.py`
- `wfcllm/common/transform/positive/formatting.py`
- `wfcllm/common/transform/positive/identifier.py`
- `wfcllm/common/transform/positive/syntax_init.py`
- `wfcllm/common/transform/negative/__init__.py`
- `wfcllm/common/transform/negative/api_calls.py`
- `wfcllm/common/transform/negative/control_flow.py`
- `wfcllm/common/transform/negative/data_structure.py`
- `wfcllm/common/transform/negative/exception.py`
- `wfcllm/common/transform/negative/expression_logic.py`
- `wfcllm/common/transform/negative/identifier.py`
- `wfcllm/common/transform/negative/system.py`
- `wfcllm/extract/calibrator.py`
- `wfcllm/extract/negative_corpus.py`
- `wfcllm/extract/offline_analysis.py`
- `wfcllm/watermark/entropy.py`
- `wfcllm/watermark/entropy_profile.py`
- `wfcllm/watermark/gamma_schedule.py`
- `wfcllm/watermark/token_channel/config.py`
- `wfcllm/watermark/token_channel/features.py`
- `wfcllm/watermark/token_channel/model.py`
- `wfcllm/watermark/token_channel/protocol.py`
- `wfcllm/watermark/token_channel/teacher.py`
- `wfcllm/watermark/token_channel/train.py`
- `wfcllm/watermark/token_channel/train_corpus.py`
- `wfcllm/watermark/token_channel/train_corpus_streaming.py`
- `wfcllm/watermark/token_channel/train_workflow.py`
- `wfcllm/watermark/generator.py`

**Shim-testing test files to delete (6 files):**
- `tests/watermark/test_generator_shim.py`
- `tests/integration/test_token_channel_layout_shims.py`
- `tests/integration/test_adaptive_gamma_shims.py`
- `tests/integration/test_evaluation_shims.py`
- `tests/integration/test_extract_calibration_shims.py`
- `tests/integration/test_common_lang_shims.py`

- [ ] **Step 1: Delete shim files**

```bash
git rm wfcllm/common/ast_parser.py wfcllm/common/dataset_loader.py wfcllm/common/offline_code_eval.py
git rm -r wfcllm/common/transform/
git rm wfcllm/extract/calibrator.py wfcllm/extract/negative_corpus.py wfcllm/extract/offline_analysis.py
git rm wfcllm/watermark/entropy.py wfcllm/watermark/entropy_profile.py wfcllm/watermark/gamma_schedule.py
git rm wfcllm/watermark/token_channel/config.py wfcllm/watermark/token_channel/features.py wfcllm/watermark/token_channel/model.py wfcllm/watermark/token_channel/protocol.py wfcllm/watermark/token_channel/teacher.py wfcllm/watermark/token_channel/train.py wfcllm/watermark/token_channel/train_corpus.py wfcllm/watermark/token_channel/train_corpus_streaming.py wfcllm/watermark/token_channel/train_workflow.py
git rm wfcllm/watermark/generator.py
```

- [ ] **Step 2: Delete shim-testing test files**

```bash
git rm tests/watermark/test_generator_shim.py
git rm tests/integration/test_token_channel_layout_shims.py
git rm tests/integration/test_adaptive_gamma_shims.py
git rm tests/integration/test_evaluation_shims.py
git rm tests/integration/test_extract_calibration_shims.py
git rm tests/integration/test_common_lang_shims.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: delete all 37 deprecation shims + 6 shim-testing test files"
```

---

### Task 7: Full suite verification

- [ ] **Step 1: Run full project test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/ tests/extract/ tests/watermark/ tests/ablation/ tests/integration/ tests/common/ tests/lang/ tests/datasets/ --tb=line -q 2>&1 | tail -20`

Expected: Pass count should be LOWER than Phase 9 (because shim-testing tests are deleted) but all remaining tests pass. The 15 baseline failures in encoder/extract/watermark/ablation should remain unchanged. Integration failures should decrease (6 shim-test files deleted = ~30 fewer tests total, some of which were in the 13 integration failures).

- [ ] **Step 2: Verify no import errors**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "
from wfcllm.watermark.orchestrator import WatermarkGenerator
from wfcllm.watermark.semantic_channel import SemanticChannel
from wfcllm.lang.python.parser import extract_statement_blocks
from wfcllm.datasets.loaders.local import load_prompts
from wfcllm.evaluation.code_execution import compute_pass_at_k
from wfcllm.extract.calibration.threshold import ThresholdCalibrator
from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows
print('All canonical imports OK')
"
```

Expected: prints "All canonical imports OK".

- [ ] **Step 3: Verify old paths are truly gone**

```bash
python -c "
import importlib, sys
old_paths = [
    'wfcllm.common.ast_parser',
    'wfcllm.watermark.generator',
    'wfcllm.watermark.entropy',
    'wfcllm.watermark.token_channel.train',
]
for p in old_paths:
    try:
        importlib.import_module(p)
        print(f'FAIL: {p} still importable')
        sys.exit(1)
    except (ImportError, ModuleNotFoundError):
        print(f'OK: {p} correctly removed')
print('All shims confirmed deleted')
"
```

Expected: all paths report "correctly removed".

---

## Self-Review

1. **Spec coverage:** §10 step 10 ("shim 清理：上述全部稳定后，一次性 commit 删除所有 deprecation shim") fully addressed. All 37 shim files identified and scheduled for deletion.

2. **Placeholder scan:** No TBD/TODO. All steps have concrete commands.

3. **Risk assessment:** MEDIUM risk. The bulk sed operations could miss edge cases (e.g., multi-line imports, `import wfcllm.common.ast_parser` without `from`). Task 7 verification catches any misses. The `wfcllm/watermark/interceptor.py` deprecated method is intentionally NOT removed (it's a method-level deprecation, not a file-level shim).

4. **Baseline preservation:** Task 7 verifies the full suite. Expected outcome: fewer total tests (shim-testing tests deleted) but zero new failures in remaining tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-repo-refactor-phase10-shim-cleanup.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
