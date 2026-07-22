# Multi-language Gated Experiment Matrix Implementation Plan

> **For Codex:** Follow the repository TDD and verification rules. Preserve the
> existing uncommitted C++/Java adapter work and stage only files owned by this
> plan.

**Goal:** Add eight server-ready full/fast experiment entry points and their
explicit no-carrier configurations for Python/HumanEval, Python/MBPP,
C++/HumanEvalPack, and Java/HumanEvalPack.

**Architecture:** Thin public shell wrappers delegate to one shared runner.
Configuration and CLI carry explicit language/dataset identity. A small
preflight module validates the matrix and rejects carrier configs before any
costly phase. Dataset loading is routed through the existing adapter registry.
Language-specific generation/detection support is enabled only where a shared
formal extractor and certified rewrite path exist; unsupported combinations
fail early and truthfully.

**Tech Stack:** Bash, JSON, Python 3, argparse, pytest, Tree-sitter adapters.

---

## Task 1: Specify the matrix contract with failing tests

**Files:**

- Create: `tests/integration/test_experiment_matrix.py`
- Create: `tests/cli/test_experiment_preflight.py`

**Steps:**

1. Add a parameterized test enumerating exactly the eight wrapper/config pairs.
2. Assert every wrapper delegates to the shared runner and fixes language,
   dataset, profile, and config.
3. Assert every config is valid JSON, declares matching generation identity,
   uses `semantic_lsh`, and never uses `keyed_text_region`.
4. Add preflight tests for valid pairs, invalid pairs, mismatched wrapper/config
   identity, and carrier rejection.
5. Run the new tests and record the expected missing-file/import failures.

## Task 2: Add explicit matrix preflight

**Files:**

- Create: `wfcllm/cli/experiment_preflight.py`
- Modify: `wfcllm/cli/arguments.py`
- Test: `tests/cli/test_experiment_preflight.py`

**Steps:**

1. Implement immutable supported-pair/profile constants.
2. Validate language, dataset, profile, generation config identity,
   `semantic_lsh`, and no-carrier invariants.
3. Add `--language` and extend `--dataset` with `humanevalpack` without changing
   legacy defaults.
4. Run focused tests until green.

## Task 3: Route generation samples through dataset adapters

**Files:**

- Modify: `wfcllm/cli/runners.py`
- Create or modify: `tests/cli/test_gated_generation_dataset_selection.py`

**Steps:**

1. Add failing unit tests using fake dataset adapters for explicit language
   selection, normalized `id`/`prompt` rows, and unsupported-pair rejection.
2. Extract a small helper that loads samples through `wfcllm.datasets.get`.
3. Preserve sample offset/limit behavior after normalization.
4. Use configured language as the source of truth and reject CLI/config
   disagreement.
5. Run focused tests until green.

## Task 4: Add eight explicit no-carrier configs

**Files:**

- Create: `configs/wfcllm/experiments/python_humaneval_full.json`
- Create: `configs/wfcllm/experiments/python_humaneval_fast.json`
- Create: `configs/wfcllm/experiments/python_mbpp_full.json`
- Create: `configs/wfcllm/experiments/python_mbpp_fast.json`
- Create: `configs/wfcllm/experiments/cpp_humanevalpack_full.json`
- Create: `configs/wfcllm/experiments/cpp_humanevalpack_fast.json`
- Create: `configs/wfcllm/experiments/java_humanevalpack_full.json`
- Create: `configs/wfcllm/experiments/java_humanevalpack_fast.json`
- Test: `tests/integration/test_experiment_matrix.py`

**Steps:**

1. Derive full configs from the validated no-carrier semantic-LSH contract.
2. Derive fast configs from the same semantic method while setting explicit
   unvalidated/fast markers and reduced gate-data scale.
3. Set dataset finalizers per dataset; never use the HumanEval finalizer for
   MBPP or HumanEvalPack.
4. Run JSON and contract tests until green.

## Task 5: Add the shared runner and eight entry scripts

**Files:**

- Create: `scripts/experiments/run_gated_experiment.sh`
- Create: the eight public wrappers listed in the design spec
- Modify: `tests/integration/test_experiment_matrix.py`

**Steps:**

1. Implement environment and config preflight before key creation or training.
2. Implement the full phase sequence using separate pilot/full catalogs.
3. Implement the fast sequence using the no-carrier fast config and bounded
   calibration input.
4. Retain current server conda-path overrides and offline environment flags.
5. Give every wrapper a collision-free default experiment root.
6. Run shell syntax and focused integration tests until green.

## Task 6: Make unsupported C++/Java method paths explicit

**Files:**

- Modify only the smallest relevant generation/detection selection modules.
- Add focused tests under `tests/generation/` and `tests/detection/`.

**Steps:**

1. Add failing tests proving C++/Java never instantiate the Python AST
   rewriter or Python-only extractor.
2. Select a language-aware extractor/rewrite path only if the corresponding
   adapter exposes the required certified capability.
3. Otherwise raise a preflight/runtime error naming the missing capability and
   required rewrite-model path.
4. Do not introduce whitespace/keyed-text fallback.
5. Run focused tests until green.

## Task 7: Verification and handoff

**Files:**

- All files changed above.

**Steps:**

1. Run `bash -n scripts/experiments/*.sh`.
2. Parse all eight JSON files with `python -m json.tool`.
3. Run focused matrix, CLI, dataset, language, generation, and detection tests
   with offline environment flags.
4. Run integration tests covering gated wrappers.
5. Run `conda run -n WFCLLM python -m compileall wfcllm run.py scripts`.
6. Run `git diff --check` and inspect `git status --short` to ensure unrelated
   user changes remain unstaged.
7. Report exact commands, results, and any intentionally fail-fast C++/Java
   limitation without claiming a GPU experiment was executed locally.
