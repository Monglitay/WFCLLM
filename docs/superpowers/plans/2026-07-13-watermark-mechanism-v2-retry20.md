# Watermark Mechanism V2 Retry-20 Implementation Plan

> **For AI agent workers:** Execute inline with `superpowers:executing-plans`; do not create a worktree because the user explicitly required direct work on `codex/watermark-mechanism-v2-retry20`. Use TDD for every behavioral change and commit each major phase.

**Goal:** Build and honestly validate a final-code-only, key-conditioned V2 watermark whose retry-20 generation objective is exactly recoverable by its detector while preserving HumanEval Pass@1.

**Architecture:** Add a small canonical-unit/signature detector beside the v1 detector, inject a V2 full-attempt selector into the existing generation pipeline, and keep the strict four-field final-code input unchanged. V2 calibration and retry ledgers have explicit schemas and fail-fast compatibility checks. A tracked experiment driver consumes the frozen preregistration, writes only public metadata, and resumes through run directories and screen sessions.

**Technical stack:** Python 3.11, dataclasses, `ast`, HMAC-SHA256, existing CodeT5/LSH components, PyTorch, pytest, offline HumanEval execution.

---

## Files and responsibilities

- Create `wfcllm/detection/canonical_v2.py`: canonical unit extraction and schema.
- Create `wfcllm/detection/signature_v2.py`: HMAC target derivation and unit/sample scoring.
- Create `wfcllm/detection/pipeline_v2.py`: v2 config, calibration, detection, serialization, and compatibility.
- Create `wfcllm/generation/quality_v2.py`: static generation-time quality gate.
- Create `wfcllm/generation/selection_v2.py`: retry-20 attempt scoring/selection and audit ledger rows.
- Modify `wfcllm/generation/pipeline.py`: optional attempt selector, sample-ID filtering, and v2 ledger sink without changing default v1 behavior.
- Modify `wfcllm/method/config.py`: explicit sample IDs, selector mode, retry-ledger path, and validation.
- Modify `scripts/wfcllm_generate.py`: explicit v2 selection flags and environment-only secret fallback.
- Create `scripts/wfcllm_v2_detect.py`: v2 split/calibrate/detect/evaluate/replay CLI that refuses v1 artifacts.
- Create `scripts/run_watermark_v2_retry20.py`: preregistered experiment orchestration, metrics, resume metadata, hashes, and paired analysis.
- Create nearby tests under `tests/detection/`, `tests/generation/`, `tests/method/`, `tests/audit/`, and `tests/integration/`.
- Update root reports after each experiment stage; never commit large JSONL/log/model artifacts.

### Task 1: Canonical contract and v2 schema (TDD)

**Files:**
- Create `tests/detection/test_canonical_v2.py`
- Create `tests/detection/test_pipeline_v2.py`
- Create `wfcllm/detection/canonical_v2.py`
- Create `wfcllm/detection/pipeline_v2.py`

- [ ] Write failing tests that extract each AST statement once, preserve source order, deduplicate exact structural/text duplicates, reject malformed code as zero evidence, and keep hashes stable under trailing whitespace.

```python
def test_extract_canonical_units_deduplicates_and_orders() -> None:
    units = extract_canonical_units(FINAL_CODE)
    assert [unit.ordinal for unit in units] == list(range(len(units)))
    assert len({unit.unit_id for unit in units}) == len(units)
    assert all(unit.schema_version == CANONICAL_UNIT_SCHEMA_VERSION for unit in units)
```

- [ ] Run `pytest tests/detection/test_canonical_v2.py -v` and confirm import/behavior failures.
- [ ] Implement immutable `CanonicalUnit` and deterministic extraction using the existing target-function and structure-context helpers.
- [ ] Write failing schema tests: v2 loader rejects v1, unversioned, extra-field, forbidden-key, mismatched key hash, and mismatched public config artifacts.
- [ ] Implement `WFCLLMV2DetectionConfig`, `V2CalibrationArtifact`, strict upper-order threshold, serialization, and compatibility validation.
- [ ] Run both files and confirm green.

### Task 2: Keyed multi-bit signature scoring (TDD)

**Files:**
- Create `tests/detection/test_signature_v2.py`
- Create `wfcllm/detection/signature_v2.py`
- Modify `wfcllm/detection/pipeline_v2.py`

- [ ] Write failing tests for deterministic target bits, key separation, bit agreement, deduplicated robust mean, and bounded scores.

```python
def test_target_signature_is_key_conditioned() -> None:
    left = derive_target_signature(UNIT, secret_key="left", bits=16)
    right = derive_target_signature(UNIT, secret_key="right", bits=16)
    assert left != right
    assert len(left) == len(right) == 16
```

- [ ] Run the test and confirm the missing API failure.
- [ ] Implement HMAC domain-separated targets and score canonical units through the existing verifier/LSH space.
- [ ] Add detector tests proving inputs with sidecar fields are rejected and outputs can be recomputed from final code only.
- [ ] Confirm wrong-key results differ while public artifacts contain only the key hash.

### Task 3: Static quality gate and aligned selection (TDD)

**Files:**
- Create `tests/generation/test_quality_v2.py`
- Create `tests/generation/test_selection_v2.py`
- Create `wfcllm/generation/quality_v2.py`
- Create `wfcllm/generation/selection_v2.py`

- [ ] Write failing tests for syntax, exact target function, signature compatibility, placeholder body, Markdown fence, suspicious tail, all-invalid fallback, score ranking, and deterministic ties.

```python
def test_selector_never_prefers_invalid_candidate_for_score() -> None:
    selected = selector.select([INVALID_HIGH_SCORE, VALID_LOWER_SCORE])
    assert selected.attempt_index == VALID_LOWER_SCORE.attempt_index
```

- [ ] Implement the smallest static gate; do not execute tests or use labels.
- [ ] Implement `V2RetryAttemptSelector` returning the selected result plus audit-only `wfcllm-v2-retry-ledger/v2` rows.
- [ ] Add replay assertion that rescoring the selected `final_code` equals its generation-time score exactly.

### Task 4: Integrate retry selection without changing v1 (TDD)

**Files:**
- Modify `tests/generation/test_evidence_retry_key.py`
- Modify `tests/generation/test_pipeline.py`
- Modify `tests/method/test_evidence_retry_seed7x3_preset.py`
- Modify `wfcllm/generation/pipeline.py`
- Modify `wfcllm/method/config.py`

- [ ] Write failing tests that v1 still selects the old key, v2 receives all 20 attempts, only requested sample IDs run, and v2 retry-ledger rows never enter final-code output.
- [ ] Add optional selector/sample IDs/ledger config with defaults preserving v1 behavior.
- [ ] Validate explicit V2 runs require exactly 20 attempts; do not change the reusable v1 library default used by unit tests.
- [ ] Run generation and method suites.

### Task 5: CLI, replay, and public artifact safety (TDD)

**Files:**
- Modify `tests/integration/test_wfcllm_generate_cli.py`
- Create `tests/integration/test_wfcllm_v2_detect_cli.py`
- Create `tests/audit/test_v2_secret_leakage.py`
- Modify `scripts/wfcllm_generate.py`
- Create `scripts/wfcllm_v2_detect.py`

- [ ] Write failing CLI tests for environment-secret resolution, retry=20 enforcement, v2 flags, v1/v2 artifact rejection, and final-code-only input.
- [ ] Integrate V2 components while loading the encoder/checkpoint/precision once per process.
- [ ] Ensure command summaries redact the secret and public artifacts reject raw `secret_key` recursively.
- [ ] Add `replay` command that takes final code plus v2 calibration/public config and verifies unit/hash/signature/score equality without a sidecar.

### Task 6: Preregistered experiment driver and analysis (TDD)

**Files:**
- Create `tests/integration/test_watermark_v2_experiment_driver.py`
- Create `scripts/run_watermark_v2_retry20.py`

- [ ] Write failing tests for preregistration hash validation, exact task/split lists, resume status, output closure, paired bootstrap determinism, and no held-out access during pilot.
- [ ] Implement subcommands `prepare`, `phase-a`, `pilot`, `analyze-pilot`, `full`, `analyze-full`, and `audit`.
- [ ] Persist public `run_status.json` with run ID, PID/session, log path, start time, completed/failed task counts, GPU snapshot, ETA, and recovery command; do not persist the key.
- [ ] Save raw attempts only in untracked audit-only run directories so the registered ablation can reuse scores without regeneration.

### Task 7: Baseline and implementation verification/commits

- [ ] Run targeted red-green cycles, then:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CONDA_ENVS_PATH=/root/autodl-tmp/conda/envs conda run -n WFCLLM \
pytest tests/generation tests/semantic tests/detection tests/audit tests/integration -q
```

- [ ] Run compileall and tracked/public artifact secret scans.
- [ ] Commit diagnosis/preregistration, implementation/tests, and Phase-A evidence as separate scoped commits.

### Task 8: Phase A contract and replay gate

- [ ] Create a private mode-0600 key outside the repository; processes read it into `WFCLLM_SECRET_KEY` without printing it or placing it in argv.
- [ ] Run 5--10 frozen debug samples with two independent encoder loads.
- [ ] Verify canonical hashes, signatures, raw scores, correct/wrong-key behavior, ordinary negatives, schema rejection, final-code replay 100%, and encoder/checkpoint/precision fingerprints.
- [ ] Do not enter pilot unless all contract checks pass.

### Task 9: Frozen 30-task two-arm pilot

- [ ] Launch Current-R20 and V2-R20 in separate recoverable screen sessions, sequentially if only one GPU is available.
- [ ] Monitor logs, GPU, OOM, stalled output, task closure, and safe single-task restart without changing config.
- [ ] Evaluate Pass posthoc; calibrate on calibration negatives; check pilot FPR only on pilot-development negatives.
- [ ] Report Pass, TPR, FPR, AUROC, score quantiles, no embedding, insufficient, sufficient-low-score, accepted/recovered ratio, objective correlation, time/candidates/retry, and failure taxonomy.
- [ ] If a pilot gate fails, perform at most one evidence-targeted repair and label the rerun a development iteration. Otherwise advance.

### Task 10: Registered ablation and full confirmation

- [ ] Re-score saved V2 raw attempts using v1 accepted-evidence selection; do not regenerate.
- [ ] If and only if pilot passes, generate full Current-R20 and V2-R20 once at seed 20260713.
- [ ] Open the 41-row held-out negative panel once after both full outputs close.
- [ ] Compute paired task deltas, deterministic paired bootstrap confidence intervals, AUROC, quantiles, failure taxonomy, replay, and secret scan.

### Task 11: Audit, result-to-claim, reports, and branch closure

**Files:**
- Create/update `PILOT_REPORT_V2_RETRY20.md`
- Create/update `FULL_REPORT_V2_RETRY20.md`
- Create/update `EXPERIMENT_AUDIT.md` and `.json`
- Create/update `RESULT_TO_CLAIM_V2_RETRY20.md`

- [ ] Record artifact hashes, Git commit, exact detector invocation with key source redacted, split isolation, output closure, replay, and secret scan.
- [ ] State the single-seed scope and all unsupported claims.
- [ ] Run the complete offline test suite and compileall immediately before final commits.
- [ ] Stage only task files; verify models, datasets, logs, run outputs, `env.sh`, and unrelated user work remain uncommitted.
- [ ] Mark the persistent Goal complete only after success or a fully documented hard-gate negative result.
