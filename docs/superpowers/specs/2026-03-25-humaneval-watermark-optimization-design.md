# Humaneval Watermark Optimization Design

**Date:** 2026-03-25
**Goal:** 在保持 `FPR=0.05` 统计口径成立的前提下，将 Humaneval 的 `watermark_rate` 提升到 `45%+`，并解释为什么结果从历史最佳 `0.3902` 下降到当前 `0.2988`。

## Confirmed Context

### Current and historical results

- Historical best: `data/results/humaneval_20260323_150658_summary.json`
  - `watermark_rate = 0.3902439024`
  - `mean_z_score = 1.2787921822`
  - `mean_blocks = 8.5487804878`
  - `embed_rate mean = 0.777694`
- Current baseline: `data/results/humaneval_20260324_184914_summary.json`
  - `watermark_rate = 0.2987804878`
  - `mean_z_score = 1.0981987049`
  - `mean_blocks = 8.3292682927`
  - `embed_rate mean = 0.77035`

### User constraints

- Primary target is **Humaneval-specific optimization**, not immediate cross-dataset generalization.
- `FPR=0.05` is a hard constraint.
- The user prefers **using saved historical artifacts first** and avoiding reruns unless they are clearly necessary.
- The preferred strategy is **A+B 串联**:
  1. audit program/statistical correctness
  2. then do Humaneval-specific recalibration and parameter optimization

### Important discrepancy found from saved artifacts

The saved historical watermarked artifact records the actual best-run watermark parameters in `data/watermarked/humaneval_20260323_150658.jsonl`, and they are not exactly the same as the user’s recalled values.

Observed historical best-run parameters:

- `lsh_d = 4`
- adaptive anchors:
  - `p10 = 0.75`
  - `p50 = 0.75`
  - `p75 = 0.50`
  - `p90 = 0.50`
  - `p95 = 0.25`

Saved artifacts should therefore be treated as the source of truth for:

- what parameters were actually used in a run
- what outputs and counters were actually recorded

But recorded derived statistics are still audit targets. If accounting logic is buggy, saved counters and summaries may faithfully preserve buggy behavior, so they cannot be treated as unquestionable algorithmic truth.

## Process Gate / Execution Order

This document is a design-only artifact. No implementation work should begin from this spec alone.

Required workflow for this task:

1. brainstorming completes and this spec is written
2. the user reviews and approves this spec
3. `using-git-worktrees` sets up an isolated worktree and verifies a clean baseline
4. `writing-plans` writes an implementation plan in the worktree
5. only then do implementation, tests, and any validation reruns begin

This gate is mandatory even if the likely outcome is “only diagnostics” or “only parameter tuning.”

## Problem Framing

The current evidence suggests the drop is more likely caused by **detection power loss** than by a major collapse in embedding coverage:

- `embed_rate mean` changed only slightly (`0.777694 -> 0.77035`)
- but `mean_z_score` dropped more noticeably (`1.2788 -> 1.0982`)
- and `watermark_rate` dropped from `0.3902` to `0.2988`

Therefore the highest-priority questions are:

1. Did the embedding side change the effective hit difficulty through `lsh_d`, adaptive gamma, or effective-region rules?
2. Did the extraction side become more conservative through `hits`, `independent_blocks`, or `z` computation?
3. Are there latent accounting mismatches that pass contract/alignment checks but still depress `z`?

## Recommended Approach

### Recommended: offline diagnosis first, then focused code audit, then minimal validation reruns

Proceed in four phases.

### Phase A — Offline artifact diagnosis

Use the already-saved artifacts first:

- `data/results/humaneval_20260323_150658_details.jsonl`
- `data/results/humaneval_20260324_184914_details.jsonl`
- `data/watermarked/humaneval_20260323_150658.jsonl`
- `data/watermarked/humaneval_20260324_184914.jsonl`

The goal of Phase A is to determine whether the regression is primarily caused by:

- parameter drift
- adaptive-gamma / effective-region behavior
- extraction-statistic conservatism
- calibration/threshold drift
- or a real implementation bug

No rerun should happen before Phase A conclusions are in hand, unless a schema mismatch prevents offline comparison.

### Phase B — Focused code/statistics audit

Only after Phase A identifies the most suspicious surfaces:

- audit embedding-side adaptive gamma and effective-region logic
- audit extraction-side `hits`, `independent_blocks`, and `z` accounting
- audit embed/extract contract alignment boundaries
- audit calibration and decision-threshold derivation

The goal of Phase B is to confirm whether the regression is explained by code behavior, statistical definition changes, or calibration drift.

### Phase C — Humaneval-specific optimization and recalibration

Only after Phases A and B:

- restore or refine the historically stronger parameter region
- improve adaptive gamma / effective-region logic if justified
- adjust calibration inputs or thresholding only under an explicitly declared calibration regime
- keep the optimization Humaneval-specific unless later evidence justifies broader generalization

### Phase D — Minimal validation reruns

Only after the hypothesis, code path, and calibration regime are explicit:

- run the smallest experiment that can validate the chosen hypothesis
- run a full Humaneval rerun only if the targeted validation succeeds or if a full rerun is needed to measure the final objective

## Scope

### In scope

1. Compare historical and current saved runs sample-by-sample.
2. Audit watermark embedding statistics recorded in saved watermarked files.
3. Audit extraction-side counting and `z` calculation logic.
4. Add or strengthen tests where logic appears ambiguous or incorrect.
5. Make targeted code changes if the offline analysis identifies real bugs or unjustified conservatism.
6. Produce a minimal experiment plan to validate fixes and parameter changes.

### Out of scope

- Broad refactors unrelated to watermarking behavior
- Immediate cross-dataset optimization
- Large blind parameter sweeps before offline diagnosis
- Relaxing the `FPR=0.05` requirement without explicit recalibration logic and evidence

## Phase A: Offline Diagnosis Design

### A0. Artifact compatibility precheck

Before any sample-by-sample comparison, verify that the two saved runs are comparable.

Required prechecks:

- same dataset and expected sample cardinality
- comparable `id` coverage and one-to-one sample matching
- same detector mode semantics for `mode`, `is_watermarked`, `hits`, `independent_blocks`, and `z_score`
- same or explicitly understood summary/detail schema meaning

If schemas or statistic semantics differ, normalize the runs before comparing deltas. If normalization is impossible from saved artifacts alone, mark that comparison as unresolved and escalate to targeted diagnostics instead of forcing a misleading direct comparison.

### A1. Result-side sample comparison

Compare the two saved detail files by `id` and compute per-sample deltas for:

- `is_watermarked`
- `z_score`
- `p_value`
- `independent_blocks`
- `hits`

Questions this must answer:

- Which samples flipped from detected to undetected?
- Are losses concentrated in a small subset or spread broadly?
- For flipped samples, is the main driver lower `hits`, higher `independent_blocks`, or lower `z` at similar hit/block ratios?

### A2. Embedding-side sample comparison

Compare the two saved watermarked files by `id` and inspect:

- `total_blocks`
- `embedded_blocks`
- `embed_rate`
- `failed_blocks`
- `fallback_blocks`
- `alignment_summary`
- per-block `entropy_units`
- per-block `gamma_target`
- per-block `gamma_effective`
- recorded `watermark_params`

If some fields exist in one artifact but not the other, compare all shared fields first, then either infer missing context from metadata/code path or explicitly mark the hypothesis as unresolved. Missing fields should narrow the conclusion, not force unsupported assumptions.

Questions this must answer:

- Is the regression due to fewer embedded blocks?
- Did adaptive anchors shift many blocks to harder effective gamma values?
- Did `lsh_d` alter effective hit probability enough to explain the `z` drop?
- Are there more failed or marginally embedded blocks even when aggregate `embed_rate` looks stable?

### A3. Ground-truth parameter reconstruction

Reconstruct the actual run parameters from saved artifacts rather than from current configs or memory.

This includes:

- historical vs current adaptive anchors
- `lsh_d`
- `lsh_gamma`
- `margin_base`
- `margin_alpha`
- calibration/profile identifiers
- profile quantiles embedded in the run artifact

Purpose:

- prevent false conclusions caused by config drift or mistaken recollection
- separate “config changed” from “code behavior changed”

### A4. Statistical anomaly screening

Use the saved detail files to flag suspicious cases such as:

- high `hits` but unexpectedly low `z_score`
- unusually large `independent_blocks`
- alignment reported as valid but with unexpectedly weak detection
- systematic low `z` for certain block-count bands

Purpose:

- infer likely bug locations before touching code
- narrow the later code audit to the smallest plausible surface area

## Phase B: Code Audit Design

The code audit should stay focused on logic that can directly change `watermark_rate`.

### B1. Watermark embedding path

Audit the implementation of:

- adaptive gamma quantile mapping
- `gamma_target -> k -> gamma_effective` discretization
- effective-region / block eligibility rules
- counters for `embedded_blocks`, `failed_blocks`, and `fallback_blocks`

Primary questions:

- Are anchors interpreted exactly as intended?
- Is quantization of gamma unintentionally too conservative?
- Did `lsh_d` changes alter the hit landscape more than expected?
- Can embed counts look healthy while effective hit quality degrades?

### B2. Extraction path

Audit the implementation of:

- block rebuilding and independent-block counting
- hit definition and accumulation
- `z`-score numerator and denominator
- expected-value and variance terms under adaptive gamma
- adaptive detection flow in `prefer-adaptive` mode

Primary questions:

- Is `z` systematically more conservative than theory requires?
- Is `independent_blocks` inflated, depressing `z`?
- Does adaptive detection apply any extra penalty even after contract checks pass?

### B3. Embed/extract accounting boundary

Audit the boundary between generator-side recorded blocks and extractor-side rebuilt blocks.

Primary questions:

- When `alignment_ok` and `contract_valid` are true, are the statistical inputs truly identical?
- Can block ordering, grouping, or effective-region interpretation differ while the surface contract still appears valid?
- Are numeric mismatch warnings only cosmetic, or can they leak into detection statistics?

### B4. Calibration and threshold layer

Audit:

- negative-corpus calibration source and distribution assumptions
- threshold derivation for `fpr=0.05`
- the final `is_watermarked` decision rule used in extraction summaries

Primary questions:

- Is the `0.05` FPR claim still justified for the current Humaneval setup?
- Are historical and current runs using effectively comparable decision thresholds?
- If recalibration is needed, can it be done without weakening the interpretation of `FPR=0.05`?

## FPR Preservation Rule

`FPR=0.05` is treated as an operational acceptance criterion, not a vague intent.

Any optimization or code change that affects detection statistics must satisfy all of the following before the result is described as preserving `FPR=0.05`:

1. the threshold is derived from a declared negative-corpus source and declared statistic definition
2. the extraction run uses the same statistic definition that was used to calibrate that threshold
3. the measured false-positive rate on the declared negative set is `<= 0.05`, subject to the finite-sample confidence interval being reported alongside the estimate
4. if the negative source, statistic definition, or threshold derivation changes, the run is labeled as a new calibration regime rather than silently treated as equivalent to the old one

This rule allows recalibration, but it forbids claiming continuity of the old `FPR=0.05` regime without evidence.

## Decision Rules

### If offline evidence points mainly to parameter differences

Then:

- do not change code first
- restore or refine the stronger historical parameter region
- validate with a minimal targeted experiment

### If offline evidence points to statistical/accounting anomalies

Then:

- fix code first
- add tests that fail before the fix and pass after it
- only then run minimal validation experiments

### If offline evidence points mainly to calibration or threshold drift

Then:

- freeze code unless a separate code bug is independently supported
- restore or rederive calibration under an explicitly named regime
- validate the resulting threshold on the declared negative set before comparing watermark-rate outcomes

### If offline evidence is suggestive but inconclusive

Then:

- add targeted diagnostics or tests
- run the smallest possible reproduction needed to resolve ambiguity
- avoid full Humaneval reruns until the ambiguity is removed

## Verification Strategy

Verification proceeds in this order:

1. offline comparison of saved artifacts
2. focused unit/integration tests for suspicious logic
3. targeted bug fix or parameter adjustment
4. minimal validation rerun if needed
5. full Humaneval rerun only after the hypothesis is strong enough

## Rerun Stopping Rules

The default is no rerun.

- No rerun if the regression is fully explained by saved-artifact parameter drift and no code/statistics ambiguity remains.
- One targeted rerun is allowed only when a specific hypothesis cannot be resolved offline or when a fix/parameter change needs direct validation.
- A full Humaneval rerun is allowed only after:
  - the hypothesis is predeclared,
  - the calibration regime is explicit,
  - focused tests pass,
  - and the targeted validation does not contradict the hypothesis.

These rules exist to preserve the user preference for saved-artifact-first analysis and to avoid ad hoc experiment expansion.

## Success Criteria

A successful outcome for this design is:

1. explain the regression from `0.3902` to `0.2988` using saved evidence
2. identify whether the dominant cause is embedding-side, extraction-side, or both
3. preserve a justified `FPR=0.05` interpretation
4. define the smallest code and experiment changes needed to push Humaneval `watermark_rate` above `0.45`

## Staged Success Criteria

### Stage 1 — Diagnosis complete

- the regression from `0.3902` to `0.2988` is explained from saved artifacts with clearly stated confidence and limits
- dominant cause is classified as embedding-side, extraction-side, calibration-side, or mixed

### Stage 2 — Change hypothesis ready

- the proposed fix or parameter change is tied to explicit evidence
- the required calibration regime is explicit
- necessary tests are identified or added

### Stage 3 — Minimal validation complete

- the smallest needed rerun has either confirmed the hypothesis or disproved it
- any FPR-sensitive change has been validated under the declared rule above

### Stage 4 — Objective validated

- a full Humaneval validation run reaches `watermark_rate >= 0.45` while preserving the declared `FPR=0.05` regime

## Expected Deliverables

1. An offline diagnosis report based on saved `details` and `watermarked` artifacts.
2. A short list of confirmed bugs or suspicious statistical behaviors.
3. Targeted tests for any corrected logic.
4. A Humaneval-specific optimization recommendation grounded in actual run artifacts.
5. A minimal validation experiment plan, used only where saved data is insufficient.
