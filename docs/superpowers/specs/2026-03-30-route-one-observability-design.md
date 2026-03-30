# Route-One Observability Design

**Date:** 2026-03-30
**Goal:** 在不改变 `FPR=0.05` 检测口径、不先盲调 `gamma/threshold` 的前提下，为 Humaneval 路线一补齐嵌入侧可观测性，解释 `failed_blocks`、`retry`、`cascade/fallback` 与最终 `hits/z_score` 的关系，并为后续最小收益优化提供直接证据。

## Confirmed Context

### What route one means in this project

Route one is **not** a detector-side change. It targets the embedding rescue path:

1. initial block verification fails
2. retry loop attempts to regenerate a watermark-valid block
3. if still failing, cascade rollback may regenerate a larger structure
4. the run either rescues the block or loses a potential hit

Relevant code paths:

- `wfcllm/watermark/generator.py`
- `wfcllm/watermark/retry_loop.py`
- `wfcllm/watermark/cascade.py`

### Why this is the right next focus

From the saved artifacts of `humaneval_20260328_143927`:

- `watermark_rate = 0.2987804878`
- `mean_z_score = 1.3482808448`
- `embed_rate mean = 0.705394`
- current calibrated threshold = `1.7168072022`

Observed structure of the misses:

- 53 samples are effectively detector-limited under the current `block_count + gamma` combination: even all-hit would not cross the threshold
- 24 missed samples are only **one additional hit** away from crossing the threshold
- among those 24 near-miss samples, 20 have `failed_blocks > 0`
- among all missed samples, the subgroup with `failed_blocks > 0` has:
  - mean blocks around `7.8`
  - mean embed rate around `0.57`
  - mean z-score around `0.49`
- this differs from the short-sample failure regime, which is primarily a route-two problem

Therefore the highest-value diagnosis before any policy tuning is:

- where exactly retry is failing
- whether retry explores genuinely new candidate blocks/signatures
- whether failures are driven by `signature`, `margin`, or `no block`
- whether cascade rollback materially rescues blocks in practice

### Important current limitation

The current artifacts preserve:

- final `embedded_blocks`
- final `failed_blocks`
- coarse `retry_diagnostics`
- `fallback_blocks`

But they do **not** preserve a complete per-block lifecycle. In particular:

- `fallback_blocks` is not currently a reliable explanation field
- retries are stored without enough structure to relate them back to final blocks
- cascade activity is not visible in a way that supports sample-by-sample postmortem

This makes parameter tuning premature. The immediate need is **better evidence**, not earlier intervention.

## Recommended Approach

### Recommended: block lifecycle ledger with dual-layer outputs

Implement a new observability layer with two outputs:

1. keep the existing `watermarked` JSONL schema stable for current tooling
2. add a dedicated block-level diagnostic ledger for route-one analysis

This balances backward compatibility with enough detail to answer the real questions.

### Why not the lighter alternatives

#### Alternative A: summary-only counters

Add only aggregate fields such as:

- retry success count
- retry exhausted count
- cascade trigger count
- failure reason counts

This is low risk, but insufficient for the current task because it cannot answer:

- which blocks were rescued
- whether retries changed signatures or only text
- whether the failed blocks that matter are concentrated in near-miss samples

#### Alternative B: experiment-only replay script

Add a standalone replay diagnostic script under `experiment/`.

This helps with case studies, but it is weaker for systematic Humaneval analysis because:

- it encourages anecdotal inspection over full-run statistics
- it requires an extra execution path to regenerate evidence
- it does not automatically tie the evidence to saved production artifacts

### Recommendation rationale

A block lifecycle ledger is the smallest design that can answer all of the following without rerunning the entire logic in a separate analysis path:

- which specific simple blocks failed at the initial verify step
- which retry attempts produced no block vs a block with a bad signature vs a block with insufficient margin
- whether cascade rollback happened and whether it rescued later blocks
- how those block outcomes aggregate at the sample level

## Design

### Design goals

1. Preserve current artifact compatibility for existing evaluation scripts.
2. Make every simple block traceable from initial verify to final outcome.
3. Standardize failure reasons so offline analysis can group them without heuristics.
4. Keep the new observability layer mostly diagnostic-only; avoid changing watermark behavior in this phase.
5. Make it straightforward to compare saved runs sample-by-sample after the change.

### Non-goals

- Do not change detector thresholds or extraction statistics in this design.
- Do not redesign adaptive gamma in this design.
- Do not optimize short-sample statistical power here.
- Do not add broad dashboard infrastructure.

## Output Layout

### Layer 1: backward-compatible watermarked artifact summary

Keep the existing per-sample record in `data/watermarked/*.jsonl`, and add a compact summary section:

- `diagnostics_version`
- `retry_summary`
- `cascade_summary`
- `failure_reason_counts`
- `rescued_blocks`
- `unrescued_blocks`

This layer is for quick joins with existing `details` and `summary` artifacts.

Example shape:

```json
{
  "id": "HumanEval/38",
  "embed_rate": 0.5454,
  "failed_blocks": 5,
  "diagnostics_version": 1,
  "retry_summary": {
    "blocks_with_retry": 7,
    "retry_rescued_blocks": 2,
    "retry_exhausted_blocks": 5,
    "attempts_total": 19,
    "attempts_no_block": 4
  },
  "cascade_summary": {
    "cascade_triggers": 1,
    "cascade_rollbacks": 1,
    "cascade_rescued_blocks": 1
  },
  "failure_reason_counts": {
    "signature_miss": 8,
    "margin_miss": 3,
    "signature_and_margin_miss": 2,
    "no_block_generated": 4
  },
  "rescued_blocks": 3,
  "unrescued_blocks": 4
}
```

### Layer 2: block lifecycle ledger

Write a new JSONL artifact under a diagnostics directory, for example:

- `data/diagnostics/humaneval_<timestamp>_block_ledger.jsonl`

Each row represents one final simple-block lifecycle, not one token and not one retry attempt alone.

Required top-level fields:

- `sample_id`
- `block_ordinal`
- `node_type`
- `parent_node_type`
- `start_line`
- `end_line`
- `block_text_hash`
- `entropy_units`
- `gamma_target`
- `gamma_effective`
- `margin_threshold`
- `initial_verify`
- `retry_attempts`
- `cascade_events`
- `final_outcome`

### Failure reason enum

Every failed verification attempt must map to one normalized reason:

- `signature_miss`
- `margin_miss`
- `signature_and_margin_miss`
- `no_block_generated`
- `cascade_replaced`
- `unknown`

This enum is deliberately small so offline analysis can depend on it.

## Data Model

### Initial verify record

`initial_verify` captures the very first block verification outcome:

- `passed`
- `signature`
- `in_valid_set`
- `min_margin`
- `failure_reason`

Purpose:

- distinguish initial easy wins from blocks that required rescue
- measure how often route one even activates

### Retry attempt records

`retry_attempts` is an ordered array. Each entry contains:

- `attempt_index`
- `produced_block`
- `block_text_hash`
- `signature`
- `in_valid_set`
- `min_margin`
- `failure_reason`

If `_generate_until_block` returns `None`, the record must still exist with:

- `produced_block = false`
- `failure_reason = "no_block_generated"`

Purpose:

- expose whether retry is exploring meaningfully new states
- separate “could not form a block” from “formed a bad block”

### Cascade event records

`cascade_events` is an array because one sample may traverse multiple compound scopes.

Each entry should include:

- `triggered`
- `compound_node_type`
- `failed_simple_count_before_cascade`
- `restored_total_blocks`
- `restored_embedded_blocks`
- `restored_failed_blocks`

Optional but useful:

- `checkpoint_key_hash`
- `retired_after_use`

Purpose:

- answer whether cascade is happening at all
- show what runtime stats were rewound
- help explain why a previously failed block disappeared from the final lineage

### Final outcome record

`final_outcome` should contain:

- `embedded`
- `rescued_by_retry`
- `rescued_by_cascade`
- `exhausted_retries`
- `ended_without_block`

Interpretation rules:

- `rescued_by_retry = true` means the initial verify failed, but a retry attempt succeeded for the same logical block
- `rescued_by_cascade = true` means the original logical block path failed, but after a cascade rollback the final regenerated block lineage produced a successful embedding in that scope
- `ended_without_block = true` means the retry budget ended without producing a replacement block

## Instrumentation Points

### In `wfcllm/watermark/generator.py`

Primary ownership:

- create per-sample diagnostic collector
- assign logical block ordinals
- record initial verify results
- finalize ledger rows and sample summaries

Instrumentation points:

1. When a simple block first appears
   - record its identity fields
   - record `initial_verify`
2. When retry succeeds
   - mark the block as `rescued_by_retry`
3. When retry exhausts
   - mark the block as pending failure
   - attach retry diagnostics
4. When cascade triggers
   - record that the prior logical path was replaced
5. At sample end
   - emit the compact per-sample summary
   - emit the full ledger rows

Important constraint:

The final `embedded_blocks` count must continue to come from final AST verification, as it does today. The new observability layer must explain the run, not redefine the run.

### In `wfcllm/watermark/retry_loop.py`

Primary ownership:

- classify retry outcomes precisely
- expose enough verification detail for downstream diagnostics

Required extensions:

- include `in_valid_set` in attempt diagnostics
- include a normalized `failure_reason`
- include `block_text_hash` for produced blocks
- preserve whether the attempt produced no block

This is the most important source of evidence for route one.

### In `wfcllm/watermark/cascade.py`

Primary ownership:

- expose structured cascade metadata instead of only stack state

Required extensions:

- return a serializable cascade event summary
- preserve the compound node type that triggered rollback
- surface the number of failed simple blocks in that scope

The current `CascadeCheckpoint` already contains most of the needed context. The main gap is making that context visible in saved artifacts.

## Offline Analysis Integration

### Extend offline helpers, do not replace them

`wfcllm/extract/offline_analysis.py` should remain the main offline comparison surface. Extend it to optionally consume:

- the new per-sample summary fields from `watermarked`
- the block ledger artifact

Useful new report capabilities:

- compare `failure_reason_counts` across two runs
- identify samples where `retry_exhausted_blocks` rose but total blocks stayed stable
- identify samples where cascade triggered but yielded no rescued blocks
- rank samples by “rescuable loss”:
  - missed detection
  - at least one exhausted retry
  - within one or two hits of threshold

### Suggested new anomaly flags

Offline analysis should be able to flag cases such as:

- `retry_diversity_low`
- `cascade_no_recovery`
- `margin_failures_dominate`
- `signature_failures_dominate`
- `near_miss_with_exhausted_retry`

These should be additive; do not remove the current anomaly flags.

## Error Handling

### Missing diagnostics must degrade safely

If a block cannot produce complete diagnostics due to an unexpected runtime condition:

- do not fail the watermark run
- emit partial diagnostics with `failure_reason = "unknown"`
- increment a compact sample-level `diagnostic_errors` counter

### Artifact compatibility

The existing loading code must tolerate older watermarked artifacts that do not include diagnostics.

This means:

- all new fields must be optional on read
- new offline analysis features must degrade gracefully when diagnostics are absent

### Size control

The detailed ledger may become large. To keep this manageable:

- store only text hashes by default, not full block text
- keep signatures and scalar verification details, not raw embeddings
- preserve ordered arrays, but avoid repeated large strings

If needed later, a verbose mode can add raw block text behind an explicit flag. That is out of scope for this design.

## Testing

### Unit tests

Add or update tests in:

- `tests/watermark/test_retry_loop.py`
- `tests/watermark/test_generator.py`
- `tests/watermark/test_cascade.py`
- `tests/extract/test_offline_analysis.py`

Minimum unit coverage:

1. retry attempt records classify `signature_miss`, `margin_miss`, and `no_block_generated` correctly
2. generator emits compact per-sample summaries without changing current required fields
3. cascade metadata is preserved and serialized when rollback occurs
4. offline analysis can load artifacts with and without diagnostics

### Integration tests

Add a focused integration test that simulates:

- initial verify failure
- retry exhaustion on one block
- cascade trigger
- later successful regenerated block

The assertion should check both:

- final embedding counters
- saved ledger semantics

### Regression tests

Add a regression test for the current known observability gap:

- `fallback_blocks` and cascade-related rescue information must not silently remain zero or invisible when cascade occurs

This test is about explanatory correctness, not watermark quality.

## Validation Plan

After implementation, the first validation target is **diagnostic correctness**, not improved watermark rate.

Required validation questions:

1. For a saved run, can we explain every `failed_block` in terms of standardized reasons?
2. For every near-miss sample, can we tell whether the missing hit was due to:
   - no retry success
   - no block produced during retry
   - signature mismatch
   - margin mismatch
   - no cascade recovery
3. For a cascade-triggered sample, can we see whether the rollback actually rescued later blocks?

Only after these questions are answerable should implementation move on to route-one optimization policy changes.

## Risks and Trade-offs

### Main trade-off

This design adds schema and bookkeeping complexity before it adds direct watermark gains.

That is intentional. The current failure mode is not under-instrumented by accident; it is under-instrumented in exactly the place where future policy tuning would otherwise become guesswork.

### Risks

- over-coupling diagnostics to current control flow
- accidentally changing runtime behavior while adding instrumentation
- producing artifacts too large for convenient offline comparison

### Mitigations

- keep diagnostics observational first
- keep legacy fields and meanings stable
- prefer hashes and scalar summaries over raw text blobs

## Decision

Proceed with the block lifecycle ledger design:

1. add compact per-sample route-one summaries to the watermarked artifact
2. add a dedicated block-ledger JSONL artifact
3. standardize retry and cascade failure reasons
4. extend offline analysis to consume the new evidence

Do not start policy tuning until this observability layer is in place and validated.

## Process Gate

This document is design-only. No implementation work should begin from this spec alone.

Required next step after user review:

- invoke the `writing-plans` workflow and produce an implementation plan for the observability changes above
