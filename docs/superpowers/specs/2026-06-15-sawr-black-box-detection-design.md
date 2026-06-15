# SAWR Black-Box Detection Design

Date: 2026-06-15

Status: design approved for documentation. Do not implement until the written
spec has been reviewed and explicitly approved.

## Summary

The SAWR detector will be a keyed black-box detector over final code only. It
must not use generation-time candidates, windows, layers, checkpoints, retry
state, rollback state, audit rows, logits, sampling traces, or the generation
model.

The selected design is a structure-aware proxy-window detector:

1. Parse the final code.
2. Enumerate a fixed family of proxy windows inside structure contexts.
3. Score each proxy window with CodeT5 + keyed LSH.
4. Calibrate window and structure-context evidence on same-model
   unwatermarked final code.
5. Aggregate calibrated context evidence into a single-sample score.
6. Use an empirical 5% FPR threshold and empirical p-value.

The detector does not recover generation-time evidence. It tests whether a
fixed final-code proxy statistic is unusually large under a rigorously
calibrated non-watermarked null distribution.

## Context From Current SAWR Embedding

The implemented SAWR generation path lives under `wfcllm/sawr/` and
`scripts/run_sawr_smoke.py`.

Relevant implementation facts:

- `PromptAwareBoundaryDetector` emits generation-time events:
  `compound_started`, `simple_candidate`, `layer_closed`, and `final_flush`.
- The current structure-aware generator supports controlled Python function
  bodies plus nested `if_statement`, `for_statement`, and `while_statement`
  layers.
- The actual semantic LSH predicate is evaluated on candidate windows, not on
  whole layers.
- A candidate window is normalized by stripping surrounding whitespace,
  right-stripping lines, and joining candidate texts with newlines.
- `SemanticLshEmbeddingRule` derives the valid LSH set from
  `parent_node_type`, `k`, and optional `ordinal`.
- `gamma` is quantized as:

```text
universe_size = 2 ** lsh_d
k = round(gamma * universe_size), clipped to [1, universe_size - 1]
gamma_effective = k / universe_size
```

- For the strongest current experiment, `lsh_d=4` and `gamma=0.75`, so
  `k=12` out of 16 signatures. This gives high generation-time acceptance
  capacity but also high nominal null hit probability.
- When a generation-time window hits, the state machine marks the current
  layer as having direct evidence and clears that layer's current window.
- When a layer closes with no direct or child hit, the generator may rollback
  and retry if budget allows; otherwise it commits fallback.
- Rollback restores model context, boundary detector state, and state-machine
  state. These states are not visible to a final-code detector.
- Final SAWR rows are detector-clean. They contain `final_code`, but not
  `watermark_params`, `blocks`, `p_value`, `z_score`, `detector_score`,
  generation ledgers, retry ledgers, or audit evidence.
- Audit rows are explicitly debug-only and are marked
  `detector_input_allowed=false`.

These facts imply that the detector cannot replay the generation-time state
machine or reconstruct accepted windows. It can only define a fixed,
calibrated proxy statistic over the final code.

## Detection Boundary

Allowed detector inputs:

- final code;
- watermark secret key;
- LSH dimension `lsh_d`;
- target `gamma` and quantized/effective `k`;
- semantic margin threshold;
- encoder model path and checkpoint identity;
- LoRA / bf16 / whitening configuration;
- `use_ordinal_keying` configuration;
- public parser, normalization, and proxy-window rules.

Forbidden detector inputs:

- generation model;
- logits;
- sampling trace;
- generation-time checkpoint;
- generation-time candidate, window, or layer identity;
- generation-time accept/reject decisions;
- rollback or retry trace;
- audit JSONL rows;
- any hidden metadata not present in detector-clean final rows.

The detector may parse final code and construct final-code proxy units. Those
proxy units are not generation-time units and must not be described as such.

## Scientific Claim

The detector tests a final-code proxy statistic induced by SAWR's
generation-time selection pressure.

It does not claim to recover embedded layers or generation-time evidence.

Accepted paper wording:

```text
We define a fixed family of final-code proxy windows and evaluate keyed LSH
evidence over them. The proxy windows are not assumed to match generation-time
windows. Their distribution is calibrated entirely on same-model
non-watermarked code.
```

Rejected paper wording:

```text
The detector recovers generation-time evidence.
The detector replays the SAWR state machine.
The detector reconstructs accepted SAWR windows.
The nominal gamma alone gives the null distribution.
Audit hits prove detectability.
High embedding rate implies high black-box detectability.
```

## Hypothesis Test

Primary goal:

```text
Single-sample detection, evaluated by TPR@5%FPR.
```

Null hypothesis:

```text
H0: The code is final code from the same task distribution, same base model,
same decoding setup, and same final-code post-processing, but without SAWR.
```

Alternative hypothesis:

```text
H1: The code is final code from SAWR generation. SAWR's generation-time
accept/retry/fallback selection pressure shifts the final-code proxy evidence
score upward.
```

Test statistic:

```text
T(code) = mean(context_score over scoreable structure contexts)
```

Each `context_score` is a calibrated structure-context score derived from the
strongest proxy-window evidence in that context. The "strongest" operation is
not a raw maximum; it is calibrated against the same operation on negative
structure contexts.

Threshold:

```text
threshold_5fpr = 95th percentile of T(code) over calibration negatives
```

Empirical p-value:

```text
p_value = (1 + number of calibration negative scores >= observed score)
          / (1 + number of calibration negatives)
```

Decision:

```text
is_watermarked = T(code) >= threshold_5fpr
```

## Proxy-Window Extraction

The detector parses final code with the project Python parser.

Scoreable structure contexts:

- controlled function body or final-code target function body;
- `if_statement`;
- `for_statement`;
- `while_statement`.

The detector should keep the initial policy aligned with current SAWR
generation support. Future extensions to `try`, `with`, `match`, helper
functions, classes, or other languages require separate calibration and
evaluation.

Within each scoreable structure context:

- collect direct simple statements only;
- do not mix child-layer statements into parent direct windows;
- do not let windows cross structure-context boundaries;
- ignore comments, whitespace-only fragments, malformed parse fragments, and
  parser-recovery statements;
- apply the same normalization used by SAWR semantic LSH.

Proxy window family:

```text
All contiguous direct-statement windows with length in
[1, max_group_statements].
```

Example for direct statements `A`, `B`, `C` and `max_group_statements=2`:

```text
[A]
[B]
[C]
[A, B]
[B, C]
```

These are fixed detector-side proxy windows. They are not assumed to be
generation-time windows.

## Window Evidence

Each proxy window is scored with the same semantic LSH machinery used for
embedding:

```text
normalized window text
  -> CodeT5 / semantic encoder
  -> embedding
  -> keyed LSH signature
  -> valid-set membership
  -> margin
```

Raw fields to retain for diagnostics:

- `in_valid_set`;
- `min_margin`;
- `passed_margin`;
- `lsh_signature`;
- `parent_node_type`;
- `window_length`;
- `structure_type`;
- `context_window_count`;
- `context_statement_count`.

Raw window evidence for calibration:

```text
window_raw = min_margin if in_valid_set and min_margin > margin_threshold
window_raw = 0 otherwise
```

Because `gamma=0.75` gives high nominal null hit probability, raw binary hits
are insufficient as the primary evidence. Margin and empirical calibration are
required.

## Structure-Context Calibration

The detector uses structure contexts to calibrate and aggregate evidence. A
structure context is not treated as the true generation-time layer.

For each scoreable context:

```text
context_raw = max(window_raw over proxy windows in the context)
```

The raw maximum is not used directly for final decisions. It is converted into
a calibrated context p-value:

```text
context_p = empirical probability that a same-condition negative context has
            context_raw >= observed context_raw
```

Conditioning variables for negative context calibration:

- structure type: function body, `if_statement`, `for_statement`,
  `while_statement`;
- parent node type;
- proxy window length mix;
- number of proxy windows in the context, bucketed;
- number of direct statements in the context, bucketed;
- sample length or total proxy-window count, bucketed;
- `use_ordinal_keying` mode.

The exact bucket edges are chosen on a development split and then frozen before
calibration and testing.

Context score:

```text
context_score = -log10(max(context_p, p_floor))
```

`p_floor` is determined by calibration corpus size, for example
`1 / (N_context_negative + 1)`, to avoid infinite scores.

Why max within context:

- SAWR's selection pressure is local: a structure context can succeed because
  at least one local window provides keyed evidence.
- Averaging all windows in a context can dilute a sparse local signal.
- The multiple-comparison effect of taking a maximum is absorbed by empirical
  context calibration, because negatives use the identical max operation.

Why not raw max:

- raw max grows with the number of proxy windows;
- raw max can be driven by structure complexity rather than watermark signal;
- raw max is not a p-value and cannot by itself support a 5% FPR claim.

## Sample Score

The sample score is the mean calibrated context score:

```text
T(code) = mean(context_score_1, ..., context_score_m)
```

where `m` is the number of scoreable contexts with at least one proxy window.

Rationale:

- a sum would make longer or more structured code naturally score higher;
- a sample-level max would allow a single anomalous context to dominate and
  increase FPR risk;
- the mean keeps the statistic closer to a per-context abnormality measure.

Samples with too little evidence should be marked:

```text
insufficient_evidence = true
```

Minimum evidence requirements are set on the development split and frozen
before evaluation. Candidate defaults:

```text
min_scoreable_contexts = 1
min_proxy_windows = 2
```

Insufficient-evidence samples must be reported separately. They should not be
silently dropped from headline metrics.

## Null Calibration

Primary calibration corpus:

```text
Same model, same task distribution, same decoding setup, same final-code
post-processing, no SAWR.
```

This is the official null for the 5% FPR threshold.

Reference-solution negatives:

```text
Dataset reference solutions are used as stress tests, not as the primary
threshold source.
```

Matched negative subsets:

```text
Length-, statement-count-, structure-count-, and pass-rate-matched subsets are
used to diagnose confounding.
```

The detector must not use positive samples to choose the 5% FPR threshold.

## Data Splits

Strict data separation is required:

- development split: choose proxy-window policy, calibration buckets, and
  statistic form;
- calibration split: use negative samples only to estimate context
  calibration and sample-level threshold;
- test split: evaluate final frozen detector on positive and negative samples.

For HumanEval-scale datasets, splits should be task-based, not row-based. If a
task has multiple generated samples, all samples from that task must remain in
the same split.

Recommended negative size for stable 5% FPR calibration:

```text
N_negative >= 1000 when feasible
```

If deterministic generation provides only one negative per task, report wide
confidence intervals and treat the result as preliminary. Strong single-sample
claims need enough held-out negative samples to estimate the 95th percentile
stably.

## Evaluation Metrics

Primary metric:

```text
single-sample TPR@5%FPR
```

Additional required metrics:

- observed FPR on held-out same-model negatives;
- AUROC;
- empirical p-value distribution;
- bootstrap 95% confidence intervals;
- score histograms or quantile summaries for positive and negative samples;
- FPR by length bucket;
- FPR by statement-count bucket;
- FPR by structure-count bucket;
- pass rate for positive and negative corpora;
- correlation between score and length, statement count, context count, and
  proxy-window count.

The detector is scientifically acceptable even if the current embedding
configuration yields weak TPR, provided the failure is reported as evidence
that high generation-time embedding rate does not necessarily imply strong
single-sample black-box detectability.

## Baselines And Ablations

Keep the detector statistical. Do not add an auxiliary learned detector in this
design.

Required baselines:

1. Simple proxy-window hit-count baseline.
2. Non-structure-aware proxy-window statistic that ignores structure type.
3. Structure-aware calibrated proxy-window detector.

Required ablations:

- raw context max vs calibrated context max;
- context mean window evidence vs calibrated context max;
- hit-only vs margin-only vs hit-plus-margin evidence;
- all contiguous windows vs shorter window families;
- different `max_group_statements`;
- different `gamma` and `lsh_d`;
- with and without whitening;
- with and without ordinal keying, if embedding used ordinal keying.

These ablations determine whether structure awareness adds detection power.
The design must not assume structure contexts are optimal.

## Outputs

Per-sample detector output:

```json
{
  "id": "...",
  "is_watermarked": true,
  "score": 1.23,
  "threshold_5fpr": 0.98,
  "p_value": 0.031,
  "fpr_target": 0.05,
  "scoreable_contexts": 4,
  "proxy_windows": 11,
  "insufficient_evidence": false,
  "detector_mode": "sawr-structure-aware-proxy-window/v1"
}
```

Diagnostics may include context and window summaries, but no generation-time
trace fields.

Forbidden detector output fields:

- generation-time checkpoint;
- generation-time candidate identity;
- generation-time window identity;
- generation-time layer identity;
- audit event references;
- retry or rollback traces;
- claims that proxy windows are recovered embedding windows.

## Compatibility With Repo Boundaries

The design should land as a future detector path that is compatible with the
isolated `wfcllm/sawr` package and existing extraction/calibration concepts.

High-level boundaries:

- reuse CodeT5 semantic encoder loading conventions from SAWR semantic LSH;
- reuse `WatermarkKeying` and `LSHSpace`;
- keep final SAWR rows detector-clean;
- keep audit rows debug-only;
- keep negative calibration separate from detection-time inputs;
- do not import legacy `experiment.*` from production modules.

The exact implementation module layout belongs in the later implementation
plan, after this spec is reviewed.

## Go / No-Go Criteria

The detector can support a paper detection claim only if all are true:

- held-out same-model observed FPR is at or below the target within confidence
  interval expectations;
- TPR@5%FPR is meaningfully above the simple hit-count baseline;
- score is not primarily explained by length, statement count, context count,
  proxy-window count, or pass/fail status;
- reference-solution stress tests do not show unacceptable false positive
  behavior;
- no audit or generation trace is used by calibration or detection;
- all thresholds are frozen before held-out testing.

If these conditions fail, the correct paper claim is a diagnostic or negative
result: current SAWR settings may provide high generation-time embedding
coverage without reliable single-sample black-box detectability.

## Open Design Decisions For Implementation Planning

These are not blockers for the detection concept, but must be fixed before
implementation:

- exact target-function selection rule when final code contains more than one
  function;
- exact calibration bucket edges;
- exact minimum evidence thresholds;
- exact JSON schema for detailed context/window diagnostics;
- exact split protocol for multiple generations per task;
- whether the first implementation is under `wfcllm/sawr/detect` or a new
  extraction-adjacent module.

These decisions should be made in the implementation planning phase, not in
this brainstorming design turn.
