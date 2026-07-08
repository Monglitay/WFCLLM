# SAWR Structure-Aware Generation Embedding Report

Date: 2026-06-15

Commit: `26e71b5 feat: add structure-aware sawr generation`

Remote workspace used for experiments:

```text
/root/autodl-tmp/WFCLLM_sawr_smoke_codex_20260612
```

Primary purpose: summarize the current SAWR generation-time embedding scheme,
implementation details, and experiment results as factual reference material.

This document describes the isolated SAWR smoke path under `wfcllm/sawr/` and
`scripts/run_sawr_smoke.py`. It does not describe the production
`wfcllm/watermark/` or `wfcllm/extract/` pipeline.

## Executive Summary

The implemented SAWR generation-time embedding path is now structure-aware for
Python function-body code. Instead of evaluating only flat statement candidates,
the generator observes parser-derived boundary events, tracks nested compound
layers, and performs bounded rollback when a layer closes without semantic LSH
evidence.

The best full HumanEval configuration tested so far is:

```text
rule_name: semantic_lsh
lsh_d: 4
lsh_gamma: 0.75
encoder: CodeT5-base + LoRA checkpoint encoder_epoch10.pt
generator: DeepSeek Coder 7B Instruct
dataset: HumanEval, 164 tasks
```

Best observed full-run metrics:

```text
sample-level embedding rate: 142 / 164 = 86.59%
layer-level embedding rate: 365 / (365 + 7) = 98.12%
window-vs-fallback rate: 439 / (439 + 7) = 98.43%
retry requests: 7
fallback layers: 7
HumanEval Pass@1: 113 / 164 = 68.90%
```

## Current Embedding Scheme

### High-Level Protocol

SAWR generation-time embedding runs while a causal LM is decoding a Python
function completion. The generator incrementally feeds decoded text into a
prompt-aware Python boundary detector. The detector emits semantic units as
events:

```text
compound_started
simple_candidate
layer_closed
final_flush
```

The state machine consumes those events and maintains a stack of active layers.
Simple candidates are grouped into a per-layer sliding window. When the
embedding rule accepts a candidate window, the layer records direct semantic
evidence. When a compound layer closes, evidence can be direct or inherited from
children. If a layer closes without evidence and rollback is still allowed, the
generator rolls back to the checkpoint at the start of that layer and samples
again. If retry/budget controls are exhausted, the layer is committed as
fallback without a hit.

The final JSONL rows are intentionally detector-clean. Audit rows contain
debugging and experimental events but are not detector inputs and are not
scientific evidence by themselves.

### Semantic LSH Rule

The currently preferred real embedding rule is `SemanticLshEmbeddingRule`.

For each candidate window:

1. Normalize the candidate texts by stripping surrounding whitespace and
   right-stripping each line.
2. Join the candidate group with newlines.
3. Encode the joined text with `SemanticEncoder`, backed by CodeT5-base and the
   LoRA checkpoint.
4. Project/sign the embedding in the keyed LSH space.
5. Derive the valid signature set from `WatermarkKeying`.
6. Accept the candidate window if the signature is in the valid set and the
   signed-space margin exceeds the configured margin.

The rule records diagnostics in `RuleDecision.reason`:

```text
lsh_signature
in_valid_set
min_margin
margin
parent_node_type
ordinal
k
gamma_target
gamma_effective
```

### Gamma Quantization

The current semantic rule quantizes `lsh_gamma` into an integer valid-set size
`k`:

```text
universe_size = 2 ** lsh_d
k = round(lsh_gamma * universe_size), clipped to [1, universe_size - 1]
gamma_effective = k / universe_size
```

Examples from the runs:

```text
lsh_d=4, lsh_gamma=0.75 -> universe=16, k=12, gamma_effective=0.75
lsh_d=4, lsh_gamma=0.50 -> universe=16, k=8, gamma_effective=0.50
lsh_d=8, lsh_gamma=0.25 -> universe=256, k=64, gamma_effective=0.25
```

The experiments show that lowering `gamma` and increasing `d` can drastically
reduce hit probability, increase rollback/fallback pressure, and lower observed
embedding coverage.

## Implementation Details

### Main Files

```text
wfcllm/sawr/boundary.py
wfcllm/sawr/state_machine.py
wfcllm/sawr/generator.py
wfcllm/sawr/pipeline.py
wfcllm/sawr/rules.py
wfcllm/sawr/semantic_lsh.py
wfcllm/sawr/config.py
scripts/run_sawr_smoke.py
```

### Boundary Detector

The boundary detector is prompt-aware. It avoids treating prompt-owned code as
generated evidence and focuses on the generated function body. It emits
`BoundaryEvent` records rather than only flat candidates.

Important event fields:

```text
kind
node_type
parent_node_type
position_id
layer_path
token_start_idx
token_count
text
start_byte
end_byte
depth
checkpoint_key
candidate
closed_layer_paths
final_flush
```

Important candidate fields:

```text
text
candidate_type
node_type
position_id
token_start_idx
token_count
parent_node_type
ordinal
layer_path
start_byte
end_byte
depth
```

The detector supports nested `if`, `for`, and `while` layers. For
`if`/`elif`/`else`, the whole conditional is treated as one compound unit for
evidence aggregation. Invalid parse fragments and parser-recovery multi-line
simple candidates are rejected until a stable statement boundary is observed.

### Layer State Machine

`SawrStateMachine` now maintains a stack of `LayerFrame` records. Each frame
tracks:

```text
path
checkpoint_key
node_type
parent_node_type
position_id
token_start_idx
checkpoint
candidates
direct_hit
child_hit
retry_count
retired
```

The state machine emits audit events such as:

```text
compound_layer_started
simple_candidate_observed
layer_window_rule_miss
accepted_generation_time_window
layer_retry_requested
retry_layer_early_closed_after_hit
layer_closed_with_child_hit
layer_closed_with_direct_hit
layer_fallback_committed_without_hit
layer_disappeared_without_hit
closed_without_hit
global_rollback_budget_exhausted
absolute_sampled_token_budget_exhausted
```

Safety maps for retired checkpoint keys, retry counts, and duplicate attempt
hashes intentionally persist across state rollback. This prevents unbounded
retry loops when a model repeatedly emits the same failing layer.

### Generator Rollback

`SawrGenerator` stores checkpoints that include:

```text
model context text
token ids
boundary detector state
state machine snapshot
```

When a retry decision is returned, the generator rolls back the model context,
boundary detector, and state machine together. This avoids mixed-time state
where the generated text and logical layer stack disagree.

Loop-prevention controls:

```text
retry_budget
global_rollback_budget
max_total_sampled_tokens
duplicate attempt hash fallback
final flush rollback disabled
```

The final flush path is deliberately non-rollbacking. It may accept a final
partial window or fallback, but it does not request another generation attempt.

### Pipeline Artifacts

Final rows are detector-clean and have a fixed schema:

```text
schema_version
artifact_type
id
dataset
prompt
final_code
scientific_claims_enabled
```

Final rows must not include audit/debug-only fields such as:

```text
audit_events
audit_only
detector_input_allowed
rule_decision
candidate_hash
layer_path
rollback_checkpoint
```

Audit rows are written to a paired JSONL. They are explicitly marked:

```text
audit_only = true
detector_input_allowed = false
scientific_claims_enabled = false
```

This distinction is part of the current artifact contract: final rows are clean
outputs, while audit rows are experimental traces.

## Experiment Setup

Remote machine:

```text
GPU: NVIDIA GeForce RTX 5090, about 32 GB VRAM
Conda env: /root/miniconda3/envs/WFCLLM
Remote repo copy: /root/autodl-tmp/WFCLLM_sawr_smoke_codex_20260612
Dataset cache: /root/autodl-tmp/WFCLLM/data/datasets
Generator model: /root/autodl-tmp/WFCLLM/data/models/deepseek-coder-7b-instruct/deepseek-ai/deepseek-coder-7b-instruct-v1___5
Encoder model: /root/autodl-tmp/WFCLLM/data/models/codet5-base
Encoder checkpoint: /root/autodl-tmp/WFCLLM/data/checkpoints/encoder/encoder_epoch10.pt
```

Common generation settings:

```text
dataset: humaneval
sample count: 164
max_new_tokens: 256
device: cuda
max_group_statements: 2
retry_budget: 1
global_rollback_budget: 2
temperature: 0.0 default
top_p: 1.0 default
top_k: 0 default
prompt_mode: completion default
```

Semantic LSH runs used:

```text
--rule-name semantic_lsh
--encoder-device cuda
--encoder-use-lora
--encoder-model-path /root/autodl-tmp/WFCLLM/data/models/codet5-base
--encoder-checkpoint-path /root/autodl-tmp/WFCLLM/data/checkpoints/encoder/encoder_epoch10.pt
```

The first semantic run without `--encoder-use-lora` failed because
`encoder_epoch10.pt` contains LoRA keys such as
`encoder.base_model.model...lora_A.default.weight`, while a non-LoRA
`SemanticEncoder` expects plain `encoder.encoder...` keys.

## Experiment Results

### Smoke Run: Five Samples

Initial five-sample semantic behavior check:

```text
final rows: 5
audit rows: 71
accepted_generation_time_window: 12
layer_fallback_committed_without_hit: 3
layer_retry_requested: 2
global_rollback_budget_exhausted: 1
HumanEval correctness: 4 / 5 = 80.00%
```

Per-sample embedding hits:

```text
HumanEval/0: 2 accepted windows, correct=false
HumanEval/1: 6 accepted windows, correct=true
HumanEval/2: 1 accepted window, correct=true
HumanEval/3: 1 accepted window, 3 fallback layers, correct=true
HumanEval/4: 2 accepted windows, correct=true
```

`HumanEval/0` failed because the generated implementation compared only
adjacent elements, while the task requires checking all pairs.

### Full HumanEval Results

All full runs used 164 HumanEval tasks.

| Rule | `lsh_d` | `lsh_gamma` | Samples With Hit | Sample Embed Rate | Layer Success | Layer Fallback | Layer Embed Rate | Accepted Windows | Retry Requests | Budget Exhausted | Pass@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hash` | n/a | n/a | 144/164 | 87.80% | 316 | 71 | 81.65% | 334 | 46 | 18 | 111/164 = 67.68% |
| `semantic_lsh` | 4 | 0.75 | 142/164 | 86.59% | 365 | 7 | 98.12% | 439 | 7 | 0 | 113/164 = 68.90% |
| `semantic_lsh` | 4 | 0.50 | 113/164 | 68.90% | 249 | 118 | 67.85% | 222 | 73 | 26 | 111/164 = 67.68% |
| `semantic_lsh` | 8 | 0.25 | 88/164 | 53.66% | 202 | 144 | 58.38% | 168 | 85 | 32 | 112/164 = 68.29% |

### Artifact Paths

Hash baseline:

```text
data/sawr_full/humaneval_sawr_final_20260613_180655.jsonl
data/sawr_full/humaneval_sawr_audit_20260613_180655.jsonl
data/sawr_full/humaneval_sawr_exec_20260613_180655.jsonl
```

Semantic LSH, `d=4`, `gamma=0.75`:

```text
data/sawr_full_semantic_lsh/humaneval_sawr_final_20260613_181621.jsonl
data/sawr_full_semantic_lsh/humaneval_sawr_audit_20260613_181621.jsonl
data/sawr_full_semantic_lsh/humaneval_sawr_exec_20260613_181621.jsonl
```

Semantic LSH, `d=4`, `gamma=0.50`:

```text
data/sawr_full_semantic_lsh_d4_g05/humaneval_sawr_final_20260613_183209.jsonl
data/sawr_full_semantic_lsh_d4_g05/humaneval_sawr_audit_20260613_183209.jsonl
data/sawr_full_semantic_lsh_d4_g05/humaneval_sawr_exec_20260613_183209.jsonl
```

Semantic LSH, `d=8`, `gamma=0.25`:

```text
data/sawr_full_semantic_lsh_d8_g025/humaneval_sawr_final_20260613_182418.jsonl
data/sawr_full_semantic_lsh_d8_g025/humaneval_sawr_audit_20260613_182418.jsonl
data/sawr_full_semantic_lsh_d8_g025/humaneval_sawr_exec_20260613_182418.jsonl
```

All paths above are relative to the remote workspace
`/root/autodl-tmp/WFCLLM_sawr_smoke_codex_20260612`.

## Interpretation

### Best Current Configuration

`semantic_lsh` with `d=4`, `gamma=0.75` is the strongest tested configuration.

Compared with hash:

```text
layer embed rate: 98.12% vs 81.65%
retry requests: 7 vs 46
fallback layers: 7 vs 71
Pass@1: 68.90% vs 67.68%
```

Compared with stricter semantic settings:

```text
d=4, gamma=0.50: layer embed rate drops to 67.85%
d=8, gamma=0.25: layer embed rate drops to 58.38%
```

The stricter configurations cause many more rollback attempts and budget
exhaustions. They do not improve functional correctness.

### Why `d=4, gamma=0.75` Works Better Here

With `d=4`, there are only 16 possible signatures. `gamma=0.75` selects 12 valid
signatures per keying context, so the rule has a high acceptance capacity. This
fits the current generator and rollback budget: it can usually accept a
structure layer quickly and avoid repeated rollback.

With `d=8, gamma=0.25`, there are 256 signatures and only 64 valid signatures.
The nominal acceptance probability is lower, and the semantic encoder margin
filter can further reduce effective acceptance. In practice this created many
misses, retries, fallback commits, and budget exhaustion events.

### Correctness vs Embedding

Higher embedding coverage did not strongly change HumanEval Pass@1. All tested
runs stayed around 67-69% Pass@1:

```text
hash: 67.68%
semantic_lsh d=4 gamma=0.75: 68.90%
semantic_lsh d=4 gamma=0.50: 67.68%
semantic_lsh d=8 gamma=0.25: 68.29%
```

This suggests that under the current deterministic generation setup, functional
correctness is dominated more by the base model and prompt/stop behavior than
by the watermark rule. The watermark rule primarily changes where the generator
rolls back or accepts, not the model's underlying problem-solving ability.

## Detection-Relevant Reference Facts

The following points are factual properties of the current implementation and
experiments.

Final rows contain:

```text
schema_version
artifact_type
id
dataset
prompt
final_code
scientific_claims_enabled
```

Audit rows contain event-level traces and are marked:

```text
audit_only = true
detector_input_allowed = false
scientific_claims_enabled = false
```

No negative corpus or FPR calibration was run in this experiment set. The
reported metrics are embedding and correctness measurements on generated
watermarked outputs only.

The best observed configuration in this experiment set was:

```text
semantic_lsh
lsh_d = 4
lsh_gamma = 0.75
max_group_statements = 2
CodeT5-base + encoder_epoch10.pt LoRA checkpoint
Python HumanEval completions
```

The poorer observed configurations were:

```text
semantic_lsh, lsh_d=4, lsh_gamma=0.50
semantic_lsh, lsh_d=8, lsh_gamma=0.25
```

Those two configurations produced more retry and fallback events than
`d=4/gamma=0.75`.

## Limitations

1. Only HumanEval was tested in full.
2. Only one generator model was tested: DeepSeek Coder 7B Instruct.
3. Runs used deterministic default decoding (`temperature=0.0`).
4. Only one encoder checkpoint was used: `encoder_epoch10.pt`.
5. No negative corpus or FPR calibration has been run for SAWR detection.
6. Audit traces are experimental traces, not final-row fields.
7. Current Pass@1 is not a robust model-quality benchmark because each task has
   only one deterministic candidate.
8. The current implementation is isolated under `wfcllm/sawr` and is not yet
   integrated into the production `wfcllm/extract` detector.
