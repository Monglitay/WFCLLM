# SAWR Structure-Aware Generation-Time Embedding Design

Date: 2026-06-12

## Status

Approved direction: implement approach A from the brainstorming discussion.

This design only changes `wfcllm/sawr`. It does not change the older
`wfcllm/watermark` implementation, extraction, calibration, or scientific
claim artifacts.

## Problem

The current SAWR smoke runner treats generated code as a mostly linear stream
of direct function-body simple statements. `PromptAwareBoundaryDetector`
currently walks only the controlled function body's direct children, and
`SawrStateMachine` owns a single `_current_group` plus one group-start
checkpoint.

That behavior does not match the intended SAWR embedding semantics:

- nested control-flow bodies such as `if`, `else`, `for`, and `while` should
  each be evaluated in their own layer;
- `if` / `elif` / `else` belongs to one `if_statement` rollback unit;
- a layer succeeds if any direct statement window in that layer hits, or if
  any child compound layer succeeds;
- a layer should be rolled back only when the layer closes with no direct hit
  and no child hit;
- retry attempts must use the new generated AST structure, not force the model
  to reproduce the first failed attempt's shape.

## Goals

1. Make SAWR generation-time embedding structure-aware for Python function-body
   code.
2. Preserve the smoke runner's final artifact contract:
   `final_code` only, with no hidden audit or detector-only data.
3. Keep `EmbeddingRule` usable for both `HashEmbeddingRule` and
   `SemanticLshEmbeddingRule`.
4. Prevent retry and rollback loops, especially under deterministic decoding.
5. Add focused tests using fake models, tokenizers, and rules.

## Non-Goals

- Do not modify `wfcllm/watermark`.
- Do not implement a SAWR detector or claim TPR/FPR.
- Do not add SAWR audit data to final rows.
- Do not require retry attempts to reproduce branches or statements from a
  discarded failed attempt.
- Do not use post-generation rewriting as the formal embedding mechanism.

## Core Semantics

Each generated compound body is represented as a layer frame. A frame can be
the controlled function body or a nested compound statement body.

Layer success is:

```text
layer_success = has_direct_hit or has_child_hit
```

A direct hit comes from an embedding rule hit on a contiguous direct-statement
window in that layer. A child hit comes from a nested compound layer that
closed successfully.

The first failed attempt's shape is not a template. It only identifies the
rollback checkpoint and the layer that failed. After rollback, the next
generated AST determines what exists and what can succeed.

## If/Else Policy

`if`, `elif`, and `else` are one `if_statement` rollback unit. The detector
must not pop an `if_statement` merely because the first branch hits. It should
keep the `if_statement` frame active until the full compound statement closes
by dedent, sibling, parent close, final flush, or generation stop.

If any branch has a direct hit or contains a successful child layer, the whole
`if_statement` succeeds.

Example:

```python
if cond:
    a = compute()   # hit
else:
    b = fallback()  # miss
```

The `if_statement` succeeds because it has at least one hit.

## Retry Structure Changes

Retry uses the new generated structure.

Case 1: the target compound returns and closes earlier than the failed attempt.

```python
# failed attempt
if cond:
    a = 1
else:
    b = 2

# retry attempt
if cond:
    a = compute()  # hit
```

The retry `if_statement` succeeds. The missing `else` is not a failure because
it belonged to a discarded attempt.

Case 2: the target compound returns and closes without any direct or child hit.

```python
if cond:
    a = compute()  # miss
```

The retry `if_statement` fails and consumes retry budget.

Case 3: the target compound does not return.

```python
# failed target was if_statement
a = compute_without_if()  # hit
```

The old `if_statement` is recorded as disappeared without hit. The sibling
simple statement is processed as a direct candidate in the parent layer. If it
hits, the parent layer succeeds, not the disappeared `if_statement`.

## Boundary Event Model

Replace the candidate-only detector output with boundary events.

Recommended event kinds:

```text
compound_started
simple_candidate
layer_closed
final_flush
```

`compound_started` opens a layer and saves a checkpoint. It does not verify the
compound header text.

`simple_candidate` is the only event kind that enters `EmbeddingRule`.

`layer_closed` closes one or more active layers when the generated AST moves to
a sibling, dedents to a parent, exits the controlled body, or reaches final
flush.

Every event should carry enough metadata for state-machine routing and audit:

```text
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
```

The detector remains prompt-aware:

- HumanEval uses the prompt-owned function body as the controlled root.
- MBPP uses the first generated function body.
- Generated top-level helper functions are ignored unless they are inside the
  controlled root.

## Layer Path and Identity

Layer identity is an implementation detail for generation and retry. It does
not enter final artifacts.

A layer identity should not rely only on `node_type`, because same-type nesting
is common:

```python
if outer:
    if inner:
        x = 1
```

Use a compound identity with at least:

```text
node_type
parent layer id
depth
generated start byte
checkpoint key
```

If a retry produces ambiguous same-type nesting that cannot be matched
confidently, the state machine should avoid assigning child success to the old
target and should process the observed compound as a new child of the current
parent.

## Direct Statement Window

`max_group_statements` becomes the maximum direct-statement window length
within a single layer.

Windows never cross layer boundaries. A parent layer's window does not include
statements from a child compound body.

On a direct `simple_candidate`:

1. Append it to the current layer's direct window.
2. Evaluate the current window with `EmbeddingRule`.
3. If hit:
   - mark `has_direct_hit = true`;
   - increment `accepted_hit_count`;
   - clear that layer's window.
4. If miss and the window is below `max_group_statements`, keep the window.
5. If miss and the window reaches `max_group_statements`, record the failed
   window. Before appending the next direct candidate, drop the oldest
   candidate so the next evaluated window remains a contiguous suffix of this
   layer's direct statements.

When a layer closes, evaluate any remaining nonempty direct window with
`final_flush=True`. If it hits, the layer succeeds. If it misses and no direct
or child hit exists, the layer fails.

## State Machine

Replace the single `_current_group` state with a stack of layer frames.

Each `LayerFrame` should include:

```text
layer_id
node_type
parent_node_type
position_id
checkpoint_before_layer
checkpoint_key
window
has_direct_hit
has_child_hit
retry_count
attempt_hashes
closed_reason
```

The state machine should still expose aggregate counters:

```text
accepted_hit_count
closed_without_hit_count
fallback_count
candidate_count
retry_count
global_rollback_count
```

Close behavior:

```text
if has_direct_hit or has_child_hit:
    mark layer success
    pop layer
    set parent.has_child_hit = true
else if retry budget remains and rollback is allowed:
    request rollback to checkpoint_before_layer
else:
    mark layer fallback/no-hit
    pop layer
    bubble no-hit result to parent
```

The controlled function-body root should not be rolled back past the prompt.
If it closes without hit, record `closed_without_hit`.

## Rollback Snapshots

Rollback must restore all state that affects future boundary emission and
window decisions.

Extend rollback snapshots so that a rollback restores:

```text
generated ids and generated text
KV cache snapshot
next logits
boundary detector state
state machine snapshot
absolute sampled token counter remains unchanged
global rollback counter remains unchanged
retired checkpoint keys remain unchanged
```

The key rule is that rollback restores local attempt state but does not erase
global safety counters.

## Loop Prevention

The implementation must satisfy this termination invariant:

Every nontrivial transition either advances final generated code, consumes a
retry or rollback budget, retires a checkpoint, or stops generation.

Required safety controls:

```text
per-layer retry budget
global rollback budget
absolute sampled token budget
retired checkpoint keys
attempt hashes
```

`generated_ids` length is not sufficient as a stop condition because rollback
shortens it. Track an absolute sampled-token counter that never decreases on
rollback.

If deterministic decoding regenerates the same failed attempt for the same
checkpoint, detect the repeated attempt hash, consume retry budget, and
fast-fail that attempt instead of repeating the full loop.

When a checkpoint's retry budget is exhausted, mark it retired for the current
parent attempt. Do not roll back to that checkpoint again unless an outer
rollback starts a new parent attempt, and even then the global rollback budget
still applies.

## Configuration

Preserve existing CLI names where possible:

- `max_group_statements`: direct-statement window cap per layer;
- `retry_budget`: per-layer retry budget.

Add bounded-generation controls:

- `global_rollback_budget`: maximum rollbacks per sample;
- `max_total_sampled_tokens`: absolute sampled-token cap per sample.

If `max_total_sampled_tokens` is omitted, derive a conservative default from
`max_new_tokens` and `retry_budget`, for example:

```text
max_new_tokens * max(2, retry_budget + 2)
```

## Audit Events

Final rows remain unchanged. Audit rows may gain structure-aware events.

Recommended new audit events:

```text
compound_layer_started
simple_candidate_observed
layer_window_rule_miss
accepted_generation_time_window
layer_retry_requested
retry_layer_early_closed_after_hit
layer_disappeared_without_hit
layer_closed_with_child_hit
layer_fallback_committed_without_hit
global_rollback_budget_exhausted
absolute_sampled_token_budget_exhausted
```

Audit rows remain:

```text
audit_only = true
detector_input_allowed = false
scientific_claims_enabled = false
```

Do not include audit data in final detector inputs.

## Stop Sequence and Final-Code Consistency

If stop-sequence truncation removes text after generation, accepted hits in the
removed suffix must not count toward final statistics.

The generator should either:

1. stop before processing boundary events beyond the retained final-code
   boundary; or
2. re-parse `final_code` after truncation and verify that accepted audit hit
   candidate hashes still appear in retained code.

The first option is preferred for the smoke runner.

## Implementation Scope

Expected files:

```text
wfcllm/sawr/boundary.py
wfcllm/sawr/state_machine.py
wfcllm/sawr/generator.py
wfcllm/sawr/config.py
wfcllm/sawr/pipeline.py
wfcllm/sawr/__init__.py
scripts/run_sawr_smoke.py
tests/sawr/
tests/integration/test_sawr_smoke_cli.py
```

Do not modify:

```text
wfcllm/watermark/
wfcllm/extract/
```

## Testing Plan

Boundary detector tests:

- recursively emits simple candidates inside `if`, `else`, `for`, and `while`;
- emits `compound_started` for headers but does not verify header text;
- treats `if` / `elif` / `else` as one `if_statement` rollback unit;
- emits layer-close events on dedent, sibling, parent close, and final flush;
- restores emitted keys and layer tracking after detector rollback.

State machine tests:

- direct hit makes the current layer succeed;
- child layer hit bubbles to parent;
- any branch hit makes the whole `if_statement` succeed;
- all windows miss and layer closes triggers layer rollback;
- retry layer early-close after hit succeeds;
- target compound disappearance routes sibling candidates to the parent layer;
- deterministic repeated attempt consumes retry budget and terminates;
- retired checkpoint is not rolled back to again in the same parent attempt;
- global rollback budget and absolute sampled-token budget stop generation.

Generator tests:

- fake model generates nested `if/else`, misses, rolls back, then early-closes
  after a hit;
- fake model repeatedly regenerates the same failing text and exits by budget;
- absolute sampled-token count does not decrease after rollback;
- stop-sequence truncation does not count hits in removed suffix.

Validation commands:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_sawr_smoke_cli.py -v
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

## Acceptance Criteria

- SAWR detects and handles nested Python statement layers during generation.
- `if` / `elif` / `else` is one rollback unit.
- A layer succeeds when it has at least one direct or child hit.
- Retry uses the newly generated AST structure.
- Early close after hit succeeds.
- Disappeared target compounds do not succeed, but replacement sibling
  statements can make the parent layer succeed.
- Rollback loops are bounded by per-layer retry, global rollback, attempt hash,
  retired checkpoint, and absolute sampled-token controls.
- Final SAWR rows remain detector-clean and audit-free.
