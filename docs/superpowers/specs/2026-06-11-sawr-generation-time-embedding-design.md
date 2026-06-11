# SAWR Generation-Time Embedding Smoke Design

Date: 2026-06-11
Status: approved for implementation planning

## Context

This design turns `docs/SAWR_GENERATION_TIME_EMBEDDING_SCHEME.md` into a
minimal engineering plan for this branch. The branch goal is narrow: run a
generation-time embedding smoke path where candidate statements are accepted,
retried, or fallen back before the generated code is treated as final.

The implementation must not extend the current semantic-LSH watermark pipeline.
The existing `wfcllm.watermark` path still carries encoder, LSH, adaptive gamma,
token-channel, extraction, and diagnostics concepts. SAWR smoke must avoid that
surface so it does not accidentally make detection or scientific claims.

The first implementation will be an isolated runner, not a new `run.py --phase`.

## Decisions

- Add a new package under `wfcllm/sawr/`.
- Add a thin script entry point at `scripts/run_sawr_smoke.py`.
- Read prompts through the existing `humaneval` / `mbpp` dataset loader.
- Write outputs under `data/sawr/`.
- Do not write into `data/watermarked/`.
- Do not update `data/run_state.json`.
- Do not call `wfcllm.extract`.
- Do not require encoder checkpoints, CodeT5, LSH, adaptive gamma, or
  token-channel artifacts.
- Use a pluggable embedding-rule interface with a deterministic hash rule as
  the default.
- In v1, only simple Python statements enter candidate groups.
- Keep candidate typing extensible so a later version can add simple if-return
  block candidates.
- Use rollback/retry when a group reaches `max_group_statements` and still
  misses.
- Use strict SAWR artifacts: final rows contain `final_code`, not
  `generated_code`, and no `watermark_params`, `blocks`, or `embed_rate`.

## Non-Goals

This design intentionally does not implement or claim:

- extraction;
- detector scoring;
- thresholds;
- calibration;
- FPR, TPR, AUROC, pass@1, or pass-cost metrics;
- semantic capacity;
- correctness-preserving generation;
- matched-null or controlled experiments;
- model training or fine-tuning;
- integration with the current `watermark` phase.

The smoke result may say that the SAWR runner produced final-code artifacts and
audit-only provenance. It must not say that a watermark was detected.

## Architecture

The isolated SAWR path is composed of six modules plus one script.

### `wfcllm/sawr/config.py`

Defines small dataclasses:

- `SawrGenerationConfig`
- `SawrRuleConfig`
- `SawrPipelineConfig`

The config surface should include only smoke-runner concerns:

- dataset and dataset path;
- output directory;
- local LM model path;
- sample limit and offset;
- generation parameters such as `max_new_tokens`, `temperature`, `top_p`,
  `top_k`, `torch_dtype`, and `device`;
- deterministic seed;
- `max_group_statements`;
- `retry_budget`;
- rule name and rule parameters.

It must not include semantic-LSH or detector parameters such as `lsh_d`,
`lsh_gamma`, `fpr_threshold`, calibration paths, or token-channel settings.

### `wfcllm/sawr/rules.py`

Defines the rule interface:

```python
class EmbeddingRule(Protocol):
    def evaluate(self, request: RuleRequest) -> RuleDecision:
        ...
```

`RuleRequest` contains:

- `sample_id`;
- `position_id`;
- `candidates`;
- `seed`;
- `final_flush`.

`RuleDecision` contains:

- `hit: bool`;
- `reason: str`;
- `rule_name: str`.

The default `HashEmbeddingRule` computes a stable digest over:

```text
seed || sample_id || position_id || normalized_candidate_group_text || final_flush
```

The digest is mapped to `[0, 1)`, then compared with a configured
`target_accept_rate`. This rule is deterministic, local, cheap, and testable.
It is only a smoke predicate. It is not a detector and does not define a future
extraction method.

### `wfcllm/sawr/boundary.py`

Defines a prompt-aware Python boundary detector. This is separate from the
existing `StatementInterceptor` because the existing interceptor parses only the
generated continuation. SAWR must often parse `prompt + generated_text`,
especially for HumanEval prompts where the model emits an indented function
body.

The boundary detector responsibilities are:

- parse `prompt + generated_text` with the existing Python parser utilities;
- locate the controlled Python function body;
- emit complete simple-statement candidates whose source span is generated
  text, not already-present prompt text;
- derive a stable `position_id`, normally `module.<function_name>.body`;
- flush a pending simple statement at generation end;
- avoid emitting comment-only, whitespace-only, string-fragment, or malformed
  parse fragments as candidates.

For HumanEval, the target function is the last function definition present in
the prompt plus continuation. For MBPP, the prompt may be natural language, so
the detector waits for a generated Python function definition and then uses the
first generated function body. MBPP support is best-effort in v1; HumanEval is
the reliable smoke target.

The boundary detector emits `Candidate` records:

```python
@dataclass(frozen=True)
class Candidate:
    text: str
    candidate_type: str  # "simple_statement" in v1
    node_type: str
    position_id: str
    token_start_idx: int
    token_count: int
```

`candidate_type` is intentionally explicit so simple if-return blocks can be
added later without changing the state machine contract.

### `wfcllm/sawr/state_machine.py`

Owns the online embedding logic and contains no model-loading or file I/O.

The state machine tracks:

- current `sample_id`;
- current `position_id`;
- current candidate group;
- group-start checkpoint identity;
- accepted hit count;
- closed-without-hit count;
- retry count for the current group;
- `max_group_statements`;
- `retry_budget`;
- audit events.

The core transitions are:

1. When a simple-statement candidate arrives and the current group is empty,
   save the group-start checkpoint.
2. Append the candidate to the current group.
3. Evaluate the rule after every append.
4. If the rule hits:
   - accept the current generated context as committed;
   - emit an `accepted_generation_time_group` audit event;
   - increment `accepted_hit_count`;
   - clear the current group and retry counter.
5. If the rule misses and the group length is below `max_group_statements`,
   continue generating.
6. If the rule misses and the group length reaches `max_group_statements`:
   - if retry budget remains, request rollback to the group-start checkpoint;
   - clear the current group;
   - increment retry count;
   - generate a replacement group;
   - if retry budget is exhausted, keep the last generated group as normal
     code, emit `fallback_committed_without_hit`, and clear the group.
7. At generation end, flush any non-empty group:
   - evaluate with `final_flush=true`;
   - if hit, accept and emit `accepted_generation_time_group`;
   - if miss, keep normal code and emit `closed_without_hit`;
   - clear the group.

The term "commit" means "the runner will not roll back this generated region
any further." During token generation, candidate text exists tentatively in the
generation context. It becomes committed only when the state machine accepts,
closes, or falls back the group.

### `wfcllm/sawr/generator.py`

Connects the local causal LM sampler, prompt-aware boundary detector, and state
machine.

It may reuse existing low-level generation infrastructure where useful:

- token sampling and KV-cache rollback concepts from `GenerationContext`;
- checkpoint shapes from the current watermark retry path;
- the Python parser utilities under `wfcllm.lang.python`.

It must not instantiate or call:

- `WatermarkGenerator`;
- `ProjectionVerifier`;
- `SemanticChannel`;
- `RetryLoop`;
- `CascadeManager`;
- token-channel runtime;
- anything under `wfcllm.extract`.

The generator loop is:

1. Format the dataset prompt for the local LM.
2. Prefill the model.
3. Generate tokens until EOS or `max_new_tokens`.
4. After each token, ask the prompt-aware boundary detector whether a complete
   candidate is available.
5. For each candidate, pass it to the state machine with the candidate's
   group-start checkpoint.
6. If the state machine asks for rollback, restore the model, generated text,
   token ids, and boundary detector state to the group-start checkpoint.
7. Continue until generation finishes.
8. Flush the state machine.
9. Return `SawrGenerateResult`.

`SawrGenerateResult` contains:

- `final_code`;
- `accepted_hit_count`;
- `closed_without_hit_count`;
- `fallback_count`;
- `candidate_count`;
- `audit_events`.

For HumanEval body-only generation, `final_code` is `prompt + generated_body`.
If an instruct model repeats the function signature, the runner may strip the
repeated prompt function before building `final_code`, using a conservative
variant of the existing pipeline helper. Any stripping must be recorded only in
audit provenance and must not introduce detector-facing hidden data.

### `wfcllm/sawr/pipeline.py`

Reads prompts and writes artifacts.

The pipeline uses `wfcllm.datasets.loaders.local.load_prompts()` with:

- `dataset="humaneval"` or `dataset="mbpp"`;
- `dataset_path`;
- `sample_limit`;
- `sample_offset`.

It writes two JSONL files under `data/sawr/`:

```text
data/sawr/<dataset>_sawr_final_<timestamp>.jsonl
data/sawr/<dataset>_sawr_audit_<timestamp>.jsonl
```

Resume support in v1 is sample-id based:

- if `--resume latest` is passed, find the latest matching final file in
  `data/sawr/`;
- skip final rows whose `id` already exists;
- append new final rows to the final file;
- append new audit rows to the paired audit file.

If the paired audit file is missing during resume, the pipeline should fail
fast. Audit is not detector input, but resume without audit would make smoke
debugging ambiguous.

### `scripts/run_sawr_smoke.py`

Thin command-line entry point.

Recommended smoke command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python scripts/run_sawr_smoke.py \
  --dataset humaneval \
  --dataset-path data/datasets \
  --model-path data/models/deepseek-coder-7b-instruct/deepseek-ai/deepseek-coder-7b-instruct-v1___5 \
  --output-dir data/sawr \
  --sample-limit 10 \
  --max-new-tokens 64 \
  --temperature 0.0 \
  --torch-dtype bf16 \
  --device cuda
```

The script can expose `--load-in-4bit` for machines that need bitsandbytes, but
4-bit loading is a runtime option, not part of the SAWR method.

## Data Flow

For each sample:

1. Load `{id, prompt}` from the dataset loader.
2. Build a model prompt.
3. Begin LM generation.
4. Maintain `generated_text` as tentative text.
5. Parse `prompt + generated_text` to detect complete simple statements inside
   a Python function body.
6. Feed each candidate to the state machine.
7. On rule hit, keep the generated text and clear the current group.
8. On full-group miss with remaining retries, roll back to the group-start
   checkpoint and regenerate.
9. On retry exhaustion, keep the latest group as normal code and mark it as
   fallback without embedding success.
10. At EOS or token budget, flush the current group.
11. Write one final-code row and zero or more audit rows.

## Artifact Contract

### Final JSONL

Each row stores only task-visible context and final code. It must not contain
audit, process, or detector fields, and it is not an extract input for the
existing detector.

```json
{
  "artifact_type": "sawr_final_code",
  "schema_version": "sawr-smoke/v1",
  "id": "HumanEval/0",
  "dataset": "humaneval",
  "prompt": "def has_close_elements(numbers, threshold):\n    ...",
  "final_code": "def has_close_elements(numbers, threshold):\n    ...",
  "scientific_claims_enabled": false
}
```

Forbidden fields in final rows:

- `generated_code`;
- `watermark_params`;
- `blocks`;
- `embed_rate`;
- `p_value`;
- `z_score`;
- `is_watermarked`;
- `tpr`;
- `fpr`;
- `correctness_result`;
- `pass_cost`;
- `detector_score`;
- `generation_ledger`;
- `retry_ledger`.

### Audit JSONL

Audit rows are developer-only provenance.

```json
{
  "artifact_type": "sawr_audit_event",
  "schema_version": "sawr-smoke/v1",
  "id": "HumanEval/0",
  "audit_only": true,
  "detector_input_allowed": false,
  "scientific_claims_enabled": false,
  "position_id": "module.has_close_elements.body",
  "event": "accepted_generation_time_group",
  "candidate_type": "simple_statement",
  "group_statement_count": 2,
  "final_flush": false,
  "rule_name": "hash",
  "decision": "hit"
}
```

Allowed audit events in v1:

- `candidate_observed`;
- `group_rule_miss`;
- `accepted_generation_time_group`;
- `rollback_requested`;
- `fallback_committed_without_hit`;
- `closed_without_hit`;
- `sample_failed`.

Audit rows may include candidate hashes and short factual reasons. They should
not include hidden material that a detector would need. They must not use names
that imply statistical detection, such as `detector_score`, `p_value`, or
`watermark_params`.

## Error Handling

The runner should fail fast for invalid configuration:

- unsupported dataset;
- missing local model path;
- non-positive `max_group_statements`;
- negative `retry_budget`;
- invalid target accept rate outside `[0, 1]`;
- missing paired audit file when resuming.

Per-sample generation failures should not abort the whole smoke run unless the
failure is a global configuration or model-loading problem. The pipeline should
write a `sample_failed` audit event and continue to the next sample.

If no function body can be found, the runner should still write a final-code
row when the model produced code, and it should write an audit event with:

```json
{
  "event": "closed_without_hit",
  "reason": "no_controlled_function_body"
}
```

This is not an embedding success.

If generation hits `max_new_tokens`, the state machine must flush the current
group before returning.

## Testing Plan

Unit tests should avoid loading real models.

Required tests:

- `HashEmbeddingRule` is deterministic for the same seed and input.
- Changing sample id, position id, or group text can change the hash decision.
- State machine clears the group after a hit.
- State machine does not reuse candidates from a previous hit.
- State machine flushes a partial group at generation end.
- State machine requests rollback when a full group misses and retries remain.
- State machine falls back after retry exhaustion.
- Prompt-aware boundary detector can emit simple statements from a HumanEval
  prompt plus an indented generated body.
- Boundary detector does not emit prompt-existing statements as generated
  candidates.
- Pipeline writes final rows without forbidden fields.
- Pipeline writes audit rows with `audit_only=true` and
  `detector_input_allowed=false`.
- Resume skips completed sample ids and requires the paired audit file.

Targeted command:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/ -v
```

Smoke syntax check:

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

After implementation, a real-model smoke run should process 1 to 10 HumanEval
samples. It should be reported only as a generation-time control smoke, not as
a detection or correctness experiment.

## Future Work

Future specs can cover:

- adding simple if-return block candidates;
- promoting the runner to a formal phase;
- defining a final-code-only extraction method;
- adding controlled experiments after extraction is defined;
- deciding whether SAWR replaces or coexists with the current semantic-LSH
  watermark path.

Those items are intentionally out of scope for this first smoke implementation.
