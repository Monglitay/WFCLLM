# Anchor Validation Runbook

## Scope

This runbook executes R001-R004 from
`docs/experiment/2026-06-02-anchor-effectiveness-validation-plan.md`.
It does not enable production AO-LSH.

## Build Candidate Pool From Explicit Per-Block Candidates

Preferred input rows contain `candidate_context_id`, `block_text`,
`context_hash`, `context_before`, `context_after`, `masked_parent_context`,
`import_and_helper_signatures`, `node_type`, `parent_node_type`, and
`block_ordinal`.

## Generate Candidates When Full Pools Are Unavailable

Use this path for R001/R002 when existing cap artifacts do not include K
candidates per context. `source_seed_contexts.jsonl` should contain parseable
`source_code` rows from reference solutions or seed completions.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python scripts/anchor_validation.py generate-pool \
  --source-jsonl data/diagnostics/anchor_validation/source_seed_contexts.jsonl \
  --output data/diagnostics/anchor_validation/per_block_candidates.jsonl \
  --sampler-mode hf \
  --lm-model-path data/models/deepseek-coder-7b-base \
  --temperatures 0.2 0.4 0.7 \
  --candidates-per-temperature 16 \
  --max-contexts-per-source 3
```

```bash
python scripts/anchor_validation.py build-pool \
  --input-jsonl data/diagnostics/anchor_validation/per_block_candidates.jsonl \
  --output data/diagnostics/anchor_validation/candidate_pools.jsonl \
  --min-candidates 8
```

## Conservative Whole-Program Fallback

Repeated whole-program candidate rows are accepted only when the masked
non-target context fingerprint matches. Ambiguous rows are rejected rather than
grouped by ordinal alone.

```bash
python scripts/anchor_validation.py build-pool \
  --input-jsonl data/watermarked/candidate_*.jsonl \
  --output data/diagnostics/anchor_validation/candidate_pools_from_programs.jsonl \
  --min-candidates 8 \
  --max-contexts-per-task 3
```

## Smoke Diagnostics With Hash Embeddings

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics \
  --pool data/diagnostics/anchor_validation/candidate_pools.jsonl \
  --output-dir data/diagnostics/anchor_validation/hash_smoke \
  --embedding-mode hash \
  --embed-dim 128 \
  --lsh-d 3 \
  --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 anchor-key-05 anchor-key-06 anchor-key-07 anchor-key-08 anchor-key-09 \
  --methods vanilla random slot context skeleton slot_context slot_context_skeleton prompt_aware seqmark_oracle \
  --gammas 0.25 0.5 0.75 \
  --retry-budgets 1 4 8 16
```

## Real Diagnostics

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics \
  --pool data/diagnostics/anchor_validation/candidate_pools.jsonl \
  --output-dir data/diagnostics/anchor_validation/encoder_r001 \
  --embedding-mode encoder \
  --encoder-model-path data/models/codet5-base \
  --encoder-device cpu \
  --embed-dim 128 \
  --lsh-d 3 \
  --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 anchor-key-05 anchor-key-06 anchor-key-07 anchor-key-08 anchor-key-09 \
  --methods vanilla random slot context skeleton slot_context slot_context_skeleton prompt_aware seqmark_oracle \
  --gammas 0.25 0.5 0.75 \
  --retry-budgets 1 4 8 16
```

Keep `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1` unless the user explicitly asks for online behavior.

## Stop/Go

Continue to end-to-end AO-LSH only if the summary shows deterministic anchors
beating vanilla and random controls, valid-hit balance within the spec
threshold, and retry simulation improvement at B=4 or B=8.
