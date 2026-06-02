# Anchor Validation Runbook

## Scope

This runbook executes R001-R004 from
`docs/experiment/2026-06-02-anchor-effectiveness-validation-plan.md`.
It does not enable production AO-LSH.

The current R001 encoder diagnostics are **not a GO result**. The strongest
signal is `seqmark_oracle` (`+0.1994` mean entropy over vanilla, CI
`[+0.1384, +0.2746]`, win rate `1.0`), which shows the candidate pool contains
usable region signal. `slot_context` and `slot_context_skeleton` have weak
positive signal over random (`+0.0204` and `+0.0240` mean entropy), but the
comparison against vanilla is unstable (`slot_context_vs_vanilla` CI crosses
zero and win rate is `0.4667`). The valid-hit balance is also poor
(`max_delta_gamma = 0.75`), so entropy gain alone is not enough to justify
production AO-LSH.

The revised objective is diagnostic: identify why current anchor construction
uses only a small part of the oracle signal, improve anchor construction with
role-aware context, quantify the oracle gap, and diagnose gamma calibration.
Production AO-LSH remains disabled until the layered gates below pass.

## Diagnostic Outputs

`run-diagnostics` writes:

- `region_metrics.jsonl`: entropy, collapse, effective region count, Hamming
  diversity, and valid-hit balance rows per context/method/key/gamma.
- `selection_simulation.jsonl`: offline retry-budget selections and quality
  proxies.
- `anchor_validation_summary.json`: data quality, oracle gap, method-oracle
  agreement proxy, empirical gamma calibration, retry, and layered gate
  evidence.

The summary includes pool quality diagnostics:

- `context_count`
- `total_candidates`
- `candidates_per_context` min / median / mean / max
- `unique_candidates_per_context` min / median / mean / max
- `node_type_distribution`
- `parent_node_type_distribution`
- `task_distribution`
- `candidate_length` min / median / mean / max
- `parse_valid_rate`
- `syntax_valid_rate`
- `embedding_pairwise_diversity` when embeddings are available

The method-oracle agreement field currently uses the
`top_rank_candidate_signature_match` proxy: for each context/key/method, it
compares the region signature of the rank-0 candidate against the
`seqmark_oracle` signature for that same rank-0 candidate. This is not direct
candidate selection agreement, but it is a stable proxy for whether a method's
region partition aligns with the oracle partition on the model-preferred
candidate.

Empirical gamma calibration is diagnostic-only in this revision. For every
context/method/key/gamma it reports `target_gamma`, `empirical_gamma`, and
`delta` in the summary aggregation. It does not change candidate selection.

## Main Method Set

The default `run-diagnostics` method set is:

- `vanilla`
- `random`
- `context`
- `slot_context`
- `slot_context_skeleton`
- `role_aware_slot_context`
- `role_aware_slot_context_skeleton`
- `seqmark_oracle`

`slot`, `skeleton`, and `prompt_aware` remain available as optional ablations
through explicit `--methods`, but they are not the main conclusion path.

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
  --gammas 0.25 0.5 0.75 \
  --retry-budgets 1 4 8 16
```

Keep `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1` unless the user explicitly asks for online behavior.

## Stop/Go

Do not continue to end-to-end AO-LSH unless the summary's layered gates pass:

- `data_quality_gate`: `context_count >= 40`, median candidates per context
  `>= 8`, at least 4 node type categories, and parse-valid rate close to 1.0.
- `anchor_signal_gate`: `role_aware_slot_context_minus_random` CI lower bound
  is positive and win rate is at least `0.60`.
- `vanilla_improvement_gate`: `role_aware_slot_context_vs_vanilla` mean gain is
  greater than `0.03`, CI lower bound is non-negative or near zero, and oracle
  gain ratio is at least `0.15`.
- `balance_gate`: `max_delta_gamma <= 0.35` and `mean_delta_gamma <= 0.15`.
- `retry_gate`: budget-4 hit acquisition improves over both vanilla and random,
  and fallback rate is at most `0.15`.

The old `first_stage_passed` field is retained, but it is now derived from
these layered gates. A failed gate includes concrete failed reasons in
`anchor_validation_summary.json`.

## Inspect Data Quality Summary

After running diagnostics:

```bash
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path(
    "data/diagnostics/anchor_validation/encoder_r001/anchor_validation_summary.json"
).read_text(encoding="utf-8"))
print(json.dumps(summary["summary"]["pool_quality"], indent=2, ensure_ascii=False))
print(json.dumps(summary["go_no_go"]["gates"], indent=2, ensure_ascii=False))
PY
```

Use the pool-quality section before interpreting entropy results. A narrow
node-type distribution or low candidate count can explain weak role-aware
anchor gains even when the oracle remains strong.
