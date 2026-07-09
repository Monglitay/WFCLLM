# Reproducing `evidence_retry_seed7x3`

This document records the official reproduction entry points for the default WFCLLM preset.

## Commands

```bash
python run.py --config configs/wfcllm/repro_humaneval_full164.json
python run.py --phase generate --config configs/wfcllm/evidence_retry_seed7x3.json
python run.py --phase calibrate --negative-input data/negative/reference_final_code.jsonl
python run.py --phase detect --input data/runs/<run_id>/inputs/final_code.jsonl
```

Run from the repository root in the `WFCLLM` conda environment. For pytest and local validation, set `HF_HUB_OFFLINE=1` unless online behavior is explicitly required.

These commands are the official `run.py` reproduction contract. In this refactor branch, the mainline phase runners currently provide phase names and run-state behavior while the low-level artifact-producing entry points remain available through scripts such as `scripts/wfcllm_generate.py`.

## Expected Artifact Root

The official run root is:

```text
data/runs/<run_id>/
```

The official detector input for generated positives is:

```text
data/runs/<run_id>/inputs/final_code.jsonl
```

## Expected Known Metrics

| Metric | Value |
| --- | --- |
| pass@1 | 104/164 = 63.41% |
| AUROC | 0.660080 |
| TP@<=8FP | 43/164 |
| FP | 8 |
| positive insufficient | 20 |
| model-negative AUROC | 0.510332 |

## Appendix A: Preset Values

### Method

| Key | Value |
| --- | --- |
| name | `evidence_retry_seed7x3` |
| strict_no_quality_gate | `true` |
| strict_code_only_detector | `true` |

### Generation

| Key | Value |
| --- | --- |
| dataset | `humaneval` |
| sample_limit | `164` in the full reproduction config |
| max_new_tokens | `256` |
| temperature | `0.25` |
| top_p | `0.95` |
| top_k | `0` |
| retry_repetition_penalty | `4.0` |
| torch_dtype | `bf16` |
| device | `cuda` |
| seed | `7` |
| load_in_4bit | `true` |
| prompt_mode | `completion` |
| max_group_statements | `2` |
| retry_budget | `2` |
| statement_retry_budget | `4` |
| window_retry_budget | `3` |
| compound_retry_budget | `2` |
| global_rollback_budget | `180` |
| max_total_sampled_tokens | `32768` |
| evidence_retry_attempts | `3` |
| evidence_retry_seed_stride | `101` |

### Semantic LSH

| Key | Value |
| --- | --- |
| rule_name | `semantic_lsh` |
| lsh_d | `4` |
| lsh_gamma | `0.25` |
| semantic_margin | `0.0` |
| use_ordinal_keying | `false` |

### Detector

| Key | Value |
| --- | --- |
| lsh_d | `4` |
| gamma | `0.25` |
| semantic_margin | `0.0` |
| max_group_statements | `2` |
| min_scoreable_contexts | `1` |
| min_proxy_windows | `2` |
| target_fpr | `0.05` |
| use_ordinal_keying | `false` |
| evidence_mode | `hit_plus_margin` |
| statistic | `calibrated_context_mean_proxy_penalized` |
| proxy_penalty_alpha | `0.4` |

### Artifacts And Runtime

| Key | Value |
| --- | --- |
| run_root | `data/runs` |
| default phases | `generate`, `calibrate`, `detect`, `report`, `audit` |
