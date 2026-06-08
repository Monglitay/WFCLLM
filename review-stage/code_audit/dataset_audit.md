# Dataset Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Verdict

HumanEval and MBPP adapters/loaders are behaving consistently under offline mode in the probe. Dataset loading is not a confirmed root cause of poor current results. There is still provenance risk because calibration/evaluation corpora use the same task IDs as watermarked corpora and because run state points to a different negative corpus than the extract summary used.

## Code Review

- Dataset registry layer is side-effect imported through `wfcllm/datasets/__init__.py`; implemented adapters are HumanEval and MBPP, with HumanEvalPack as a not-yet-implemented skeleton.
- HumanEval adapter maps rows to stable `id`, `prompt`, and reference/generated code fields; MBPP does the same for MBPP records.
- Local loaders use `datasets.load_dataset(..., cache_dir=..., download_mode="reuse_cache_if_exists")`; offline env variables are necessary to prevent Hub fallback.
- HumanEvalPack is registered but intentionally raises `NotImplementedError`, so it is not accidentally used for current Python runs.

## Probe Evidence

The read-only probe loaded datasets with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`:

- HumanEval prompts: 164 rows, first IDs `HumanEval/0` to `HumanEval/4`, no duplicate IDs, no empty IDs, no empty prompts.
- HumanEval references: 164 rows, no duplicate IDs, no empty generated code.
- MBPP prompts/references: 974 rows, first IDs `mbpp/601` to `mbpp/605`, no duplicate IDs, no empty prompts or generated code.
- Loader stdout confirmed cached local datasets were used under offline mode.

## Negative Corpus Notes

- `data/negative_corpus.jsonl` has 164 rows and is the calibration source in `data/results/humaneval_full_cap9_summary.json`.
- `data/negative_corpus_llm.jsonl` also has 164 rows and is the run-state `generate-negative` output.
- ID overlap with watermarked task IDs is expected for same benchmark tasks but must not be confused with artifact pairing. Leakage cannot be ruled out by ID alone; compare generated code/provenance when using those corpora for claims.

## Tests

`tests/datasets/ -v --tb=short` passed: 9 passed.
