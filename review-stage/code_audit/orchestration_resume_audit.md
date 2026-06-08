# Orchestration and Resume Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Repair Addendum

Updated: 2026-06-08T12:31:30Z

The stale integration test for phase order was repaired to match the current source-of-truth phase contract:

- `tests/integration/test_orchestration.py:9` imports `OPTIONAL_PHASES`.
- `tests/integration/test_orchestration.py:324-332` now asserts `PHASES`, current `OPTIONAL_PHASES`, and `ALL_PHASES == PHASES + OPTIONAL_PHASES`.

Verification:

- Local focused phase-order tests: 3 passed.
- First remote combined targeted run before test repair: 1 failed, 227 passed, 4 warnings.
- Final remote combined targeted run after repair: 228 passed, 4 warnings.

The actual run-state provenance bug is still open. No run-state schema/provenance validation was implemented, and current `data/run_state.json` remains an invalid experiment bundle.

## Verdict

Current `data/run_state.json` is not a coherent provenance record for the completed experiment. The extract summary/details are paired with a different watermarked artifact than the watermarked artifact recorded as completed in run state. This is a confirmed artifact-pairing problem and is enough to invalidate the current `run.py --status` combination as a method-effect conclusion.

## Run State Evidence

`data/run_state.json` currently reports:

- encoder checkpoint: `data/checkpoints/encoder/encoder_epoch9.pt`
- encoder best model: `data/models/encoder/best_model.pt`
- watermark output: `data/watermarked/humaneval_20260523_130234.jsonl`
- extract details: `data/results/humaneval_full_cap9_details.jsonl`
- extract summary: `data/results/humaneval_full_cap9_summary.json`
- generate-negative output: `data/negative_corpus_llm.jsonl`

The extract summary itself says `meta.input_file = data/watermarked/humaneval_full_cap9.jsonl`. The manifest recorded this as:

- run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- base_config extract.calibration_corpus 'data/negative_corpus.jsonl' differs from run_state generate-negative output 'data/negative_corpus_llm.jsonl'

## Quantified Artifact Mismatch

The semantic probe loaded the run-state watermarked artifact and the run-state extract details:

- `data/watermarked/humaneval_20260523_130234.jsonl`: 55 rows, first IDs `HumanEval/55` through `HumanEval/64`.
- `data/results/humaneval_full_cap9_details.jsonl`: 141 rows, first IDs include `HumanEval/0`, `HumanEval/1`, `HumanEval/2`, `HumanEval/3`, `HumanEval/4`, then gaps.
- Details rows missing from the run-state watermarked file include many IDs beginning with `HumanEval/0`, `HumanEval/1`, `HumanEval/11`, `HumanEval/110`, etc.
- Watermarked IDs missing from details include `HumanEval/74`, `HumanEval/86`, `HumanEval/87`, `HumanEval/97`.

This is not a harmless filename issue. The current status output combines artifacts from different runs.

## Resume Logic Review

- Phase skipping behavior is mechanically correct: `wfcllm/orchestration/pipeline.py:58-67` skips done phases unless forced/eval-only and bypasses extract skip when explicit input exists.
- Extract resume filename validation exists: `wfcllm/extract/pipeline.py:53-58` requires the resume details filename to equal `<input_stem>_details.jsonl`. This prevents obvious resume-file stem mismatches, but it does not validate input content hashes, config hashes, checkpoint hashes, or calibration identity.
- Watermark resume validation is weaker at provenance level: `WatermarkPipeline._validate_resume_path` checks dataset/filename prefix, and ledger alignment checks help row-level resume, but there is no config/checkpoint/tokenizer fingerprint for the entire resumed run.
- Generated artifacts are under `data/` paths and are intended to be gitignored experiment artifacts.

## Calibration Provenance Drift

`configs/base_config.json` uses `extract.calibration_corpus = data/negative_corpus.jsonl`, while run state says the completed `generate-negative` phase wrote `data/negative_corpus_llm.jsonl`. The summary used `data/negative_corpus.jsonl`, so the summary is internally honest, but run state is not a single coherent pipeline trace.

## Verdict on Current Results

The `run.py --status` tuple must not be used as a method-effect result. At minimum, rerun extract with an explicit input artifact and write a fresh run-state/provenance record after the eval-mode issue is fixed or explicitly patched in an audit branch.
