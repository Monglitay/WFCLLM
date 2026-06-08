# Calibration and Negative Corpus Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Verdict

Calibration provenance is inconsistent across run state and summary metadata. The threshold formula implementation is mostly coherent, but the direct calibration script appears stale/broken because it loads a raw Hugging Face `AutoModel` instead of the project `SemanticEncoder` checkpoint. Current calibrated summary values are not tied to a coherent run-state pipeline and should not be used as final evidence.

## Current Artifacts

- `data/negative_corpus.jsonl`: 164 rows, sha256 `3f05cc2e453d32c7c67ad69f92959388373b946aafb73e8027e64ccdee17c7e0`.
- `data/negative_corpus_llm.jsonl`: 164 rows, sha256 `f33e7eb91406516ee91c95158cbe9fa2b00b1ac12d7244d4b1344cf450f27bf2`.
- Run state says `generate-negative` completed with `data/negative_corpus_llm.jsonl`.
- `data/results/humaneval_full_cap9_summary.json` says calibration source was `data/negative_corpus.jsonl`, FPR 0.05, threshold 1.519126609118742, hypothesis mode adaptive, joint threshold 1.843003476632067.

This is internally valid for that summary file but not coherent with the current run-state pipeline.

## Threshold Code Review

- Inline extract calibration in `wfcllm/cli/runners.py:500-544` uses `ThresholdCalibrator`, `BlockScorer`, `WatermarkKeying`, `LSHSpace`, and `ProjectionVerifier` around the same encoder object used by extraction. That path inherits the eval-mode bug if encoder was not set to eval.
- Joint threshold calibration at `wfcllm/cli/runners.py:546-600` uses the same negative corpus and a temporary detector, but calls `_tmp_detector.detect(_code)` without watermark metadata. If positives are adaptive metadata mode and joint negatives are effectively fixed/no-metadata mode, joint calibration can mismatch the evaluated positive distribution.
- Direct script path `scripts/calibrate_threshold.py` dispatches `wfcllm/extract/calibration/runner.py`; `runner.py:47-55` loads `AutoTokenizer` and `AutoModel` directly, not `SemanticEncoder` or the fine-tuned checkpoint. `ProjectionVerifier` expects an encoder returning an embedding tensor. This direct script should be treated as stale until a failing test/fix is approved.

## Leakage / Pairing Risk

The negative corpus shares HumanEval task IDs with watermarked artifacts. ID overlap alone is normal for benchmark negative calibration, but because run state, watermarked artifact, and details artifact are already mismatched, calibration leakage/pairing cannot be safely ruled out from current artifacts alone.

## Negative Quality Probe

The read-only dataset/negative probe found no empty IDs, prompts, or generated code in loaded datasets. The manifest found no duplicate IDs in key negative JSONL files.

## Verdict on Current Results

The calibration threshold recorded in `humaneval_full_cap9_summary.json` may be the threshold used for that summary, but it is not traceable to the run-state `generate-negative` output and not reproducible under the current eval-mode-safe detector. Treat calibrated rates as historical artifacts, not method claims.
