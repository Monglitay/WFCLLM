# 2026-03-19 Extract Repair Verification

## Commands

- Unit slice:
  - `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest -q tests/test_run.py tests/test_run_config.py tests/watermark/test_pipeline.py tests/extract/test_pipeline.py tests/extract/test_extract_param_resolution.py tests/watermark/test_embed_metadata_roundtrip.py tests/experiment/embed_extract_alignment/test_aligner.py tests/watermark/test_generator_integration.py`
- Extract replay:
  - `HF_HUB_OFFLINE=1 /root/miniconda3/envs/WFCLLM/bin/python run.py --phase extract --force --config configs/base_config.json --input-file /root/autodl-tmp/WFCLLM/data/watermarked/humaneval_20260318_204554.jsonl --secret-key 1010 --calibration-corpus /root/autodl-tmp/WFCLLM/data/negative_corpus.jsonl`
- Prompt spot checks:
  - `HF_HUB_OFFLINE=1 /root/miniconda3/envs/WFCLLM/bin/python tools/debug_extract_alignment.py --prompt-id HumanEval/128 --input-file /root/autodl-tmp/WFCLLM/data/watermarked/humaneval_20260318_204554.jsonl --use-embedded-params`
  - `HF_HUB_OFFLINE=1 /root/miniconda3/envs/WFCLLM/bin/python tools/debug_extract_alignment.py --prompt-id HumanEval/23 --input-file /root/autodl-tmp/WFCLLM/data/watermarked/humaneval_20260318_204554.jsonl --use-embedded-params`
  - `HF_HUB_OFFLINE=1 /root/miniconda3/envs/WFCLLM/bin/python tools/debug_extract_alignment.py --prompt-id HumanEval/45 --input-file /root/autodl-tmp/WFCLLM/data/watermarked/humaneval_20260318_204554.jsonl --use-embedded-params`

## Unit Verification

- Result: `73 passed in 3.13s`

## Extract Replay

- Archived summary before fix: `/root/autodl-tmp/WFCLLM/data/results/humaneval_20260318_204554_summary.json`
- Replayed summary after fix: `data/results/humaneval_20260318_204554_summary.json`
- Calibration output:
  - `M_r = 2.8933`
  - `FPR = 0.01`
  - `n_samples = 164`

### Before / After

- Watermark rate:
  - Before: `0.12195121951219512`
  - After: `0.0`
- Mean z-score:
  - Before: `0.970238352788241`
  - After: `-0.31271312992083394`
- Mean sample hit ratio:
  - Before: `0.6657496451436811`
  - After: `0.7061319014140172`
- Global hit ratio:
  - Before: `0.6906202723146747`
  - After: `0.68910741301059`

## Prompt Spot Checks

All three spot-check commands resolved extract parameters to `lsh_d=4`, `lsh_gamma=0.75`.

### HumanEval/128

- Old detail:
  - `z_score = -2.5298221281347035`
  - `hits = 1 / 10`
- New detail:
  - `z_score = -4.016632088371218`
  - `hits = 2 / 10`
- Debug script:
  - Resolved params: `4 / 0.75`
  - No matching prompt-level diagnostic report auto-discovered

### HumanEval/23

- Old detail:
  - `z_score = -1.1338934190276817`
  - `hits = 2 / 7`
- New detail:
  - `z_score = 1.5275252316519468`
  - `hits = 7 / 7`
- Debug script:
  - Resolved params: `4 / 0.75`
  - No matching prompt-level diagnostic report auto-discovered

### HumanEval/45

- Old detail:
  - `z_score = -1.386750490563073`
  - `hits = 4 / 13`
- New detail:
  - `z_score = 1.4411533842457842`
  - `hits = 12 / 13`
- Debug script:
  - Resolved params: `4 / 0.75`
  - No matching prompt-level diagnostic report auto-discovered

## Notes

- The archived watermark JSONL at `/root/autodl-tmp/WFCLLM/data/watermarked/humaneval_20260318_204554.jsonl` predates metadata persistence, so prompt-level spot checks resolved `4 / 0.75` from aligned config defaults rather than embedded record metadata.
- The available prompt-level diagnostic report did not contain `HumanEval/128`, `HumanEval/23`, or `HumanEval/45`, so the debug CLI correctly downgraded to parameter-only output for those prompts.
- The repair clearly changed per-prompt hit behavior for known bad prompts (`HumanEval/23`, `HumanEval/45`), but did **not** improve the archived replay’s top-line `watermark_rate`; further investigation is still needed around thresholding and/or dataset/encoder state.
