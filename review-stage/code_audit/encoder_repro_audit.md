# Encoder Checkpoint, Tokenizer, and Reproducibility Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Repair Addendum

Updated: 2026-06-08T12:31:30Z

The verifier-side eval-mode bug was fixed after user approval. `ProjectionVerifier.__init__` now calls `encoder.eval()` when available (`wfcllm/watermark/verifier.py:25-32`). This fixes the shared verifier path used by watermark generation and extraction.

Verification:

- Remote `pytest tests/watermark/ -v --tb=short`: 501 passed, 29 warnings.
- Remote `pytest tests/extract/ -v --tb=short`: 131 passed, 1 warning.
- Remote `pytest tests/encoder/ tests/ablation/ -v --tb=short`: 86 passed, 14 warnings.

Historical persisted semantic details remain invalidated unless recomputed, because old details may have been produced before eval-mode enforcement and are also entangled with artifact pairing drift.

## Verdict

The encoder checkpoint and local model load successfully, but production inference paths do not force eval mode. This is a confirmed detector/generation nondeterminism bug. The probe showed train-mode embeddings and semantic hit vectors change substantially, while eval mode is stable. Existing persisted semantic details do not exactly reproduce for several sampled IDs.

## Artifact State

- Local CodeT5 model directory exists: `data/models/codet5-base`. The manifest notes `tokenizer.json` is missing, but `AutoTokenizer.from_pretrained('data/models/codet5-base')` loads from `vocab.json`, `merges.txt`, and tokenizer config.
- Best encoder checkpoint: `data/models/encoder/best_model.pt`, sha256 `6e340d66edcb8684504ae87e5d55425d92114ac00d16ac247198ea55abd0e3f5a`, size 222,067,430 bytes.
- Run-state checkpoint: `data/checkpoints/encoder/encoder_epoch9.pt`, sha256 `02cd9feaf288eb86253e33995f01407fc58e8388b90e1d2c8d2f970ed85a97dc`.
- Probe-loaded checkpoint top keys: `best_metric`, `config`, `epoch`, `model_state_dict`; checkpoint epoch 10; `strict=False` missing/unexpected key lists were empty.
- Checkpoint config matches the expected current encoder defaults: model `data/models/codet5-base`, embed_dim 128, LoRA enabled r=16 alpha=32 dropout=0.1, bf16 enabled, epochs 10.

## Eval-Mode Bug Evidence

Source:

- `wfcllm/cli/runners.py:255-273` loads the encoder for watermark and moves it to `encoder_device`, but does not call `encoder.eval()`.
- `wfcllm/cli/runners.py:462-481` loads the encoder for extract and moves it to device, but does not call `encoder.eval()`.
- `wfcllm/watermark/verifier.py:31-57` wraps `verify()` in `torch.no_grad()` but never calls eval mode; `no_grad` does not disable dropout.
- `wfcllm/extract/detector.py:36-42` constructs `ProjectionVerifier` around the provided encoder without changing mode.

Probe:

- Embedding probe on CUDA, 3 short snippets plus 3 generated snippets, repeated 10 times.
- Train mode: all 6 samples had 10 unique embedding hashes; max L2 drift ranged up to about 1.18 and max cosine-distance drift up to about 0.70.
- Eval mode: all 6 samples had 1 unique embedding hash; L2 drift was 0.0 for all samples, with only tiny floating cosine artifacts around `-2.38e-07` in two cases.
- Semantic detector probe, 10 watermarked + 10 negative, repeated 5:
  - eval/watermarked: 0 of 10 unstable z, hit vector, or prediction.
  - eval/negative: 0 of 10 unstable z, hit vector, or prediction.
  - train/watermarked: 10 of 10 unstable z and hit vectors; mean z range 1.999, max z range 3.546.
  - train/negative: 9 of 10 unstable z and hit vectors; mean z range 1.524, max z range 2.309.

## Persisted Detail Reproduction

Eval-mode recomputation was stable within the current process but did not reproduce several persisted details exactly:

- `HumanEval/55`: persisted z -4.5771 / hits 6; eval recompute z -4.2531 / hits 7.
- `HumanEval/57`: persisted z -1.3628 / hits 8; eval recompute z -2.2014 / hits 6.
- `HumanEval/61`: persisted z -1.5316 / hits 8; eval recompute z -1.9615 / hits 7.
- Several negative samples also differed materially.

This is consistent with previous extraction being run in train mode and/or artifact pairing drift. It invalidates those persisted semantic z values as reproducible evidence until regenerated under eval mode with pinned provenance.

## Not Run

- CPU repeated encoder probe was started but stopped as infeasible in the remote session; it was CPU-bound and did not write a completed result.
- No training was run.
