# Extract Detector and Scorer Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Repair Addendum

Updated: 2026-06-08T12:31:30Z

The confirmed detector issues accepted into the repair scope were fixed:

- `ProjectionVerifier` now forces eval mode (`wfcllm/watermark/verifier.py:25-32`).
- `WatermarkDetector.detect()` now forwards `prompt` to all semantic/dual `_attach_channel_results` branches (`wfcllm/extract/detector.py:47-76`, `detector.py:80-107`, `detector.py:109-125`).
- Token-channel artifact loading now handles non-module mocks/devices safely in the detector path.
- `ExtractPipeline` no longer crashes on non-int/mock `min_blocks`.
- `ReplayTokenChannelDetector` fails closed or skips unscorable structure-masked spans instead of fabricating positive structure evidence (`wfcllm/extract/token_channel.py:83-103`).

Verification:

- Remote `pytest tests/extract/ -v --tb=short`: 131 passed, 1 warning.
- Remote `pytest tests/watermark/ -v --tb=short`: 501 passed, 29 warnings.

Old persisted details and summaries are still not retroactively valid. Recompute detector outputs with coherent input/provenance before using them as method-effect evidence.

## Verdict

This is the highest-risk component. There is a confirmed eval-mode nondeterminism bug, a confirmed dual-channel prompt propagation bug, failed extract/token-channel tests, and non-reproducible persisted semantic details. Existing detector outputs should not be treated as stable method-effect evidence until detector inference is forced to eval mode and artifacts are regenerated with coherent provenance.

## Semantic Detector Correctness

Formula review:

- Adaptive expected hits and variance use per-block `gamma_effective`: `wfcllm/extract/hypothesis.py:83-95`.
- Fixed expected hits and variance use `m * gamma` and `m * gamma * (1-gamma)`: `hypothesis.py:97-100`.
- Z score is `(observed - expected) / sqrt(variance)`, with zero-variance infinities/zero handled explicitly: `hypothesis.py:103-116`.
- P-value uses one-sided survival function: `hypothesis.py` imports `scipy.stats.norm` and uses it in tester/fusion paths.

These formulas are coherent. The bug is not the z-score formula; it is unstable embedding/scoring state and artifact provenance.

## Confirmed Nondeterminism

Source:

- Detector receives encoder without eval mode: `wfcllm/cli/runners.py:462-481`; `wfcllm/extract/detector.py:36-42`; `wfcllm/watermark/verifier.py:31-57`.

Probe:

- Eval mode: all 20 semantic-only repeated detections stable across 5 repeats.
- Train mode: 19 of 20 samples changed z; 19 of 20 changed hit-vector hash. Watermarked mean z range 1.999, max 3.546; negative mean z range 1.524, max 2.309.
- Existing persisted details differ from current eval-mode recomputation for multiple IDs.

This satisfies the user’s detector-nondeterminism invalidation criterion.

## Dual-Channel Prompt Propagation Bug

- Public detect signature accepts `prompt` at `wfcllm/extract/detector.py:47-52`.
- Lexical-only path passes prompt correctly: `detector.py:54-66`.
- Semantic/dual path drops prompt: empty/no-block returns call `_attach_channel_results(result, code)` at `detector.py:76` and `detector.py:86`; normal semantic path calls `_attach_channel_results(result, code)` at `detector.py:107`.
- `_attach_channel_results` has a `prompt` parameter and passes it to `_detect_lexical` at `detector.py:109-117`, but callers in the semantic/dual path never pass the user prompt.
- `ExtractPipeline` does pass prompt into `detect()` (`wfcllm/extract/pipeline.py:205-215`), so the bug is specifically inside `WatermarkDetector.detect()`.

Impact: lexical and joint results for dual-channel extraction can be wrong whenever lexical replay depends on prompt context. Semantic `z_score` is kept as semantic score in dual mode (`detector.py:123-126`), so this bug contaminates lexical/joint fields and final `is_watermarked`, not the raw semantic z directly.

## Min-Blocks and Summary Bias

- `ExtractPipeline.run()` skips records before detection if embedded `blocks` length is below `min_blocks`: `wfcllm/extract/pipeline.py:195-203`.
- This changes `scored_samples`, summary rates, and details row count. It is not merely a classification threshold.
- The current `humaneval_full_cap9` input has 164 rows, while details have 141 rows; this is consistent with pre-detection skipping/invalid handling.

## Test Failures

`tests/extract/ -v --tb=short`: 10 failed, 120 passed.

Representative failures:

- Mocked token-channel artifact path fails because `_get_token_channel_artifact()` unconditionally calls `.model.to(device)` on a mock object.
- Joint score test expected 3.5, actual 3.3955, reflecting changed Stouffer/support-factor semantics or stale test expectation.
- Lexical-only pipeline summary expected watermark_rate 0.0 but got 1.0.
- Several pipeline tests crashed on `len(item["blocks"]) < self._detector._config.min_blocks` when `_config.min_blocks` was a `MagicMock`.
- Replay detector failed expected structure masking/fail-closed behavior.

## What Is Ruled Out

- With `encoder.eval()` set manually in the probe, semantic-only detection was deterministic on sampled CUDA runs.
- LSH keying and hyperplane construction are deterministic by HMAC source review.

## Current Result Validity

- Persisted semantic z/details from current run-state artifacts are invalid for reproducibility claims.
- Dual/joint summary fields are invalid or at least suspected invalid because prompt is dropped and token-channel tests fail.
- A fresh eval-mode semantic-only recomputation with coherent input/details pairing could become trustworthy, but that is not the current artifact state.
