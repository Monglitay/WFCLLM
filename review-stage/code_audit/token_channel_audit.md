# Token-Channel Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Repair Addendum

Updated: 2026-06-08T12:31:30Z

The token-channel code/test drift found by the audit was repaired for the covered paths:

- Dual-channel prompt propagation into lexical replay is fixed (`wfcllm/extract/detector.py:47-76`, `detector.py:80-107`, `detector.py:109-125`).
- Replay detection skips structure-masked/unscorable tokens and fails closed when feature prep fails (`wfcllm/extract/token_channel.py:83-103`).
- Token-channel model hidden-size/head compatibility is fixed by resolving a legal `nhead` (`wfcllm/watermark/token_channel/core/model.py:190-208`).
- Training cache split and selected-row loading preserve deterministic shuffled order (`wfcllm/watermark/token_channel/training/corpus_streaming.py:125-208`).
- Training workflow imports/export boundaries are patchable and streaming-compatible (`wfcllm/watermark/token_channel/training/workflow.py:16-34`, `workflow.py:292-325`, `workflow.py:430-474`).
- Generation now rejects short lexical-biased simple blocks and regenerates without lexical bias before accepting them (`wfcllm/watermark/orchestrator.py:202-211`, `orchestrator.py:374-404`, `orchestrator.py:791-840`, `orchestrator.py:879-905`).

Verification:

- Remote `pytest tests/watermark/ -v --tb=short`: 501 passed, 29 warnings.
- Remote `pytest tests/extract/ -v --tb=short`: 131 passed, 1 warning.

Existing lexical/joint artifacts remain invalidated if they were produced before prompt propagation and short-block regeneration fixes.

## Verdict

Token-channel evidence is not currently trustworthy as method-effect evidence. Artifact metadata/fingerprints are present and checked in code, but the dual-channel detector drops prompt context, current token-channel tests fail broadly, and evaluation summaries can be dominated by lexical/joint predictions while semantic scores remain weak.

## Config and Metadata

- `TokenChannelConfig.from_mapping` accepts both `channel_mode` and legacy `mode`, and both nested `joint` values and legacy `joint_*` scalar keys. This reduces config-key drift risk.
- Watermarked rows include token-channel metadata and fingerprints. Example manifest/probe values:
  - `model_sha256 = 9a67295a41b0edea5175ef7e5629d57d0977070325db7da33bf587d64ddfc85e`
  - `metadata_sha256 = a5e09e70f9a2703ee56d11be859bb28ec0684888cd5c553441b60be84321380a`
- The artifact files exist: `data/models/token-channel/model.pt`, `metadata.json`, `training_state.pt`, and `training_evidence.json`.
- `_assert_token_channel_pin_matches` checks persisted mode/context/repeat flags, artifact metadata, and artifact fingerprints at `wfcllm/extract/detector.py:186-221`.

## Confirmed Prompt Bug

Dual-channel semantic path drops `prompt` before lexical replay (`wfcllm/extract/detector.py:76`, `86`, `107` call `_attach_channel_results(result, code)` without prompt). Lexical-only path passes prompt. This can alter lexical replay positions, green hits, lexical z, joint score, and final `is_watermarked`.

## Test Failures

- `tests/watermark/token_channel/ -v --tb=short`: 38 failed, 159 passed.
- `tests/watermark/ -v --tb=short`: 57 failed, 443 passed; most failures are token-channel-related.
- Dominant failure: `TokenChannelModel(vocab_size=8, context_width=4, hidden_size=12)` raises `AssertionError: embed_dim must be divisible by num_heads` in PyTorch Transformer initialization. This indicates test fixture/API drift or an unvalidated model constructor constraint.
- Other failures show training corpus/workflow drift: generator has no `len()`, batch inference mock not called, workflow no longer exposes `save_training_cache` as tests expect.
- Extract token-channel failures show replay detector no longer skips/fails closed as expected in structure-masked or AST-feature failure cases.

## Summary Contamination Risk

`humaneval_full_cap9_summary.json` reports:

- `mean_z_score = -0.1188` and `mean_p_value = 0.5017` for semantic evidence.
- `mean_lexical_z_score = 8.1821`, `mean_joint_score = 5.7016`, and `joint_prediction_rate = watermark_rate = 0.84397`.

This means the headline watermark rate is driven by joint/lexical evidence. Given the prompt bug and failing token-channel tests, that headline rate is not reliable evidence.

## Current Result Validity

- Semantic-only fields need eval-mode regeneration.
- Lexical/joint fields need prompt propagation and token-channel test drift resolved.
- Token-channel artifact compatibility checks are present, but the surrounding runtime/tests are not green; do not use lexical or joint metrics as final claims yet.
