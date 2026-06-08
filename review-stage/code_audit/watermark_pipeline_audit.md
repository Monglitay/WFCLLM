# Watermark Generation Pipeline Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Repair Addendum

Updated: 2026-06-08T12:31:30Z

Two generation-path issues were fixed after user approval:

- `ProjectionVerifier` now forces eval mode when constructed, reducing generation-time pass/fail nondeterminism from dropout-bearing encoders (`wfcllm/watermark/verifier.py:25-32`).
- Token-channel biased short blocks are no longer accepted directly. Lexical-only generation rolls back and regenerates without lexical bias; dual-channel generation also rechecks successful retry blocks and reruns without lexical bias when the retry block is still too short (`wfcllm/watermark/orchestrator.py:202-211`, `orchestrator.py:374-404`, `orchestrator.py:791-840`, `orchestrator.py:879-905`).

Verification:

- Initial local RED reproduction for the four short-block tests: 4 failed.
- Post-fix local result for the same tests: 4 passed.
- Remote `pytest tests/watermark/ -v --tb=short`: 501 passed, 29 warnings.
- Remote compileall: passed.

Existing generated artifacts are not retroactively corrected. Any lexical/dual-channel generation result produced before this fix must be regenerated before use as method-effect evidence.

## Verdict

No generation was run. Code/artifact review found that public artifact schema is mostly stable and `secret_key` is not present in sampled public `watermark_params`, but watermark generation uses the same non-eval encoder path as extraction. Therefore existing watermark pass/fail block outcomes may be nondeterministic if produced through the current runner.

## Source Findings

- `WatermarkGenerator` is built from `WatermarkConfig` and config-resolved adaptive/token-channel sections in the runner.
- `wfcllm/cli/runners.py:247-273` loads `SemanticEncoder`, loads checkpoint weights, moves to device, and creates tokenizer; it does not call `encoder.eval()`.
- `ProjectionVerifier.verify()` in `wfcllm/watermark/verifier.py:31-57` uses `torch.no_grad()` but not eval mode.
- `LSHSpace` is deterministic: hyperplanes are HMAC-derived from `secret_key` (`wfcllm/watermark/lsh_space.py:47-54`).
- `WatermarkKeying` is deterministic: valid region set `G` is HMAC-derived from parent node type and optional ordinal (`wfcllm/watermark/keying.py:57-77`).
- Public watermark params are constructed in `wfcllm/watermark/pipeline.py:140-160` and include LSH/adaptive/token public metadata, not `secret_key`.

## Artifact Schema and Public Metadata

The sampled watermarked rows include stable fields: `id`, `dataset`, `prompt`, `generated_code`, `blocks`, `adaptive_mode`, `profile_id`, `alignment_summary`, `total_blocks`, `embedded_blocks`, `failed_blocks`, `fallback_blocks`, `embed_rate`, `diagnostics_ledger_rows`, `watermark_params`, and optional `token_channel`.

Probe scanned the first 50 rows of the run-state watermarked artifact and did not find the literal `secret_key` in public rows. Token-channel metadata and fingerprints are present in watermarked rows, including model and metadata sha256 values.

## Suspicious Artifact Signals

- `data/watermarked/humaneval_20260523_130234.jsonl` has only 55 rows and starts at `HumanEval/55`. That can be a resume/limit artifact, but it is not paired with the run-state extract summary.
- Some rows show `alignment_summary.generator_total_blocks` much smaller than `final_block_count` and `block_count_matches_total_blocks=false`, for example `HumanEval/55` has final block count 39 but generator total blocks 5. This may be expected because final AST contains additional module statements/comments, but it should be inspected before using embed-rate as a clean generation statistic.
- Repeated duplicate block hashes appear in generated code with many repeated print statements. This can inflate simple block counts and interact with ordinal/hash contract logic.

## Resume Risk

Watermark resume has row/ledger checks, but it does not pin full config, encoder checkpoint hash, tokenizer identity, LM model fingerprint, or calibration identity. A resumed output can be operationally valid while still being poor provenance for a paper-style result.
