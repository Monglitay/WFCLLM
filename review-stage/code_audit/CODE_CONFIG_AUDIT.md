# WFCLLM Code / Config / Artifact Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Audit Repair Addendum

Updated: 2026-06-08T12:31:30Z

After user approval to repair, targeted production/test fixes were applied. This addendum supersedes the original scope sentence above for the repair phase only. No configs, data artifacts, checkpoint/model files, artifact schema, or block-contract schema were modified.

### Repair Verdict

The confirmed code/test bugs found in the audit and accepted into the repair scope have been fixed and verified in the remote `WFCLLM` conda environment. However, historical experiment artifacts remain invalidated as method-effect evidence until regenerated or re-evaluated with coherent provenance.

Current poor performance still must not be attributed primarily to method limitations, because the old artifacts were produced before the reproducibility/provenance fixes and still include known artifact pairing/calibration-source drift.

### Fixed Confirmed Bugs

- Fixed encoder eval-mode enforcement at verifier construction: `wfcllm/watermark/verifier.py:25-32`.
- Fixed dual-channel prompt propagation into lexical replay: `wfcllm/extract/detector.py:47-76`, `detector.py:80-107`, `detector.py:109-125`.
- Fixed lexical replay fail-closed behavior for AST feature/preparation failure and structure-masked spans: `wfcllm/extract/token_channel.py:83-103`.
- Fixed token-channel model hidden-size/head compatibility by resolving a legal attention-head count while validating bad `nhead`: `wfcllm/watermark/token_channel/core/model.py:190-208`.
- Fixed token-channel artifact compatibility to include tokenizer-name checks: `wfcllm/watermark/token_channel/core/model.py` compatibility path.
- Fixed streaming training split/order drift: `wfcllm/watermark/token_channel/training/corpus_streaming.py:125-168`, `corpus_streaming.py:171-208`.
- Fixed token-channel training workflow patchability and streaming artifact export boundaries: `wfcllm/watermark/token_channel/training/workflow.py:16-34`, `workflow.py:292-325`, `workflow.py:430-474`.
- Fixed generation-time short block guard for token-channel biased blocks. The generator now rejects short lexical-biased simple blocks and regenerates without lexical bias before accepting them: `wfcllm/watermark/orchestrator.py:202-211`, `orchestrator.py:374-404`, `orchestrator.py:791-840`, `orchestrator.py:879-905`.
- Fixed stale phase-order integration test to match the current phase source of truth: `tests/integration/test_orchestration.py:9`, `test_orchestration.py:324-332`.

### Post-Repair Verification

Remote commands were run under:

```bash
cd /root/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/root/miniconda3/bin/conda run -n WFCLLM ...
```

Verified:

- `pytest tests/watermark/ -v --tb=short`: 501 passed, 29 warnings.
- `pytest tests/extract/ -v --tb=short`: 131 passed, 1 warning.
- `pytest tests/common/ tests/datasets/ tests/lang/ tests/evaluation/ tests/integration/test_cli_resolution.py tests/integration/test_orchestration.py -v --tb=short`: 228 passed, 4 warnings.
- `pytest tests/encoder/ tests/ablation/ -v --tb=short`: 86 passed, 14 warnings.
- `python -m compileall wfcllm run.py scripts tools`: passed.

Local pre-sync checks also passed for the original four short-block failures, adjacent watermark targets, `tests/watermark/`, and compileall.

Local pre-commit checks on `/home/monglitay/PycharmProjects/WFCLLM` also passed:

- `git diff --check`: passed.
- `pytest tests/extract/ tests/watermark/ tests/integration/test_orchestration.py -v --tb=short`: 667 passed, 30 warnings.
- `python -m compileall wfcllm run.py scripts tools`: passed.

### Still Invalidated

These are not repaired by code changes alone:

- Existing watermarked/extract result pair recorded in `data/run_state.json` remains mismatched. Old status-level results are not a coherent single experiment.
- Existing lexical/dual-channel generated artifacts produced before the short-block lexical-bias guard are not suitable for method-effect claims.
- Existing semantic detector outputs produced before eval-mode enforcement should be treated as non-reproducible unless regenerated/recomputed.
- Existing calibration conclusions remain provenance-limited when negative corpus path/hash differs between run state and summary.

### Still Suspected / Not Fixed

- JSON `encoder` config section is still suspected to be ignored by `run_encoder()`.
- Direct calibration script path may still use a raw `AutoModel` instead of `SemanticEncoder`.
- Evaluation auto-generation helpers may still use environment-fragile subprocess commands.
- Run-state artifact lineage validation is still not implemented; this requires a deliberate provenance schema/design change.

### Safe Next Minimal Experiments

1. Re-run semantic-only extract on one explicit watermarked JSONL and its matched negative corpus with eval-mode code.
2. Re-run token-channel/dual extraction on newly generated or clearly provenance-matched artifacts only.
3. Regenerate a small watermarked sample after the short-block guard fix, then run extract and compare with old artifacts.
4. Do not use `data/run_state.json` as a claim bundle until watermark/extract/calibration provenance paths and hashes match.

## Executive Verdict

Current poor performance cannot be attributed to the method itself yet. The audit found confirmed code/config/artifact issues that are sufficient to invalidate the current experimental results as method-effect evidence:

- Detector and watermark verification use the encoder in train mode unless caller manually sets eval mode. Probe confirms train-mode nondeterminism changes embeddings, semantic z, and hit vectors.
- `data/run_state.json` combines a 55-row watermarked artifact with extract details/summary computed from a different 164-row artifact.
- Dual-channel detection drops prompt context before lexical replay in the semantic/dual path.
- Token-channel tests are failing broadly, and the current headline watermark rate is driven by lexical/joint evidence, not semantic z.
- Calibration/negative provenance is inconsistent between run state and summary metadata.

Therefore, the current artifacts are not suitable for concluding that WFCLLM’s current poor effect is mainly a method limitation. Reproducibility/provenance must be repaired first.

## Confirmed Bugs / Invalidating Issues

### P0: Encoder Inference Not Forced to Eval Mode

Evidence:

- `wfcllm/cli/runners.py:255-273` and `runners.py:462-481` load/move the encoder but do not call `encoder.eval()`.
- `wfcllm/watermark/verifier.py:31-57` uses `torch.no_grad()` but not eval mode.
- Probe: train mode changed z/hit vectors on 19 of 20 repeated semantic detections; eval mode was stable on all 20.

Impact: current semantic detector details and watermark verification outcomes may be nondeterministic. This alone invalidates persisted detector outputs until regenerated.

### P0: Run-State Artifact Pairing Mismatch

Evidence:

- Run state watermark: `data/watermarked/humaneval_20260523_130234.jsonl`, 55 rows, first ID `HumanEval/55`.
- Run state extract summary/details: `data/results/humaneval_full_cap9_*`; summary meta says input file was `data/watermarked/humaneval_full_cap9.jsonl`, 164 rows.
- Manifest global note: run-state watermark output stem `humaneval_20260523_130234` differs from extract details input stem `humaneval_full_cap9`.

Impact: the current `run.py --status` tuple is not a single experiment and must not be used for claims.

### P1: Dual-Channel Prompt Dropped

Evidence:

- `WatermarkDetector.detect()` accepts `prompt` at `wfcllm/extract/detector.py:47-52`.
- Lexical-only passes prompt at `detector.py:54-66`.
- Semantic/dual path calls `_attach_channel_results(result, code)` without prompt at `detector.py:76`, `86`, and `107`.

Impact: lexical/joint detection can be wrong; `is_watermarked` in dual mode uses joint prediction. This invalidates lexical/joint headline rates until fixed/regenerated.

### P1: Token-Channel Test/Runtime Drift

Evidence:

- `tests/watermark/token_channel/`: 38 failed, 159 passed.
- Dominant constructor failure: `embed_dim must be divisible by num_heads` for test fixture hidden_size 12.
- Extract token-channel tests also fail for structure masking/fail-closed behavior.

Impact: token-channel results are not credible as final evidence until tests/runtime behavior are reconciled.

### P1: Calibration and Run-State Negative Corpus Drift

Evidence:

- `configs/base_config.json` extract calibration corpus: `data/negative_corpus.jsonl`.
- Run-state generate-negative output: `data/negative_corpus_llm.jsonl`.
- Summary metadata used `data/negative_corpus.jsonl`.

Impact: the summary may be internally labeled, but the run-state pipeline is not coherent.

## Suspected Bugs / Risks

- `run_encoder()` ignores the JSON `encoder` config section (`wfcllm/cli/runners.py:113-194`). Current base config aligns with checkpoint metadata, but future or existing non-default experiment configs can silently drift.
- Direct calibration script loads `AutoModel` instead of `SemanticEncoder` checkpoint (`wfcllm/extract/calibration/runner.py:47-55`), likely stale/broken.
- Ablation runner builds incomplete args (`wfcllm/ablation/runner.py:119-134`), likely failing real phase runners or hiding config fallbacks.
- Evaluation auto-generation uses bare `python run.py` and incomplete offline env (`wfcllm/evaluation/dual_channel.py:287-354`, `wfcllm/evaluation/benchmark.py:195-237`).
- Parser can extract blocks from syntactically bad code, while token-channel tests expect syntax-invalid cases to be skipped/fail closed.
- Repeated identical block hashes in generated artifacts can make hash-keyed contract lookup ambiguous when duplicate statements occur.

## Ruled-Out Hypotheses

- Environment command path: audited commands used the `WFCLLM` conda environment, not base.
- CUDA/GPU visibility: torch sees CUDA and RTX 5090.
- Local dataset availability: HumanEval and MBPP load from cached local paths under offline mode.
- LSHSpace and WatermarkKeying determinism: both use deterministic HMAC-derived seeds.
- Semantic formula shape: expected hits/variance/z-score formulas are coherent in code.
- Manual AUROC/FPR helpers: evaluation tests pass and code implements pairwise AUROC without sklearn.

## Artifact Provenance Verdict

Artifact provenance is not acceptable for final claims. The manifest contains 362 files and flags these global issues:

- run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- base_config extract.calibration_corpus 'data/negative_corpus.jsonl' differs from run_state generate-negative output 'data/negative_corpus_llm.jsonl'

The manifest files are:

- `review-stage/code_audit/ARTIFACT_MANIFEST.json`
- `review-stage/code_audit/ARTIFACT_MANIFEST.md`

## Targeted Test Results

- Passed: `tests/common/` 67 passed; `tests/datasets/` 9 passed; `tests/lang/` 9 passed; `tests/evaluation/` 87 passed; `tests/integration/test_cli_resolution.py` 21 passed.
- Failed: `tests/extract/` 10 failed / 120 passed; `tests/watermark/` 57 failed / 443 passed; `tests/watermark/token_channel/` 38 failed / 159 passed; `tests/integration/test_orchestration.py` 1 failed / 34 passed.
- No full pytest was run.

Details are in `review-stage/code_audit/TEST_RESULTS.md` and logs under `review-stage/code_audit/test_logs/`.

## Safe Next Fixes Requiring Approval

Do not patch production in this audit turn. Recommended approval-gated next steps:

1. Add failing tests proving encoder eval mode is required in watermark/extract runners and `ProjectionVerifier`, then patch runner/verifier setup.
2. Add a failing dual-channel test that prompt reaches lexical replay in semantic/dual mode, then patch `_attach_channel_results` calls.
3. Add provenance validation/fingerprint checks for run-state extract/watermark pairing and calibration source.
4. Repair token-channel test/runtime drift before trusting lexical/joint summaries.
5. Rerun minimal eval-mode semantic-only extract on a single explicit watermarked input and matching negative corpus.

## Approval Checkpoint

Production changes are intentionally not made. Approval is required before modifying `wfcllm/`, tests, configs, data artifacts, artifact schema, or block-contract schema.
