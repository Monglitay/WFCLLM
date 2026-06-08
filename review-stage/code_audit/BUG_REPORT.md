# WFCLLM Bug Report

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Audit Repair Status

Updated: 2026-06-08T12:31:30Z

User approved repair after the audit. The original report below is preserved as the audit snapshot; this section records the current fixed/open status.

### Fixed in Code

- **P0 Encoder train-mode inference:** fixed at `wfcllm/watermark/verifier.py:25-32` by forcing eval mode in `ProjectionVerifier`. Verified by `tests/watermark/test_verifier.py`, full remote `tests/watermark/`, and full remote `tests/extract/`.
- **P1 dual-channel prompt drop:** fixed at `wfcllm/extract/detector.py:47-76`, `detector.py:80-107`, and `detector.py:109-125` by forwarding `prompt` into all semantic/dual `_attach_channel_results` paths. Verified by remote `tests/extract/`.
- **P1 token-channel runtime/test drift:** fixed for the audited failures:
  - legal attention-head resolution in `wfcllm/watermark/token_channel/core/model.py:190-208`;
  - tokenizer compatibility rechecked in token-channel artifact compatibility;
  - replay detector skips/fails closed on missing or structure-masked features at `wfcllm/extract/token_channel.py:83-103`;
  - streaming split/order and row loading fixed at `wfcllm/watermark/token_channel/training/corpus_streaming.py:125-208`;
  - workflow streaming/export monkeypatch boundaries fixed at `wfcllm/watermark/token_channel/training/workflow.py:16-34`, `workflow.py:292-325`, `workflow.py:430-474`.
- **P1 token-channel short-block acceptance bug:** fixed at `wfcllm/watermark/orchestrator.py:202-211`, `orchestrator.py:374-404`, `orchestrator.py:791-840`, and `orchestrator.py:879-905`. Short lexical-biased simple blocks are regenerated without lexical bias before acceptance.
- **P3 stale phase-order test:** fixed at `tests/integration/test_orchestration.py:9` and `test_orchestration.py:324-332`.

### Still Open

- **P0 run-state artifact pairing mismatch:** not fixed. Requires provenance validation/schema work or explicit artifact regeneration. Existing status bundle remains invalidated.
- **P1 calibration/negative provenance drift:** not fixed. Requires calibration corpus path/hash recording and coherent run-state linkage.
- **P2 encoder JSON config section ignored by `run_encoder()`:** not fixed in this repair pass.
- **P2 direct calibration raw AutoModel path:** not fixed in this repair pass.
- **P2 evaluation auto harness environment drift:** not fixed in this repair pass.

### Post-Repair Verification Commands

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/ -v --tb=short
# 501 passed, 29 warnings

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/extract/ -v --tb=short
# 131 passed, 1 warning

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/common/ tests/datasets/ tests/lang/ tests/evaluation/ tests/integration/test_cli_resolution.py tests/integration/test_orchestration.py -v --tb=short
# 228 passed, 4 warnings

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/encoder/ tests/ablation/ -v --tb=short
# 86 passed, 14 warnings

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
# passed
```

### Current Result Validity

The repairs fix code paths for future runs and re-evaluations. They do **not** retroactively validate old artifacts. Any current result that depends on old semantic details, old lexical/dual generation, old joint summaries, or run-state-combined artifact tuples remains unsuitable as final method-effect evidence until regenerated or recomputed with coherent provenance.

## P0: Encoder Inference Runs in Train Mode

Severity: P0

Reproduction command:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   /root/miniconda3/envs/WFCLLM/bin/python review-stage/code_audit/probes/semantic_detector_probe.py
```

Expected: repeated inference/detection with the same checkpoint and input produces identical embeddings, LSH hit vectors, and z-scores.

Actual: train mode changed embedding hashes on all 6 embedding samples and changed semantic z/hit-vector hashes on 19 of 20 detector samples. Eval mode was stable on all probed samples.

Root cause: watermark and extract runners load/move the encoder but never call `encoder.eval()` (`wfcllm/cli/runners.py:255-273`, `runners.py:462-481`). `ProjectionVerifier.verify()` uses `torch.no_grad()` but does not force eval mode (`wfcllm/watermark/verifier.py:31-57`).

Affected artifacts/results: semantic hit vectors, z-scores, p-values, persisted details, calibrated thresholds if computed through the same train-mode detector, and generation-time block pass/fail decisions.

Invalidates current results: yes. Existing semantic detector outputs are not reproducible evidence.

Minimal failing test proposal:

- Add a test that creates a `SemanticEncoder` or dropout-bearing fake encoder in train mode, constructs the real runner/verifier path, and asserts the runner/verifier sets or preserves eval mode before scoring.
- Add an integration-style detector determinism test with a dropout fake encoder: repeated `detect()` calls must have one unique hit vector.

Proposed fix scope: call `encoder.eval()` immediately after checkpoint loading and device transfer in watermark/extract/calibration paths; consider `ProjectionVerifier.__init__` either asserting eval mode or explicitly calling eval with a documented contract.

Forbidden shortcuts: do not seed randomness as a substitute for eval mode; do not suppress nondeterminism by rounding z; do not alter artifact schema.

## P0: Run-State Watermark/Extract Artifact Pairing Mismatch

Severity: P0

Reproduction command:

```bash
/root/miniconda3/bin/conda run -n WFCLLM python run.py --status
/root/miniconda3/bin/conda run -n WFCLLM python -m json.tool data/results/humaneval_full_cap9_summary.json
```

Expected: completed extract details/summary in `data/run_state.json` were produced from the completed watermark output recorded in the same run state.

Actual: run state watermark output is `data/watermarked/humaneval_20260523_130234.jsonl`, but extract summary meta says input was `data/watermarked/humaneval_full_cap9.jsonl`.

Root cause: run-state schema stores phase completion paths independently without validating cross-phase artifact lineage, input stem, checksum, or config/checkpoint provenance.

Affected artifacts/results: `data/run_state.json`, `data/results/humaneval_full_cap9_details.jsonl`, `data/results/humaneval_full_cap9_summary.json`, any status-based report combining those files.

Invalidates current results: yes. Current status is not a single coherent experiment.

Minimal failing test proposal:

- Create a temporary run state with watermark output `a.jsonl` and extract details `b_details.jsonl`; status/provenance validation should flag mismatch.
- Add extract completion metadata recording input file hash and require it to match run-state watermark output when extract is run as a pipeline phase without explicit input.

Proposed fix scope: add provenance metadata and validation around `RunStateManager.mark_done` for extract; include input file path/hash, calibration corpus path/hash, encoder checkpoint hash, token-channel artifact hash.

Forbidden shortcuts: do not rename files or edit run_state manually to hide mismatch; do not treat matching dataset names as sufficient lineage.

## P1: Dual-Channel Detection Drops Prompt Before Lexical Replay

Severity: P1

Reproduction command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   /root/miniconda3/bin/conda run -n WFCLLM pytest tests/extract/test_pipeline.py::TestExtractPipelineStatistics::test_run_without_metadata_keeps_detail_rows_legacy_shaped -v --tb=short
```

Expected: when `ExtractPipeline` passes prompt to `WatermarkDetector.detect()`, lexical replay in dual-channel mode receives that prompt.

Actual: semantic/dual path calls `_attach_channel_results(result, code)` without prompt at `wfcllm/extract/detector.py:76`, `86`, and `107`. `_attach_channel_results` defaults prompt to empty string.

Root cause: prompt parameter was added to the public `detect()` path but not forwarded through semantic/dual branches.

Affected artifacts/results: lexical z, green fraction, joint score, joint prediction, dual-mode `is_watermarked`, and summary watermark_rate when driven by joint prediction.

Invalidates current results: yes for lexical/joint/dual summaries; semantic z is affected separately by the eval-mode bug.

Minimal failing test proposal:

- Mock `ReplayTokenChannelDetector.detect` and assert dual-channel `WatermarkDetector.detect(code, prompt="p")` calls it with `prompt="p"` for normal semantic path, no-block path, and no-simple-block path.

Proposed fix scope: pass `prompt=prompt` into all `_attach_channel_results` calls in `WatermarkDetector.detect()`.

Forbidden shortcuts: do not disable prompt use in lexical replay; do not reinterpret current promptless results as prompt-aware.

## P1: Token-Channel Runtime/Test Drift

Severity: P1

Reproduction command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/token_channel/ -v --tb=short
```

Expected: token-channel model/runtime/training workflow tests pass under the pinned environment.

Actual: 38 failed, 159 passed. Dominant failure is `AssertionError: embed_dim must be divisible by num_heads` when tests instantiate `TokenChannelModel(... hidden_size=12)`. Other failures indicate training corpus and workflow API drift.

Root cause: current token-channel production code and tests disagree about constructor defaults/API and training workflow functions.

Affected artifacts/results: token-channel training reproducibility, lexical runtime confidence, dual/joint metrics.

Invalidates current results: yes for lexical/joint conclusions; token-channel artifact files exist but the runtime/test surface is not green.

Minimal failing test proposal:

- Preserve the current failing token-channel tests and decide whether tests should use hidden sizes divisible by current `num_heads` or the model should adapt `num_heads`/validate with a project-level error.
- Add a prompt-aware dual detector test after prompt bug is fixed.

Proposed fix scope: reconcile model constructor defaults, test fixtures, training cache API, and AST feature fail-closed behavior.

Forbidden shortcuts: do not delete failing tests; do not bypass metadata compatibility checks.

## P1: Calibration/Negative Provenance Drift

Severity: P1

Reproduction command:

```bash
/root/miniconda3/envs/WFCLLM/bin/python -m json.tool data/run_state.json
/root/miniconda3/envs/WFCLLM/bin/python -m json.tool data/results/humaneval_full_cap9_summary.json
```

Expected: run-state negative corpus output and extract summary calibration source are the same or explicitly documented as different.

Actual: run state records `data/negative_corpus_llm.jsonl`; summary used `data/negative_corpus.jsonl`.

Root cause: run-state completion paths do not bind extract calibration source to a specific negative-corpus phase artifact.

Affected artifacts/results: calibrated threshold interpretation and any claim that generate-negative/extract were a coherent pipeline run.

Invalidates current results: yes for run-state-level calibrated conclusions; the summary file remains an internally labeled historical artifact.

Minimal failing test proposal: after `generate-negative`, run extract with config calibration corpus differing from generated output and assert a warning/provenance note is recorded.

Proposed fix scope: record calibration corpus path/hash in extract run state and summary; optionally require explicit override to use a different negative corpus.

Forbidden shortcuts: do not overwrite negative corpus files to match paths; do not rely on ID count alone.

## P2: Encoder JSON Config Section Ignored by `run_encoder()`

Severity: P2

Reproduction command:

```bash
/root/miniconda3/bin/conda run -n WFCLLM python run.py --phase encoder --config configs/base_config.json --status
```

Expected: `configs/*.json` `encoder` section is merged into `EncoderConfig` unless overridden by CLI.

Actual: `wfcllm/cli/runners.py:113-194` creates defaults and applies direct CLI args only.

Root cause: config-system refactor did not complete the encoder runner path.

Affected artifacts/results: any encoder experiment depending on JSON-only encoder overrides.

Invalidates current results: not by itself for current checkpoint because checkpoint config matches defaults closely; can invalidate non-default runs.

Minimal failing test proposal: config with `encoder.embed_dim = 64` and no CLI override should call encoder train/eval with `embed_dim=64`.

Proposed fix scope: add resolver for encoder section and unit/integration tests.

Forbidden shortcuts: do not duplicate override logic outside config resolver.

## P2: Direct Calibration Script Uses Raw AutoModel

Severity: P2

Reproduction command:

```bash
/root/miniconda3/bin/conda run -n WFCLLM python scripts/calibrate_threshold.py --help
```

Expected: calibration uses the same `SemanticEncoder` projection checkpoint as extraction.

Actual: `wfcllm/extract/calibration/runner.py:47-55` loads `AutoModel.from_pretrained(model)` and passes it into `ProjectionVerifier`.

Root cause: stale script runner from pre-refactor calibration path.

Affected artifacts/results: any thresholds produced by `scripts/calibrate_threshold.py` direct path.

Invalidates current results: suspected for direct-script calibration artifacts; current `humaneval_full_cap9` summary appears produced by inline extract calibration path, not necessarily this script.

Minimal failing test proposal: direct calibration with local CodeT5 should fail currently or produce wrong model output type; test should require loading `SemanticEncoder` checkpoint.

Proposed fix scope: update script runner to use `SemanticEncoder`, checkpoint path, and eval mode.

Forbidden shortcuts: do not calibrate with unfine-tuned base model unless explicitly labeled as baseline.

## P2: Evaluation Auto Harness Environment Drift

Severity: P2

Reproduction command:

```bash
/root/miniconda3/bin/conda run -n WFCLLM pytest tests/evaluation/ -v --tb=short
```

Expected: auto-generation harness launches project phases in the required `WFCLLM` conda env with all offline variables.

Actual: source uses bare `python run.py` and sets only `HF_HUB_OFFLINE`.

Root cause: evaluation helpers were written as generic subprocess calls and do not encode the project environment contract.

Affected artifacts/results: auto-generated benchmark/dual evaluation outputs when invoked from base or a drifted shell.

Invalidates current results: only if those outputs were produced through the auto harness from a drifted env; not proven for current summary.

Minimal failing test proposal: assert auto-generated commands include the configured Python/conda executable or allow an explicit command prefix and all offline variables.

Proposed fix scope: make command runner configurable and default to current interpreter or documented conda command with all offline env variables.

Forbidden shortcuts: do not assume shell activation state.
