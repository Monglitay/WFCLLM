# Reproduction Commands and Audit Log

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Remote Setup

```bash
ssh -F /dev/null -p 19953 root@connect.bjb1.seetacloud.com
cd ~/autodl-tmp/WFCLLM
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## Post-Audit Repair Commands

Updated: 2026-06-08T12:31:30Z

The following commands/actions were run after user approval to repair audit findings.

### Local Focused Reproduction and Verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_lexical_only_short_block_is_regenerated_without_token_channel_bias \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_short_block_with_early_bias_is_regenerated_without_token_channel_bias \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_retry_success_rechecks_short_retry_block_before_accepting_it \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_lexical_only_short_block_regeneration_preserves_compound_for_outer_loop \
  -vv --tb=long
```

Initial result before the short-block fix: 4 failed. This was the RED reproduction.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_lexical_only_short_block_is_regenerated_without_token_channel_bias \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_short_block_with_early_bias_is_regenerated_without_token_channel_bias \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_retry_success_rechecks_short_retry_block_before_accepting_it \
  tests/watermark/test_generator.py::TestTokenChannelGeneration::test_lexical_only_short_block_regeneration_preserves_compound_for_outer_loop \
  -v --tb=short
```

Post-fix result: 4 passed.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_context.py::TestForwardAndSample::test_last_block_checkpoint_restores_final_biased_logits_from_real_bias_path \
  tests/watermark/test_context.py::TestForwardAndSample::test_block_start_checkpoint_logits_are_not_overwritten_by_later_tokens_in_block \
  tests/watermark/test_generator.py::TestTokenChannelGeneration \
  tests/watermark/test_pipeline.py::TestWatermarkPipelineRun::test_run_creates_jsonl \
  tests/watermark/test_pipeline.py::TestWatermarkPipelineRun::test_run_serializes_fixed_metadata_when_adaptive_config_is_enabled_but_runtime_is_not \
  -v --tb=short
```

Result: 21 passed.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -v --tb=short
```

Result: 501 passed, 29 warnings.

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Result: passed.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/integration/test_orchestration.py::test_phases_order \
  tests/integration/test_orchestration.py::test_all_phases_includes_build_entropy_profile \
  tests/integration/test_orchestration.py::test_run_state_manager_tracks_build_entropy_profile \
  -v --tb=short
```

Initial result after test edit but before import fix: 1 failed with `NameError: OPTIONAL_PHASES is not defined`.
Final result after import fix: 3 passed.

### Remote Sync Commands

```bash
scp -F /dev/null -P 19953 \
  /home/monglitay/PycharmProjects/WFCLLM/wfcllm/watermark/orchestrator.py \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/WFCLLM/wfcllm/watermark/orchestrator.py
```

```bash
scp -F /dev/null -P 19953 \
  /home/monglitay/PycharmProjects/WFCLLM/tests/integration/test_orchestration.py \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/WFCLLM/tests/integration/test_orchestration.py
```

Other repaired files had already been synchronized before this addendum:

- `wfcllm/watermark/verifier.py`
- `wfcllm/extract/detector.py`
- `wfcllm/extract/token_channel.py`
- `wfcllm/extract/pipeline.py`
- `wfcllm/watermark/token_channel/core/model.py`
- `wfcllm/watermark/token_channel/training/corpus_streaming.py`
- `wfcllm/watermark/token_channel/training/workflow.py`
- corresponding focused tests under `tests/extract/`, `tests/watermark/`, and `tests/watermark/token_channel/`

### Latest Remote Verification Commands

```bash
cd ~/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/ -v --tb=short
```

Result: 501 passed, 29 warnings.

```bash
cd ~/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/extract/ -v --tb=short
```

Result: 131 passed, 1 warning.

```bash
cd ~/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/common/ tests/datasets/ tests/lang/ tests/evaluation/ tests/integration/test_cli_resolution.py tests/integration/test_orchestration.py -v --tb=short
```

First run result before stale phase-order test repair: 1 failed, 227 passed, 4 warnings.
Final run result after test repair: 228 passed, 4 warnings.

```bash
cd ~/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/encoder/ tests/ablation/ -v --tb=short
```

Result: 86 passed, 14 warnings.

```bash
cd ~/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Result: passed.

## Preflight Commands Run

```bash
git status --short
/root/miniconda3/bin/conda run -n WFCLLM python run.py --status
/root/miniconda3/bin/conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Preflight result: status and compileall succeeded. Git status had unrelated/untracked local files including `review-stage/`; none were reverted.

## Environment Commands Run

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   /root/miniconda3/bin/conda run -n WFCLLM python -c "...environment/version probe..."
```

Environment summary:

- Shell transport used SSH into `/root/autodl-tmp/WFCLLM`; project commands were run through `/root/miniconda3/bin/conda run -n WFCLLM` or the env Python `/root/miniconda3/envs/WFCLLM/bin/python`.
- Explicit offline variables were used for probes/tests: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`.
- Python: `/root/miniconda3/envs/WFCLLM/bin/python`, 3.11.14.
- torch: 2.10.0+cu128, CUDA available, CUDA 12.8, GPU `NVIDIA GeForce RTX 5090`.
- transformers 4.46.3, datasets 4.6.1, numpy 2.4.2, scipy 1.17.1, bitsandbytes 0.49.2, peft 0.13.2, huggingface_hub 0.36.2, pytest 9.0.2.
- `sklearn` is not installed. That is acceptable for current production metrics because AUROC/FPR utilities are implemented manually in `wfcllm/evaluation/code_execution.py:99-135` and `requirements.txt` does not list scikit-learn.
- tree-sitter packages are installed at versions pinned by `requirements.txt`; import worked in project tests/probes. Their `__version__` attribute is not exposed by the modules.


## Documents and Configs Read

- `CLAUDE.md`
- `AGENTS.md`
- `README.md`
- `requirements.txt`
- `configs/base_config.json`
- all `configs/*.json` inventory
- `configs/ablation/example_dual_channel.yaml`
- `data/run_state.json`
- relevant `docs/design/*` and `docs/plans/*` headings for pipeline/config/offline/LSH/token-channel design

## Code Areas Read

- CLI/config/orchestration: `run.py`, `wfcllm/cli/*`, `wfcllm/orchestration/*`
- shared/data/lang: `wfcllm/common/*`, `wfcllm/datasets/*`, `wfcllm/lang/*`, `wfcllm/lang/python/*`
- encoder: `wfcllm/encoder/config.py`, `model.py`, `dataset.py`, `train.py`, `evaluate.py`
- watermark: `wfcllm/watermark/config.py`, `orchestrator.py`, `pipeline.py`, `verifier.py`, `lsh_space.py`, `keying.py`, `semantic_channel.py`, `retry_loop.py`, `cascade.py`, `context.py`, `interceptor.py`, `adaptive_gamma/*`, `token_channel/*`
- extract: `wfcllm/extract/config.py`, `detector.py`, `scorer.py`, `hypothesis.py`, `dp_selector.py`, `alignment.py`, `pipeline.py`, `token_channel.py`, `calibration/*`
- evaluation/ablation/scripts/tools: `wfcllm/evaluation/*`, `wfcllm/ablation/*`, `scripts/evaluate.py`, `scripts/build_entropy_profile.py`, `scripts/calibrate_threshold.py`, `scripts/generate_negative_corpus.py`, `scripts/run_ablation.py`, `scripts/evaluate_dual_channel.py`, relevant `tools/*`
- Tests under the requested groups were run/read through pytest logs and targeted source inspection.

## Probe Commands Run

Main completed probe:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   /root/miniconda3/envs/WFCLLM/bin/python review-stage/code_audit/probes/semantic_detector_probe.py
```

Probe output: `review-stage/code_audit/probes/semantic_detector_probe_results.json`.

Selected watermarked sample IDs:

```text
HumanEval/55, HumanEval/56, HumanEval/57, HumanEval/58, HumanEval/59, HumanEval/60, HumanEval/61, HumanEval/62, HumanEval/63, HumanEval/64
```

Selected negative sample IDs:

```text
HumanEval/0, HumanEval/1, HumanEval/2, HumanEval/3, HumanEval/4, HumanEval/5, HumanEval/6, HumanEval/7, HumanEval/8, HumanEval/9
```

Artifacts read by probe:

- `data/watermarked/humaneval_20260523_130234.jsonl`
- `data/negative_corpus.jsonl`
- `data/results/humaneval_full_cap9_details.jsonl`
- `data/results/humaneval_full_cap9_summary.json`
- `data/models/encoder/best_model.pt`
- `data/models/codet5-base`
- `data/run_state.json`

Probe attempts intentionally stopped:

- Full dual-channel 10+10 repeated 5 probe was stopped because lexical replay was CPU-bound and exceeded practical audit runtime.
- CPU encoder repeated-10 probe was stopped because it was infeasible on the remote CPU path.

## Artifact Manifest Commands Run

A Python manifest generator was run through SSH stdin. Outputs:

- `review-stage/code_audit/ARTIFACT_MANIFEST.json`
- `review-stage/code_audit/ARTIFACT_MANIFEST.md`

Manifest coverage includes configs, run state, models, checkpoints, watermarked outputs, results, negative corpora, diagnostics, calibration, and eval output paths. It hashes large model/checkpoint files.

## Test Commands Run

All with offline env:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/common/ -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/datasets/ -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/lang/ -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/extract/ -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/ -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/token_channel/ -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/evaluation/ -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v --tb=short
```

Raw logs: `review-stage/code_audit/test_logs/*.log` and `review-stage/code_audit/TEST_RESULTS.raw.json`.

Additional local pre-commit verification after syncing repairs to `/home/monglitay/PycharmProjects/WFCLLM`:

```bash
git diff --check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ tests/watermark/ tests/integration/test_orchestration.py -v --tb=short
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Results: `git diff --check` passed; pytest reported 667 passed and 30 warnings; compileall passed.

## Commands Intentionally Not Run

- No full pytest suite: targeted tests already found multiple local failures, and user requested no full pytest unless clearly necessary.
- No generation run.
- No training run.
- No artifact schema migration.
- No block-contract schema migration.
- No config or data artifact modification.
- No network/Hugging Face Hub access.
