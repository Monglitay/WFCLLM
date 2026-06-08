# Config and CLI Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Repair Addendum

Updated: 2026-06-08T12:31:30Z

The stale `ALL_PHASES` test expectation was repaired. Current tests now assert the documented source of truth:

- `PHASES == ["encoder", "watermark", "extract"]`
- `OPTIONAL_PHASES == ["pretrain", "generate-negative", "token-channel-train", "build-entropy-profile"]`
- `ALL_PHASES == PHASES + OPTIONAL_PHASES`

Verification:

- Remote `pytest tests/common/ tests/datasets/ tests/lang/ tests/evaluation/ tests/integration/test_cli_resolution.py tests/integration/test_orchestration.py -v --tb=short`: 228 passed, 4 warnings.
- Remote `pytest tests/encoder/ tests/ablation/ -v --tb=short`: 86 passed, 14 warnings.

Remaining config/CLI risks from the original audit are not fixed in this repair pass: JSON `encoder` config resolution, direct calibration script model loading, and evaluation auto-harness environment handling still need separate focused fixes.

## Verdict

Config and CLI behavior is only partially trustworthy. The parser and phase registry mostly match the documented surface, but there is a confirmed config-resolution gap in the encoder runner and several evaluation/ablation entry points bypass the same runtime contract that manual CLI execution uses.

## Evidence

- `run.py` is a thin shim: it forwards to `wfcllm.cli.entry:main` and re-exports `resolve_extract_lsh_params`; no business logic was moved back into `run.py`.
- Phase source of truth is consistent with docs: `wfcllm/orchestration/state.py:9-13` defines main phases `encoder`, `watermark`, `extract` and optional phases `pretrain`, `generate-negative`, `token-channel-train`, `build-entropy-profile`.
- Orchestrator default behavior matches docs: `wfcllm/orchestration/pipeline.py:28-34` runs one explicit phase or the default main phase list; `pipeline.py:58-67` skips completed phases unless forced/eval-only, and explicit extract input bypasses the skip.
- Config loading and bool parsing are centralized: `wfcllm/cli/config_resolver.py:14-20` parses explicit true/false strings; `config_resolver.py:23-32` returns empty config for a missing file and exits on invalid JSON.
- LSH extraction prefers embedded metadata: `wfcllm/cli/config_resolver.py:35-43` reads `watermark_params.lsh_d` and `watermark_params.lsh_gamma` from the first record before falling back to extract config.
- Compare-only mode is guarded in `wfcllm/cli/runners.py:35-45` and `_validate_compare_only_args` at `runners.py:79-91`.

## Confirmed Problems

- `run_encoder()` does not read the `encoder` section from the JSON config. `wfcllm/cli/runners.py:113-194` builds `EncoderConfig()` defaults and applies only CLI flags such as `--model-name`, `--embed-dim`, `--lr`, `--batch-size`, `--epochs`, `--margin`, `--no-lora`, and `--no-bf16`. This contradicts `docs/plans/2026-03-09-config-system.md`, which has a task for `run_encoder()` to use config dict. Current `configs/base_config.json` is mostly aligned with checkpoint metadata, so this is not sufficient by itself to explain the current poor result, but it can silently invalidate experiments using non-default encoder config values.
- The ablation runner builds an incomplete `argparse.Namespace`: `wfcllm/ablation/runner.py:119-134` sets `_config_cache`, `force`, `eval_only`, `resume`, `checkpoint`, `input_file`, and `config` only. Real phase runners read direct args such as `dataset`, `dataset_path`, `output_dir`, token-channel flags, and extract overrides. This can make ablation runs fail or unintentionally fall back to missing attributes.
- Evaluation auto-generation invokes bare `python run.py` instead of the required conda command: `wfcllm/evaluation/dual_channel.py:287-354` and `wfcllm/evaluation/benchmark.py:195-237`. Those harnesses set only `HF_HUB_OFFLINE` (`dual_channel.py:72-74`, `benchmark.py:186-188`) and omit `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE`. Manual commands used in this audit did not have that problem.

## Ruled Out / Low Risk

- Boolean CLI parsing itself is covered and passed targeted integration tests.
- Phase names are registered and accepted by the parser. The failing `tests/integration/test_orchestration.py::test_phases_order` is a stale expected-order test, not evidence that the current parser rejects phases.
- Missing `sklearn` is not a current production metric blocker because manual AUROC/FPR code is present and tests under `tests/evaluation/` passed.

## Required Follow-up Before Method Conclusions

Fix or explicitly gate the encoder config resolution, ablation runner arg synthesis, and evaluation harness environment. Until then, experiments launched through these paths are not equivalent to the documented conda/offline workflow.
