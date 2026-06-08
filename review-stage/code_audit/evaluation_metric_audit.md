# Evaluation Scripts and Metrics Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Verdict

Metric helper tests under `tests/evaluation/` pass, and manual AUROC/FPR implementations look coherent. However, evaluation auto-generation has environment drift risk, `exec --reference` is reference-code equality rather than true test execution, and current summary artifacts are not paired with run state. Therefore current report numbers are not reliable method-effect evidence.

## Code Review

- `scripts/evaluate.py` routes to `exec`, `detection`, `dual`, and `bench` subcommands.
- Manual AUROC uses pairwise ranking with tie half-credit at `wfcllm/evaluation/code_execution.py:99-112`.
- TPR at FPR scans thresholds and chooses best TPR at or below target FPR at `code_execution.py:115-135`.
- `annotate_correctness_from_references` at `code_execution.py:38-57` marks correctness by normalized generated-code equality with reference. This is not benchmark test execution. It is safe only when described as reference-code match, not pass@k correctness.
- Benchmark runner’s actual executor tests passed under `tests/evaluation/`.

## Environment Drift

- Dual-channel auto harness uses `python run.py` and only sets `HF_HUB_OFFLINE` (`wfcllm/evaluation/dual_channel.py:72-74`, `287-354`).
- Benchmark auto-generation uses `python run.py` and only sets `HF_HUB_OFFLINE` (`wfcllm/evaluation/benchmark.py:186-237`).
- The user and project docs require `conda run -n WFCLLM` and offline behavior; these harnesses do not enforce it.

## Existing Summary Risks

`data/results/humaneval_full_cap9_summary.json` is internally labeled with `input_file=data/watermarked/humaneval_full_cap9.jsonl`, but run state points to `data/watermarked/humaneval_20260523_130234.jsonl`. Report numbers sourced from run state are therefore pairing-invalid.

## Test Evidence

`tests/evaluation/ -v --tb=short`: 87 passed, 4 warnings. This rules out broad failures in the metric helper tests but does not rescue the current artifact pairing or detector nondeterminism issues.
