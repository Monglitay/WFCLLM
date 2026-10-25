# WFCLLM retained best results and configuration

Snapshot date: 2026-10-25 (Asia/Shanghai)

This file retains only complete, auditable 100-sample runs. Development probes,
counterfactual re-scores, and mixed-run metrics are excluded from the table.

| Profile | Retained run | Pass@1 | Primary AUROC | TPR @ calibrated FPR=5% | Independent AUROC | Independent actual FPR | Target triple |
|---|---|---:|---:|---:|---:|---:|---|
| Python / HumanEval | R3 | 80 | 78.24 | 56 | 77.125 | 9/100 | No |
| Python / MBPP | R6 `python-mbpp-encoder10-v6` | 37 | 95.17 | 72 | 95.000 | 4/100 | No |
| C++ / HumanEvalPack | R18 `cpp-humanevalpack-runtime-v18-bupt-v1` | 61 | 91.095 | 75 | 91.335 | 10/100 | No |
| Java / HumanEvalPack | v14 `java-humanevalpack-single-call-v14-bupt-v1` | 67 | 73.915 | 42 | 66.875 | 19/100 | No |
| JavaScript / HumanEvalPack | v6 `js-humanevalpack-structural-v6-bupt-v1` | 73 | 80.370 | 30 | 74.190 | 10/100 | No |

The target triple requires Pass@1, primary AUROC, TPR, and independent actual
FPR to pass in the same whole run. No profile currently meets it. The retained
run roots remain on the remote reproduction host; this repository stores only
the result summary and configuration identities.

## Retained canonical configuration

The production configuration remains the five canonical `full` profiles. All
use `gated_semantic_window_v1`, `lsh_d=12`, `lsh_gamma=0.25`, and fresh encoder
and Gate artifacts per run.

| Profile | Configuration file | SHA-256 |
|---|---|---|
| Python / HumanEval | `configs/wfcllm/experiments/python_humaneval_full.json` | `3AC0081E6388E8D6B27046EC0E36644F6618D19C9C02887E473819D1E8E185A0` |
| Python / MBPP | `configs/wfcllm/experiments/python_mbpp_full.json` | `E09EF1ABD0F567F675055B06819A90E24C4A8F8F6E9DC784124D3666A02CD425` |
| C++ / HumanEvalPack | `configs/wfcllm/experiments/cpp_humanevalpack_full.json` | `B4360E7545F99838577C2334B30DE9130BF89D803D7268DF1F1A25EBA90103E6` |
| Java / HumanEvalPack | `configs/wfcllm/experiments/java_humanevalpack_full.json` | `BC38D19EB13D8B7927522AAD8D4D3E56E9AB15D3A3C27D6E804175BADD23AFD2` |
| JavaScript / HumanEvalPack | `configs/wfcllm/experiments/js_humanevalpack_full.json` | `5156F14099DDB424562718470B4166178C012B598A066308E9B0FC8969FD7671` |

Shared implementation identity at snapshot time:

| File | SHA-256 |
|---|---|
| `wfcllm/method/presets.py` | `C3B0D0BD455ACA2DBC6568E721742CA003CB024A78505C49F29F3DD4B076E15E` |
| `wfcllm/encoder/projection_training.py` | `08A7674E53C8EC7D29EE60E42CD93CD9B128F354E8F4CB4D4E0E67878DA84432` |
| `wfcllm/generation/gated_generator.py` | `D6092FC67FD900086EBA7A533EE7D70D6998B539CB85641D6D401CC12DDBD0D1` |
| `wfcllm/detection/gated_pipeline.py` | `DE4D8B7C191A7D124B66261EC616871BD3B4D1356EA6449FE201A23F18D93760` |

No Git stage, commit, push, branch, or history rewrite was performed.
