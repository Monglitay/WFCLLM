# Reproduction Environment Checklist

Executable provisioning list for reproducing the Gated Lineage
Reproduction Targets on a fresh machine. Captured read-only from
server A (AutoDL) on 2026-07-26 before its expected disappearance.

## Interpreter and key dependencies

Conda env `WFCLLM`, Python 3.11.

```text
torch==2.11.0+cu130        (CUDA 13.0)
transformers==4.46.3
tokenizers==0.20.3
datasets==4.6.1
accelerate==1.14.0
numpy==2.4.2
scipy==1.17.1
pandas==3.0.1
peft==0.13.2
safetensors==0.8.0
sentencepiece==0.2.2
bitsandbytes==0.49.2
tree-sitter==0.25.2
tree-sitter-cpp==0.23.4
tree-sitter-java==0.23.5
tree-sitter-python==0.25.0
pytest==9.0.2
```

Note: the server never had a `tree-sitter-javascript` wheel installed;
`wfcllm/lang/js/parser.py` implements JS statement parsing without it.

Historical hardware: 2x NVIDIA GeForce RTX 5090. Gate bundles are
retrained on reproduction (ADR 0002), so exact GPU parity is not
required; metric drift from retraining is accepted by the acceptance
bar (same magnitude, same conclusion).

## Models (re-download; never committed)

| Identifier | Revision / source | Role |
| --- | --- | --- |
| deepseek-ai/deepseek-coder-1.3b-instruct | e063262dac8366fc1f28a4da0ff3c50ea66259ca | generation (JS, Java, C++, gamma25) |
| deepseek-ai/deepseek-coder-7b-instruct-v1.5 | 2a050a4c59d687a85324d32e147517992117ed30 | posthoc regeneration (JS 84/164), MBPP variants |
| OpenCoder-8B-Instruct | local copy under models/ | generation (OpenCoder target) |
| OpenCoder-1.5B-Instruct | local copy under models/ | auxiliary |
| Salesforce/codet5-small | local copy under models/ | gate base + semantic encoder base (run scripts) |
| Salesforce/codet5-base | local copy under data/models/ | semantic encoder base (gamma25 driver) |
| qwen2.5-coder-0.5b | local copy under models/ | auxiliary |

Expected local layout (override via CLI/env): `data/models/<name>`.

## Datasets (re-download; never committed)

- HumanEvalPack (bigcode/humanevalpack), HF datasets cache fingerprint
  `9a41762f73a8cb23bb5811b73d5aab164efcf378` (cpp and java configs,
  version 0.0.0). The JS split was stored as local JSONL under
  `data/datasets/humanevalpack/js/` (adapter reads `<lang>/test.jsonl`
  or `<lang>.jsonl`).
- HumanEval (openai_humaneval) and MBPP (google-research-datasets/mbpp)
  via their standard HF checkouts under `data/datasets/`.

## Offline environment variables

```bash
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Redirect caches to a large local volume as needed:
export HF_HOME=<volume>/huggingface
export HF_HUB_CACHE=<volume>/huggingface/hub
export HF_DATASETS_CACHE=<volume>/huggingface/datasets
export TORCH_HOME=<volume>/torch
export PIP_CACHE_DIR=<volume>/pip-cache
```

## Key material (kept outside the repository)

The Deployment Key is a symmetric watermark key shared by embedding and
detection; it cannot be regenerated. Key material is never committed -
runs receive it via the existing injection channels
(`--secret-key-file`, `--training-key-bank-file`,
`--holdout-key-bank-file`, or `WFCLLM_SECRET_KEY`). The local rescue
convention stores server A key material under `~/.wfcllm-secrets/`
mirroring the server paths (mode 600), including per-run `private/`
directories (deployment.key + 32-key training bank + 8-key holdout
bank).

Key Identifiers (sha256 of the deployment key) recorded per target:

| Reproduction Target | key_identifier_sha256 |
| --- | --- |
| HumanEvalPack-JS full164 | 5c60c72008e4feff09f5e50d66ae6fc4159ff42f922cd1242d92e07d23eb4248 |
| HumanEval-Python OpenCoder-8B full164 | 245d516cdec1f41c37149215da140a2c7ca101c9a2cfe8a2338483b27e33f064 |
| MBPP full interface-aware | bbfb7c680e2a9a5a1549c36e397c8a06b86eb19a5cf1c740373290035ee213c2 |
| HumanEvalPack-Java full164 | 736156fc5d7a75931d197f3fa3374d78b52b41acae6e83c6cae1c4356f9f3b0d |
| HumanEvalPack-C++ full164 | c13ea01c9bbef0fa35df0f0f0234dccc4a12f3d6e288726a67d786e3bcfa1c64 |
| C++ basic full | none (no-watermark wiring run) |
| HumanEval-Python gamma25 | not recorded in rescued artifacts (key at gated-extreme run `private/full/deployment.key`) |

## Reproduction entry points

- Matrix runs: `scripts/experiments/run_<language>_<dataset>_<profile>.sh`
  via the shared `run_gated_experiment.sh` (trains the per-dataset
  semantic encoder, then gate-data, gate-train, generate, calibrate,
  detect, report; audit on full profile). Historical launch parameter
  records live in the rescued `run_*.sh` scripts under
  `~/wfcllm-rescue/run-side/`.
- Metric rows: `scripts/wfcllm_metric_contract.py <run-dir>...` emits
  one self-identifying JSONL row per run, unconditionally.
- Gate bundles and the shared historical encoder checkpoint are not
  archived (ADR 0002); rerun gate-train / the encoder phase instead.
  Historical `gate_bundle_sha256` values identify vanished artifacts
  and will not match retrained bundles.
