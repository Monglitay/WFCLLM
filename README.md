# WFCLLM: Structure-Aware Generation-Time Semantic Watermarking for Code LLMs

WFCLLM is a structure-aware generation-time semantic watermarking method for code LLMs. The default and official baseline preset remains `evidence_retry_seed7x3`. The optional `gated_semantic_window_v1` preset is experimental; offline fake tests establish its implementation contract and wiring, not real gate effectiveness or detector improvement.

WFCLLM embeds semantic LSH evidence while code is generated, using structure-aware boundary detection, rollback-capable evidence-only retry, and strict final-code-only statistical detection. The official protocol forbids pass/test/correctness proxies during generation, retry, calibration, detection, and final candidate selection.

## Quick Start

Run commands from the repository root in the `WFCLLM` conda environment.

```bash
python run.py --status
python run.py --config configs/base_config.json
python run.py --phase generate
python run.py --phase detect --input data/runs/<run_id>/inputs/final_code.jsonl
python run.py --phase audit --run-dir data/runs/<run_id>
```

To select the experimental gated configuration, use
`--config configs/wfcllm/gated_semantic_window_v1.json`. Its default phase
sequence is `gate-data`, `gate-train`, `gate-validate`, `generate`, `calibrate`,
`detect`, `report`, `audit`. A separately validated external bundle may start
at `generate` and then uses the normal five main phases.

The default run root is `data/runs/`. Generated official detector inputs are written as `inputs/final_code.jsonl` with exactly `id`, `dataset`, `prompt`, and `final_code`.

The command block above is the official `run.py` workflow contract for the mainline phases. In this refactor branch, the phase runners currently establish phase names and run-state behavior; for direct artifact-producing generation while those runners are being wired to the full pipeline, use `scripts/wfcllm_generate.py` with the same preset values documented in [docs/REPRO_EVIDENCE_RETRY_SEED7X3.md](docs/REPRO_EVIDENCE_RETRY_SEED7X3.md).

## Remote Gated HumanEval Experiment

The current gated semantic HumanEval experiment is the no-carrier variant:
semantic evidence is embedded by the gated statement-window model and semantic
LSH rewrite path, not by `keyed_text_region` carrier text. The runnable full
HumanEval wrapper is:

```bash
scripts/experiments/run_python_humaneval_full.sh
```

On AutoDL-style servers, keep the repository and experiment state on the data
disk and use an explicit conda prefix. The server used for the latest run had
the environment at `/root/autodl-tmp/conda/envs/WFCLLM`.

```bash
cd /root/autodl-tmp
git clone <repo-url> WFCLLM
cd /root/autodl-tmp/WFCLLM

/root/miniconda3/bin/conda create -p /root/autodl-tmp/conda/envs/WFCLLM python=3.11 -y
/root/miniconda3/bin/conda run -p /root/autodl-tmp/conda/envs/WFCLLM pip install -r requirements.txt
```

All model, dataset, source-catalog, negative-corpus, and key material must be
local. The runner does not download these resources and does not write private
keys into public configs or reports.

```bash
export CONDA_EXE=/root/miniconda3/bin/conda
export CONDA_ENV_PREFIX=/root/autodl-tmp/conda/envs/WFCLLM

export GENERATION_MODEL_PATH=/root/autodl-tmp/models/<local-code-generation-model>
export REWRITE_MODEL_PATH="${GENERATION_MODEL_PATH}"
export SEMANTIC_ENCODER_MODEL_PATH=/root/autodl-tmp/models/<local-semantic-encoder>
export GATE_BASE_MODEL_PATH=/root/autodl-tmp/models/codet5-small

export DATASET_PATH=/root/autodl-tmp/data/datasets
export PILOT_SOURCE_CATALOG=/root/autodl-tmp/data/gate_sources/python_humaneval_pilot.jsonl
export FULL_SOURCE_CATALOG=/root/autodl-tmp/data/gate_sources/python_humaneval_full.jsonl
export NEGATIVE_INPUT=/root/autodl-tmp/data/negative/humaneval_negative_final_code.jsonl

export EXPERIMENT_ROOT=/root/autodl-tmp/wfcllm-runs/python-humaneval-full
export SAMPLE_LIMIT=164
export CALIBRATION_LIMIT=32
export GATE_EPOCHS=3

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  scripts/experiments/run_python_humaneval_full.sh
```

For a full run, `PILOT_SOURCE_CATALOG` and `FULL_SOURCE_CATALOG` must be
different files. The wrapper enforces this because pilot gate feasibility and
the full embedding run must not share the same source-catalog split.

The run writes the formal artifacts under:

```text
${EXPERIMENT_ROOT}/run/
  inputs/final_code.jsonl
  calibration/reference_calibration.json
  detection/
  reports/
  audit/
```

Posthoc utility and diagnostic summaries are computed only after generation,
calibration, detection, report, and audit complete. They must not feed back into
generation, retry, final-code selection, calibration, or detection.

The latest saved server summary for this experiment was:

```text
/tmp/gamma25_gate_pass113/humaneval_summary.json
/tmp/gamma25_gate_pass113/humaneval_results.jsonl
/tmp/gamma25_gate_pass113/n_ge_3_tpr_summary.json
```

Those files reported full HumanEval single-candidate Pass@1 as
`113/164 = 68.9024%`. The conditional diagnostic TPR with eligibility
`reliable_window_count >= 3` and threshold `hit_rate >= 0.60` was
`80/130 = 61.5385%`; `34/164` positives were outside that conditional
denominator. The heldout negative diagnostic count at that threshold was
`4/128` false positives. This thresholded `n >= 3` TPR is a saved diagnostic
summary, not the official detector calibration claim.

## Official Method

The main method preset is documented in [docs/WFCLLM_METHOD.md](docs/WFCLLM_METHOD.md). Reproduction notes for the default preset are in [docs/REPRO_EVIDENCE_RETRY_SEED7X3.md](docs/REPRO_EVIDENCE_RETRY_SEED7X3.md).

Core protocol references:

- [No Quality Gate Protocol](docs/NO_QUALITY_GATE_PROTOCOL.md)
- [Strict Code-Only Detector](docs/STRICT_CODE_ONLY_DETECTOR.md)
- [Artifact Schema](docs/ARTIFACT_SCHEMA.md)
- [Diagnostic Upper Bound](docs/DIAGNOSTIC_UPPER_BOUND.md)
- [Legacy Archive](docs/LEGACY_ARCHIVE.md)

Known metrics for the current official preset are intentionally modest. Detector evidence remains the main limitation, and the preset is the best official single-configuration generation-time candidate rather than a solved detector result.

## Repository Layout

```text
wfcllm/
  method/
  generation/
  semantic/
  detection/
  audit/
  diagnostics/
  cli/
  orchestration/
  datasets/
  lang/
  encoder/
  common/
configs/
  base_config.json
  wfcllm/
scripts/
  wfcllm_generate.py
  wfcllm_sanitize_final_code.py
  evaluate.py
tests/
docs/
archive/
```

The live package is `wfcllm/`. Archived legacy code and historical documentation are kept under `archive/` and `docs/archive/` for traceability.

## Environment

Use the conda environment named `WFCLLM`.

```bash
conda create -n WFCLLM python=3.11 -y
conda run -n WFCLLM pip install torch
conda run -n WFCLLM pip install -r requirements.txt
```

Development and tests should assume offline execution unless a user explicitly requests online behavior.

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

For model-sensitive tests, prefer local resources under `data/models/` and `data/datasets/`.

The gated workflow requires all resources locally: a supported base generation
model, the semantic encoder, `data/models/codet5-small`, source datasets, a
private 32-key training bank, a disjoint private 8-key holdout bank, and a
deployment key supplied through the CLI's file/environment options. It never
downloads models or datasets automatically. Keys stay outside configs and
public artifacts; only identifiers and hashes may be published.

## Mainline Phases

`run.py` is a thin entrypoint that forwards to `wfcllm.cli.entry:main`.

Default mainline phases:

- `generate`
- `calibrate`
- `detect`
- `report`
- `audit`

The experimental gated preset prepends three phases:

- `gate-data` builds provenance-bound, no-quality-proxy supervision.
- `gate-train` trains the close/suitable gate with the fixed six-loss contract.
- `gate-validate` fits thresholds, validates float/int8 stability, and is the
  only formal gate-bundle publisher.

The default config is `configs/base_config.json`, which points to the official method preset and the `data/runs/` artifact root.

## Protocol Summary

Detector input is final code only. Audit rows, generation traces, retry decisions, candidate sidecars, and posthoc pass reports are not detector inputs.

Posthoc pass reports must carry the marker described in [docs/NO_QUALITY_GATE_PROTOCOL.md](docs/NO_QUALITY_GATE_PROTOCOL.md). They can be used to report utility after the run, but they cannot influence generation, retry, final selection, calibration, or detection.

Diagnostic selector outputs must be marked `diagnostic_only=true` and `not_official_method=true`; see [docs/DIAGNOSTIC_UPPER_BOUND.md](docs/DIAGNOSTIC_UPPER_BOUND.md).

Do not commit `data/runs/`, gate datasets, key banks, models, checkpoints,
bundles, generated JSONL, logs, or other experiment artifacts. The repository
tracks contracts, code, small configs, tests, and documentation only.
