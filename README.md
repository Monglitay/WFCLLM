# WFCLLM: Structure-Aware Generation-Time Semantic Watermarking for Code LLMs

WFCLLM is a structure-aware generation-time semantic watermarking method for code LLMs. The default method preset is `evidence_retry_seed7x3`.

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

The default run root is `data/runs/`. Generated official detector inputs are written as `inputs/final_code.jsonl` with exactly `id`, `dataset`, `prompt`, and `final_code`.

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

## Mainline Phases

`run.py` is a thin entrypoint that forwards to `wfcllm.cli.entry:main`.

Default mainline phases:

- `generate`
- `calibrate`
- `detect`
- `report`
- `audit`

The default config is `configs/base_config.json`, which points to the official method preset and the `data/runs/` artifact root.

## Protocol Summary

Detector input is final code only. Audit rows, generation traces, retry decisions, candidate sidecars, and posthoc pass reports are not detector inputs.

Posthoc pass reports must carry the marker described in [docs/NO_QUALITY_GATE_PROTOCOL.md](docs/NO_QUALITY_GATE_PROTOCOL.md). They can be used to report utility after the run, but they cannot influence generation, retry, final selection, calibration, or detection.

Diagnostic selector outputs must be marked `diagnostic_only=true` and `not_official_method=true`; see [docs/DIAGNOSTIC_UPPER_BOUND.md](docs/DIAGNOSTIC_UPPER_BOUND.md).
